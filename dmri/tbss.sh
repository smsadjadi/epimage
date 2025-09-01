#!/usr/bin/env bash
set -euo pipefail

# -------------------- defaults (override via flags or env) --------------------
DATASET="${DATASET:-/home/ubuntu/Github/codex/datasets/dti}"
TBSSDIR="${TBSSDIR:-$DATASET/tbss}"
SUBJECTS="${SUBJECTS:-}"
TBSS_THR="${TBSS_THR:-0.2}"
GROUPS_TSV="${GROUPS_TSV:-}"

# -------------------- parse flags --------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)   DATASET="$2"; shift 2;;
    --tbssdir)   TBSSDIR="$2"; shift 2;;
    --subjects)  SUBJECTS="$2"; shift 2;;
    --thr)       TBSS_THR="$2"; shift 2;;
    --groups)    GROUPS_TSV="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 2;;
  esac
done

# -------------------- checks --------------------
command -v tbss_1_preproc >/dev/null || { echo "FSL TBSS not found in PATH"; exit 1; }
mkdir -p "$TBSSDIR"
cd "$TBSSDIR"

# -------------------- collect subjects --------------------
if [[ -z "${SUBJECTS}" ]]; then
  mapfile -t SUBJ_ARR < <(find "$DATASET" -maxdepth 1 -mindepth 1 -type d -printf "%f\n" \
                          | sort)
else
  read -r -a SUBJ_ARR <<< "$SUBJECTS"
fi
if [[ ${#SUBJ_ARR[@]} -eq 0 ]]; then
  echo "No subjects found. Check --dataset or --subjects."; exit 1
fi

echo "TBSS workspace: $TBSSDIR"
echo "Dataset root  : $DATASET"
echo "Subjects      : ${SUBJ_ARR[*]}"
echo "Skel thr      : $TBSS_THR"

# -------------------- link / compute inputs --------------------
mkdir -p MD AD RD
shopt -s nullglob

for s in "${SUBJ_ARR[@]}"; do
  SROOT="$DATASET/$s/analyzed_fsl"
  FA="$SROOT/dti_FA.nii.gz"
  MD="$SROOT/dti_MD.nii.gz"
  L1="$SROOT/dti_L1.nii.gz"
  L2="$SROOT/dti_L2.nii.gz"
  L3="$SROOT/dti_L3.nii.gz"

  [[ -f "$FA" ]] || { echo "Missing FA for $s ($FA)"; exit 1; }
  ln -sf "$FA" "${s}_FA.nii.gz"

  if [[ -f "$MD" ]]; then
    ln -sf "$MD" "MD/${s}_MD.nii.gz"
  elif [[ -f "$L1" && -f "$L2" && -f "$L3" ]]; then
    echo "Deriving MD for $s from L1/L2/L3"
    fslmaths "$L1" -add "$L2" -add "$L3" -div 3 "MD/${s}_MD.nii.gz"
  else
    echo "No MD or eigenvalues for $s"; exit 1
  fi

  # AD = L1
  if [[ -f "$L1" ]]; then
    ln -sf "$L1" "AD/${s}_AD.nii.gz"
  else
    echo "Missing L1 for AD in $s ($L1)"; exit 1
  fi

  # RD = (L2 + L3) / 2
  if [[ -f "$L2" && -f "$L3" ]]; then
    fslmaths "$L2" -add "$L3" -div 2 "RD/${s}_RD.nii.gz"
  else
    echo "Missing L2/L3 for RD in $s ($L2 / $L3)"; exit 1
  fi
done

# -------------------- TBSS steps for FA --------------------
echo ">>> TBSS: FA preprocessing"
tbss_1_preproc *_FA.nii.gz

echo ">>> TBSS: registration to FMRIB58 (FNIRT via -T)"
tbss_2_reg -T

echo ">>> TBSS: postreg (create mean FA skeleton in standard space)"
tbss_3_postreg -S

echo ">>> TBSS: projecting FA onto skeleton @ thr=${TBSS_THR}"
tbss_4_prestats "$TBSS_THR"

# -------------------- TBSS for non-FA (MD / AD / RD) --------------------
for MOD in MD AD RD; do
  if compgen -G "${MOD}/*_${MOD}.nii.gz" > /dev/null; then
    echo ">>> TBSS: projecting ${MOD} onto FA skeleton"
    tbss_non_FA "${MOD}"
  else
    echo "No ${MOD} images found; skipping tbss_non_FA ${MOD}"
  fi
done

# -------------------- Optional: simple two-sample stats (FA only) --------------------
if [[ -n "${GROUPS_TSV}" && -f "${GROUPS_TSV}" ]]; then
  echo ">>> Stats: building two-sample design from ${GROUPS_TSV}"
  ALL_FA_4D="stats/all_FA_skeletonised.nii.gz"
  MASK="stats/mean_FA_skeleton_mask.nii.gz"
  [[ -f "$ALL_FA_4D" && -f "$MASK" ]] || { echo "TBSS FA outputs missing"; exit 1; }

  ls *_FA.nii.gz | sed 's/_FA\.nii\.gz$//' | sort > tbss_subjects.txt

  rm -f group1.txt group2.txt
  while IFS=$'\t' read -r subj grp; do
    [[ -z "$subj" || "$subj" == subject ]] && continue
    if [[ "$grp" == "1" ]]; then echo "$subj" >> group1.txt
    elif [[ "$grp" == "2" ]]; then echo "$subj" >> group2.txt
    else echo "Invalid group for $subj: $grp"; exit 1; fi
  done < "$GROUPS_TSV"

  N1=$(wc -l < group1.txt || echo 0)
  N2=$(wc -l < group2.txt || echo 0)
  [[ "$N1" -gt 0 && "$N2" -gt 0 ]] || { echo "Need >=1 subject in each group"; exit 1; }

  design_ttest2 design $N1 $N2

  echo ">>> Running randomise (TFCE, 5000 perms) on FA skeleton"
  randomise -i "$ALL_FA_4D" -o stats/tbss_FA \
            -m "$MASK" -d design.mat -t design.con \
            -n 5000 --T2 -V
fi

echo "Done. Results in: $TBSSDIR"
echo "Key outputs:"
echo "  stats/all_FA_skeletonised.nii.gz"
echo "  stats/all_MD_skeletonised.nii.gz (if MD present)"
echo "  stats/all_AD_skeletonised.nii.gz (if AD present)"
echo "  stats/all_RD_skeletonised.nii.gz (if RD present)"
