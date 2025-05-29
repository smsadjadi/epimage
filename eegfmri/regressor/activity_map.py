import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def activity_map(eeg_data, spike_timing):
    
    spike_data = []
    window = 50

    for t in spike_timing:
        if t - window >= 0 and t + window < eeg_data.shape[1]:
            spike_data.append(eeg_data[:, t - window:t + window])

    spike_data = np.array(spike_data)
    mean_spike = np.mean(spike_data, axis=0)

    # Normalize by Global Field Power (GFP)
    gfp = np.std(mean_spike, axis=0)
    peak_idx = np.argmax(gfp)
    epileptic_map = mean_spike[:, peak_idx]
    epileptic_map /= np.linalg.norm(epileptic_map)  # Normalize to unit norm

    return epileptic_map, mean_spike, gfp, peak_idx


def generate_topography(activity_values, electrode_positions, plot=True):
    
    if len(activity_values) != len(electrode_positions):
        raise ValueError("Mismatch between activity values and electrode positions.")

    x, y = zip(*electrode_positions.values())
    x, y = np.array(x), np.array(y)

    grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
    z = griddata((x, y), activity_values, (grid_x, grid_y), method='cubic')

    mask = np.sqrt(grid_x**2 + grid_y**2) > 1
    z[mask] = np.nan
    
    if plot:
        plt.contourf(grid_x, grid_y, z, levels=50, cmap='RdBu_r')
        plt.colorbar(label='Activity')
        plt.scatter(x, y, c=activity_values, cmap='RdBu_r', edgecolor='k', s=100)
        
        circle = plt.Circle((0, 0), 1, color='k', fill=False)
        plt.gca().add_artist(circle)

        plt.axis('off')
        plt.title('Activation Topography')

    return z