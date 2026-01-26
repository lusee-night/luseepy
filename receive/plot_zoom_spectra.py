import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys

def plot_zoom_spectra(hdf5_file):
    """Plot zoom spectra from HDF5 file."""

    with h5py.File(hdf5_file, 'r') as f:
        print(f"Opened {hdf5_file}")
        print(f"Number of groups: {f.attrs.get('n_groups', 0)}")
        print()

        # Find groups with zoom spectra
        groups_with_zoom = []
        for group_name in f.keys():
            if 'calibrator/zoom_spectra' in f[group_name]:
                zoom_group = f[group_name]['calibrator/zoom_spectra']
                n_packets = zoom_group.attrs.get('count', 0)
                if n_packets > 0:
                    groups_with_zoom.append((group_name, n_packets))
                    print(f"{group_name}: {n_packets} zoom spectra packets")

        if not groups_with_zoom:
            print("No zoom spectra found in this file!")
            return

        # Plot zoom spectra from the first group with data
        group_name, n_packets = groups_with_zoom[0]
        zoom_group = f[group_name]['calibrator/zoom_spectra']

        # Read the data
        ch1_autocorr = zoom_group['ch1_autocorr'][:]
        ch2_autocorr = zoom_group['ch2_autocorr'][:]
        ch1_2_corr_real = zoom_group['ch1_2_corr_real'][:]
        ch1_2_corr_imag = zoom_group['ch1_2_corr_imag'][:]
        timestamps = zoom_group['timestamps'][:]
        pfb_indices = zoom_group['pfb_indices'][:]
        unique_ids = zoom_group['unique_ids'][:]

        # Frequency axis (0-63 for 64 FFT bins)
        freq_bins = np.arange(64)

        # Create figure with subplots
        fig = plt.figure(figsize=(16 * 0.9, 12 * 0.9))
        gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Select which packets to plot (first, middle, last)
        indices_to_plot = [0, 10, n_packets//2, n_packets - 10, n_packets-1] if n_packets > 15 else list(range(n_packets))

        # Plot 1: Channel 1 autocorrelation
        ax1 = fig.add_subplot(gs[0, 0])
        for idx in indices_to_plot:
            label = f"Packet {idx} (t={timestamps[idx]:.1f}s, pfb={pfb_indices[idx]})"
            ax1.semilogy(freq_bins, ch1_autocorr[idx], label=label, alpha=0.9, linewidth=0.6)
        ax1.set_xlabel('Frequency bin')
        ax1.set_ylabel('Ch1 Autocorr Power')
        ax1.set_title(f'Channel 1 Autocorrelation ({group_name})')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Channel 2 autocorrelation
        ax2 = fig.add_subplot(gs[0, 1])
        for idx in indices_to_plot:
            label = f"Packet {idx} (t={timestamps[idx]:.1f}s, pfb={pfb_indices[idx]})"
            ax2.semilogy(freq_bins, ch2_autocorr[idx], label=label, alpha=0.7, linewidth=0.6)
        ax2.set_xlabel('Frequency bin')
        ax2.set_ylabel('Ch2 Autocorr Power')
        ax2.set_title(f'Channel 2 Autocorrelation ({group_name})')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Cross-correlation REAL part
        ax3 = fig.add_subplot(gs[1, 0])
        for idx in indices_to_plot:
            label = f"Packet {idx}"
            ax3.plot(freq_bins, ch1_2_corr_real[idx], label=label, alpha=0.7, linewidth=0.6)
        ax3.set_xlabel('Frequency bin')
        ax3.set_ylabel('Real Part')
        ax3.set_title('Ch1-Ch2 Cross-correlation (Real)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Plot 4: Cross-correlation IMAGINARY part
        ax4 = fig.add_subplot(gs[1, 1])
        for idx in indices_to_plot:
            label = f"Packet {idx}"
            ax4.plot(freq_bins, ch1_2_corr_imag[idx], label=label, alpha=0.7, linewidth=0.6)
        ax4.set_xlabel('Frequency bin')
        ax4.set_ylabel('Imaginary Part')
        ax4.set_title('Ch1-Ch2 Cross-correlation (Imaginary)')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)

        # Plot 5: Cross-correlation amplitude
        ax5 = fig.add_subplot(gs[2, 0])
        for idx in indices_to_plot:
            cross_amp = np.sqrt(ch1_2_corr_real[idx]**2 + ch1_2_corr_imag[idx]**2)
            label = f"Packet {idx}"
            ax5.semilogy(freq_bins, cross_amp, label=label, alpha=0.7, linewidth=0.6)
        ax5.set_xlabel('Frequency bin')
        ax5.set_ylabel('Cross-corr Amplitude')
        ax5.set_title('Ch1-Ch2 Cross-correlation Amplitude')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

        # Plot 6: Cross-correlation phase
        ax6 = fig.add_subplot(gs[2, 1])
        for idx in indices_to_plot:
            phase = np.arctan2(ch1_2_corr_imag[idx], ch1_2_corr_real[idx]) + 1
            label = f"Packet {idx}"
            ax6.semilogy(freq_bins, phase - np.min(phase) + 0.001, label=label, alpha=0.7, linewidth=0.6)
        ax6.set_xlabel('Frequency bin')
        ax6.set_ylabel('Phase (degrees)')
        ax6.set_title('Ch1-Ch2 Cross-correlation Phase')
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-np.pi, np.pi)

        # Plot 7: Waterfall plot of Ch1 autocorr
        ax7 = fig.add_subplot(gs[3, :])
        im = ax7.imshow(np.log10(ch1_autocorr + 1e-10), aspect='auto',
                       cmap='viridis', origin='lower')
        ax7.set_xlabel('Frequency bin')
        ax7.set_ylabel('Packet number')
        ax7.set_title(f'Channel 1 Autocorr Evolution (log scale, {n_packets} packets)')
        cbar = plt.colorbar(im, ax=ax7)
        cbar.set_label('log10(Power)')

        # Add PFB indices as text if not too many packets
        if n_packets <= 20:
            for i in range(n_packets):
                ax7.text(-2, i, f"{pfb_indices[i]}", fontsize=6, ha='right')

        plt.suptitle(f'Zoom Spectra from {hdf5_file}', fontsize=14)
        plt.tight_layout()
        plt.show()

        # Print some statistics
        print("\n--- Statistics ---")
        print(f"Timestamps range: {timestamps.min():.2f} to {timestamps.max():.2f} seconds")
        print(f"Unique PFB indices: {np.unique(pfb_indices)}")
        print(f"Unique packet IDs range: {unique_ids.min()} to {unique_ids.max()}")

        # Check for any issues
        if np.any(ch1_autocorr < 0) or np.any(ch2_autocorr < 0):
            print("\nWARNING: Negative values in autocorrelation!")

        # Compute and print coherence
        coherence = np.abs(ch1_2_corr_real + 1j*ch1_2_corr_imag)**2 / (ch1_autocorr * ch2_autocorr + 1e-10)
        mean_coherence = np.mean(coherence, axis=1)
        print(f"\nMean coherence per packet: min={mean_coherence.min():.3f}, max={mean_coherence.max():.3f}")

        # Print statistics on real and imaginary parts
        print(f"\nCross-correlation Real part range: [{ch1_2_corr_real.min():.2e}, {ch1_2_corr_real.max():.2e}]")
        print(f"Cross-correlation Imag part range: [{ch1_2_corr_imag.min():.2e}, {ch1_2_corr_imag.max():.2e}]")

        return ch1_autocorr, ch2_autocorr, ch1_2_corr_real, ch1_2_corr_imag, timestamps, pfb_indices


def compare_zoom_spectra_across_groups(hdf5_file):
    """Compare zoom spectra across different metadata groups."""

    with h5py.File(hdf5_file, 'r') as f:
        groups_with_zoom = []

        for group_name in f.keys():
            if 'calibrator/zoom_spectra' in f[group_name]:
                zoom_group = f[group_name]['calibrator/zoom_spectra']
                n_packets = zoom_group.attrs.get('count', 0)
                if n_packets > 0:
                    groups_with_zoom.append(group_name)

        if len(groups_with_zoom) < 2:
            print("Need at least 2 groups with zoom spectra for comparison")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, group_name in enumerate(groups_with_zoom[:4]):  # Compare up to 4 groups
            zoom_group = f[group_name]['calibrator/zoom_spectra']
            ch1_autocorr = zoom_group['ch1_autocorr'][:]

            # Average over all packets in this group
            mean_spectrum = np.mean(ch1_autocorr, axis=0)
            std_spectrum = np.std(ch1_autocorr, axis=0)

            ax = axes[i] if i < 4 else axes[3]
            freq_bins = np.arange(64)

            ax.semilogy(freq_bins, mean_spectrum, 'b-', label='Mean')
            ax.fill_between(freq_bins,
                           mean_spectrum - std_spectrum,
                           mean_spectrum + std_spectrum,
                           alpha=0.3, color='blue', label='±1 std')

            ax.set_xlabel('Frequency bin')
            ax.set_ylabel('Ch1 Power')
            ax.set_title(f'{group_name} (n={ch1_autocorr.shape[0]} packets)')
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.suptitle('Zoom Spectra Comparison Across Metadata Groups')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_zoom_spectra.py <hdf5_file>")
        print("Example: python plot_zoom_spectra.py session_000_20241105_112220.h5")
        sys.exit(1)

    hdf5_file = sys.argv[1]

    # Plot zoom spectra
    data = plot_zoom_spectra(hdf5_file)

    # Compare across groups if available
    compare_zoom_spectra_across_groups(hdf5_file)