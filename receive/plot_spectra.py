#!/usr/bin/env python3
"""
Script to plot normal spectra from LuSEE HDF5 files.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from typing import Tuple, List, Optional

# Product mapping
# Products 0-3: autocorrelations (ch0, ch1, ch2, ch3)
# Products 4-15: cross-correlations
PRODUCT_NAMES = {
    0: 'Ch0 Auto', 1: 'Ch1 Auto', 2: 'Ch2 Auto', 3: 'Ch3 Auto',
    4: 'Ch0×Ch1 Re', 5: 'Ch0×Ch1 Im', 6: 'Ch0×Ch2 Re', 7: 'Ch0×Ch2 Im',
    8: 'Ch0×Ch3 Re', 9: 'Ch0×Ch3 Im', 10: 'Ch1×Ch2 Re', 11: 'Ch1×Ch2 Im',
    12: 'Ch1×Ch3 Re', 13: 'Ch1×Ch3 Im', 14: 'Ch2×Ch3 Re', 15: 'Ch2×Ch3 Im'
}

def load_spectra(hdf5_file: str, group_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load spectra data from HDF5 file."""
    with h5py.File(hdf5_file, 'r') as f:
        groups = list(f.keys())
        if group_idx >= len(groups):
            raise ValueError(f"Group index {group_idx} out of range. File has {len(groups)} groups.")

        group = f[groups[group_idx]]

        if 'spectra/data' not in group:
            raise ValueError(f"No spectra data in group {groups[group_idx]}")

        spectra = group['spectra/data'][:]
        unique_ids = group['spectra/unique_ids'][:]

        # Get metadata
        metadata = {}
        for key in ['Navgf', 'Navg1_shift', 'Navg2_shift', 'corr_products_mask', 'format']:
            if key in group.attrs:
                metadata[key] = group.attrs[key]

        return spectra, unique_ids, metadata

def plot_autocorrelations(spectra: np.ndarray, unique_ids: np.ndarray, title_suffix: str = ""):
    """Plot autocorrelation spectra for all 4 channels."""
    n_time, n_products, n_freq = spectra.shape
    freq_bins = np.arange(n_freq)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ch in range(4):
        ax = axes[ch]

        # Plot a few time samples
        time_indices = [0, n_time//2, n_time-1] if n_time > 2 else list(range(n_time))

        for t_idx in time_indices:
            spectrum = spectra[t_idx, ch, :]
            # Filter out NaN and zero values for log plot
            valid = (spectrum > 0) & ~np.isnan(spectrum)
            if np.any(valid):
                label = f"t={unique_ids[t_idx]}"
                ax.semilogy(freq_bins[valid], spectrum[valid],
                           alpha=0.7, label=label)

        ax.set_xlabel('Frequency bin')
        ax.set_ylabel('Power')
        ax.set_title(f'Channel {ch} Autocorrelation')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(0, n_freq)

    plt.suptitle(f'Autocorrelation Spectra{title_suffix}')
    plt.tight_layout()
    plt.show()

def plot_cross_correlations(spectra: np.ndarray, unique_ids: np.ndarray,
                           ch1: int = 0, ch2: int = 1, title_suffix: str = ""):
    """Plot cross-correlation between two channels."""
    n_time, n_products, n_freq = spectra.shape
    freq_bins = np.arange(n_freq)

    # Find the product indices for this channel pair
    # Cross products start at index 4
    # Mapping: (0,1)->4,5, (0,2)->6,7, (0,3)->8,9, (1,2)->10,11, (1,3)->12,13, (2,3)->14,15
    if ch1 > ch2:
        ch1, ch2 = ch2, ch1

    product_map = {
        (0,1): (4, 5), (0,2): (6, 7), (0,3): (8, 9),
        (1,2): (10, 11), (1,3): (12, 13), (2,3): (14, 15)
    }

    if (ch1, ch2) not in product_map:
        print(f"Invalid channel pair: {ch1}, {ch2}")
        return

    real_idx, imag_idx = product_map[(ch1, ch2)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select time samples to plot
    time_indices = [0, n_time//2, n_time-1] if n_time > 2 else list(range(n_time))

    # Plot 1: Real part
    ax1 = axes[0, 0]
    for t_idx in time_indices:
        real_part = spectra[t_idx, real_idx, :]
        valid = ~np.isnan(real_part)
        if np.any(valid):
            ax1.plot(freq_bins[valid], real_part[valid],
                    alpha=0.7, label=f"t={unique_ids[t_idx]}")
    ax1.set_xlabel('Frequency bin')
    ax1.set_ylabel('Real Part')
    ax1.set_title(f'Ch{ch1}×Ch{ch2} Cross-correlation (Real)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Imaginary part
    ax2 = axes[0, 1]
    for t_idx in time_indices:
        imag_part = spectra[t_idx, imag_idx, :]
        valid = ~np.isnan(imag_part)
        if np.any(valid):
            ax2.plot(freq_bins[valid], imag_part[valid],
                    alpha=0.7, label=f"t={unique_ids[t_idx]}")
    ax2.set_xlabel('Frequency bin')
    ax2.set_ylabel('Imaginary Part')
    ax2.set_title(f'Ch{ch1}×Ch{ch2} Cross-correlation (Imaginary)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot 3: Amplitude
    ax3 = axes[1, 0]
    for t_idx in time_indices:
        real_part = spectra[t_idx, real_idx, :]
        imag_part = spectra[t_idx, imag_idx, :]
        amplitude = np.sqrt(real_part**2 + imag_part**2)
        valid = ~np.isnan(amplitude) & (amplitude > 0)
        if np.any(valid):
            ax3.semilogy(freq_bins[valid], amplitude[valid],
                        alpha=0.7, label=f"t={unique_ids[t_idx]}")
    ax3.set_xlabel('Frequency bin')
    ax3.set_ylabel('Amplitude')
    ax3.set_title(f'Ch{ch1}×Ch{ch2} Cross-correlation Amplitude')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # Plot 4: Phase
    ax4 = axes[1, 1]
    for t_idx in time_indices:
        real_part = spectra[t_idx, real_idx, :]
        imag_part = spectra[t_idx, imag_idx, :]
        phase = np.arctan2(imag_part, real_part)
        valid = ~np.isnan(phase)
        if np.any(valid):
            ax4.plot(freq_bins[valid], np.degrees(phase[valid]),
                    alpha=0.7, label=f"t={unique_ids[t_idx]}")
    ax4.set_xlabel('Frequency bin')
    ax4.set_ylabel('Phase (degrees)')
    ax4.set_title(f'Ch{ch1}×Ch{ch2} Cross-correlation Phase')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    ax4.set_ylim(-180, 180)

    plt.suptitle(f'Cross-correlation Spectra Ch{ch1}×Ch{ch2}{title_suffix}')
    plt.tight_layout()
    plt.show()

def plot_waterfall(spectra: np.ndarray, unique_ids: np.ndarray,
                  product: int = 0, title_suffix: str = ""):
    """Plot waterfall diagram for a specific product."""
    n_time, n_products, n_freq = spectra.shape

    if product >= n_products:
        print(f"Product {product} out of range (0-{n_products-1})")
        return

    data = spectra[:, product, :]

    # Handle log scale for autocorrelations
    if product < 4:  # Autocorrelation
        # Replace zeros and NaNs with small value for log
        data_plot = np.copy(data)
        data_plot[data_plot <= 0] = np.nan
        data_plot = np.log10(data_plot + 1e-10)
        cmap = 'viridis'
        cbar_label = 'log10(Power)'
    else:  # Cross-correlation
        data_plot = data
        cmap = 'RdBu_r' if product % 2 == 0 else 'RdBu_r'  # Real and imag use same colormap
        cbar_label = 'Value'

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    # Waterfall plot
    ax1 = axes[0]
    im = ax1.imshow(data_plot, aspect='auto', cmap=cmap,
                   origin='lower', interpolation='nearest')
    ax1.set_xlabel('Frequency bin')
    ax1.set_ylabel('Time index')
    ax1.set_title(f'{PRODUCT_NAMES.get(product, f"Product {product}")} - Waterfall{title_suffix}')

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(cbar_label)

    # Mean spectrum
    ax2 = axes[1]
    mean_spectrum = np.nanmean(data, axis=0)
    std_spectrum = np.nanstd(data, axis=0)

    freq_bins = np.arange(n_freq)

    if product < 4:  # Autocorrelation - use log scale
        valid = (mean_spectrum > 0) & ~np.isnan(mean_spectrum)
        if np.any(valid):
            ax2.semilogy(freq_bins[valid], mean_spectrum[valid], 'b-', label='Mean')
            ax2.fill_between(freq_bins[valid],
                            np.maximum(mean_spectrum[valid] - std_spectrum[valid], 1e-10),
                            mean_spectrum[valid] + std_spectrum[valid],
                            alpha=0.3, color='blue')
    else:  # Cross-correlation - linear scale
        ax2.plot(freq_bins, mean_spectrum, 'b-', label='Mean')
        ax2.fill_between(freq_bins,
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.3, color='blue')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    ax2.set_xlabel('Frequency bin')
    ax2.set_ylabel('Mean value')
    ax2.set_title('Time-averaged spectrum')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, n_freq)

    plt.tight_layout()
    plt.show()

def print_statistics(spectra: np.ndarray, unique_ids: np.ndarray, metadata: dict):
    """Print statistics about the spectra."""
    n_time, n_products, n_freq = spectra.shape

    print("=" * 60)
    print("SPECTRA STATISTICS")
    print("=" * 60)
    print(f"Time samples: {n_time}")
    print(f"Products: {n_products}")
    print(f"Frequency bins: {n_freq}")
    print(f"Unique ID range: {unique_ids.min()} to {unique_ids.max()}")

    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    print("\nProduct statistics:")
    for prod in range(min(n_products, 16)):
        data = spectra[:, prod, :]
        valid_data = data[~np.isnan(data)]
        if len(valid_data) > 0:
            print(f"  {PRODUCT_NAMES.get(prod, f'Product {prod}'):15s}: "
                  f"min={valid_data.min():12.3e}, max={valid_data.max():12.3e}, "
                  f"mean={valid_data.mean():12.3e}")

    # Check for issues
    print("\nData quality checks:")
    nan_count = np.sum(np.isnan(spectra))
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaN values found ({100*nan_count/spectra.size:.2f}%)")

    for ch in range(4):
        auto_data = spectra[:, ch, :]
        neg_count = np.sum(auto_data < 0)
        if neg_count > 0:
            print(f"  WARNING: {neg_count} negative values in Ch{ch} autocorrelation")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_spectra.py <hdf5_file> [group_index]")
        print("Example: python plot_spectra.py session_000_20241105_112220.h5 0")
        sys.exit(1)

    hdf5_file = sys.argv[1]
    group_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Load data
    try:
        spectra, unique_ids, metadata = load_spectra(hdf5_file, group_idx)
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

    title_suffix = f" ({hdf5_file})"

    # Print statistics
    print_statistics(spectra, unique_ids, metadata)

    # Plot autocorrelations
    plot_autocorrelations(spectra, unique_ids, title_suffix)

    # Plot cross-correlations for channel pair 0-1
    plot_cross_correlations(spectra, unique_ids, 0, 1, title_suffix)

    # Plot waterfall for channel 0 autocorrelation
    plot_waterfall(spectra, unique_ids, product=0, title_suffix=title_suffix)

    # Interactive prompt for additional plots
    while True:
        print("\nOptions:")
        print("  1. Plot different cross-correlation (enter: cross <ch1> <ch2>)")
        print("  2. Plot different waterfall (enter: waterfall <product>)")
        print("  3. Quit (enter: q)")

        choice = input("Enter choice: ").strip().lower()

        if choice.startswith('q'):
            break
        elif choice.startswith('cross'):
            parts = choice.split()
            if len(parts) == 3:
                ch1, ch2 = int(parts[1]), int(parts[2])
                plot_cross_correlations(spectra, unique_ids, ch1, ch2, title_suffix)
        elif choice.startswith('waterfall'):
            parts = choice.split()
            if len(parts) == 2:
                product = int(parts[1])
                plot_waterfall(spectra, unique_ids, product, title_suffix)

if __name__ == "__main__":
    main()