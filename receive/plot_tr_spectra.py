#!/usr/bin/env python3
"""
Script to plot time-resolved spectra from LuSEE HDF5 files.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
from typing import Tuple, List, Optional

# Product mapping (same as regular spectra)
PRODUCT_NAMES = {
    0: 'Ch0 Auto', 1: 'Ch1 Auto', 2: 'Ch2 Auto', 3: 'Ch3 Auto',
    4: 'Ch0×Ch1 Re', 5: 'Ch0×Ch1 Im', 6: 'Ch0×Ch2 Re', 7: 'Ch0×Ch2 Im',
    8: 'Ch0×Ch3 Re', 9: 'Ch0×Ch3 Im', 10: 'Ch1×Ch2 Re', 11: 'Ch1×Ch2 Im',
    12: 'Ch1×Ch3 Re', 13: 'Ch1×Ch3 Im', 14: 'Ch2×Ch3 Re', 15: 'Ch2×Ch3 Im'
}

def load_tr_spectra(hdf5_file: str, group_idx: int = 0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load time-resolved spectra data from HDF5 file."""
    with h5py.File(hdf5_file, 'r') as f:
        groups = list(f.keys())
        if group_idx >= len(groups):
            raise ValueError(f"Group index {group_idx} out of range. File has {len(groups)} groups.")

        group = f[groups[group_idx]]

        if 'tr_spectra/data' not in group:
            raise ValueError(f"No TR spectra data in group {groups[group_idx]}")

        tr_spectra = group['tr_spectra/data'][:]
        unique_ids = group['tr_spectra/unique_ids'][:]

        # Get metadata
        metadata = {}
        metadata['Navg2'] = group.attrs.get('tr_spectra_Navg2', 1)
        metadata['tr_length'] = group.attrs.get('tr_spectra_tr_length', 0)

        # Get other relevant metadata
        if 'meta' in group:
            meta_group = group['meta']
            for key in ['tr_start', 'tr_stop', 'tr_avg_shift', 'Navg1_shift', 'Navg2_shift']:
                if key in meta_group.attrs:
                    metadata[key] = meta_group.attrs[key]

        return tr_spectra, unique_ids, metadata

def load_normal_spectra(hdf5_file: str, group_idx: int = 0) -> Optional[np.ndarray]:
    """Load normal spectra data from HDF5 file."""
    try:
        with h5py.File(hdf5_file, 'r') as f:
            groups = list(f.keys())
            if group_idx >= len(groups):
                return None

            group = f[groups[group_idx]]

            if 'spectra/data' not in group:
                return None

            spectra = group['spectra/data'][:]
            return spectra
    except:
        return None

def extract_tr_from_normal_spectra(normal_spectra: np.ndarray, metadata: dict) -> np.ndarray:
    """Extract and process the TR frequency range from normal spectra."""
    if normal_spectra is None:
        return None

    tr_start = metadata.get('tr_start', 0)
    tr_stop = metadata.get('tr_stop', 2048)
    tr_avg_shift = metadata.get('tr_avg_shift', 0)

    # Extract the TR frequency range
    n_time, n_products, n_freq = normal_spectra.shape
    tr_data = normal_spectra[:, :, tr_start:tr_stop]

    # Apply frequency averaging if needed
    if tr_avg_shift > 0:
        avg_factor = 2 ** tr_avg_shift
        n_tr_bins = (tr_stop - tr_start) // avg_factor

        # Reshape and average
        tr_data_averaged = np.zeros((n_time, n_products, n_tr_bins))
        for i in range(n_tr_bins):
            start_idx = i * avg_factor
            end_idx = start_idx + avg_factor
            tr_data_averaged[:, :, i] = np.mean(tr_data[:, :, start_idx:end_idx], axis=2)

        tr_data = tr_data_averaged

    # Average over time to get single spectrum per product
    tr_data_mean = np.nanmean(tr_data, axis=0)  # Shape: (n_products, n_tr_bins)

    return tr_data_mean

def plot_tr_autocorrelations_with_normal(tr_spectra: np.ndarray, unique_ids: np.ndarray,
                                         normal_tr_data: Optional[np.ndarray],
                                         metadata: dict, title_suffix: str = ""):
    """Plot time-resolved autocorrelation spectra with normal spectra overlay."""
    n_time, n_products, Navg2, tr_length = tr_spectra.shape
    freq_bins = np.arange(tr_length)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Select which time sample to plot (middle one)
    time_idx = n_time // 2

    for ch in range(4):
        ax = axes[ch]

        # Plot different Navg2 indices
        navg2_indices = [0, Navg2//2, Navg2-1] if Navg2 > 2 else list(range(Navg2))

        for navg2_idx in navg2_indices:
            spectrum = tr_spectra[time_idx, ch, navg2_idx, :]
            valid = (spectrum > 0) & ~np.isnan(spectrum)
            if np.any(valid):
                label = f"TR Navg2={navg2_idx}"
                ax.semilogy(freq_bins[valid], spectrum[valid],
                           alpha=0.7, label=label)

        # Overlay normal spectra data if available
        if normal_tr_data is not None and ch < normal_tr_data.shape[0]:
            normal_spectrum = normal_tr_data[ch, :]
            valid = (normal_spectrum > 0) & ~np.isnan(normal_spectrum)
            if np.any(valid):
                ax.semilogy(freq_bins[valid], normal_spectrum[valid],
                           'k--', linewidth=2, alpha=0.8,
                           label='Normal spectra (time avg)')

        ax.set_xlabel('Frequency bin (reduced)')
        ax.set_ylabel('Power')
        ax.set_title(f'Ch{ch} TR Autocorr (t={unique_ids[time_idx]})')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(0, tr_length)

    plt.suptitle(f'Time-Resolved vs Normal Spectra Comparison{title_suffix}')
    plt.tight_layout()
    plt.show()

def plot_tr_normal_comparison(tr_spectra: np.ndarray, unique_ids: np.ndarray,
                             normal_tr_data: Optional[np.ndarray],
                             metadata: dict, title_suffix: str = ""):
    """Detailed comparison between TR spectra and normal spectra."""
    if normal_tr_data is None:
        print("Normal spectra data not available for comparison")
        return

    n_time, n_products, Navg2, tr_length = tr_spectra.shape

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Compare different products
    products_to_compare = [0, 1, 2, 3, 4, 10]  # Ch0, Ch1, Ch2, Ch3 auto + 2 cross products

    for idx, prod in enumerate(products_to_compare):
        if prod >= n_products:
            continue

        ax = axes[idx // 2, idx % 2]

        # Average TR spectra over all time and Navg2
        tr_mean = np.nanmean(tr_spectra[:, prod, :, :], axis=(0, 1))
        tr_std = np.nanstd(tr_spectra[:, prod, :, :], axis=(0, 1))

        # Normal spectra for this product
        normal_spectrum = normal_tr_data[prod, :]

        freq_bins = np.arange(tr_length)

        if prod < 4:  # Autocorrelation
            # Plot TR mean with error band
            valid_tr = (tr_mean > 0) & ~np.isnan(tr_mean)
            if np.any(valid_tr):
                ax.semilogy(freq_bins[valid_tr], tr_mean[valid_tr],
                           'b-', label='TR mean', linewidth=2)
                ax.fill_between(freq_bins[valid_tr],
                               np.maximum(tr_mean[valid_tr] - tr_std[valid_tr], 1e-10),
                               tr_mean[valid_tr] + tr_std[valid_tr],
                               alpha=0.3, color='blue')

            # Plot normal spectra
            valid_normal = (normal_spectrum > 0) & ~np.isnan(normal_spectrum)
            if np.any(valid_normal):
                ax.semilogy(freq_bins[valid_normal], normal_spectrum[valid_normal],
                           'r--', label='Normal spectra', linewidth=2)

            # Calculate and show relative difference
            if np.any(valid_tr & valid_normal):
                overlap = valid_tr & valid_normal
                rel_diff = np.abs(tr_mean[overlap] - normal_spectrum[overlap]) / normal_spectrum[overlap]
                mean_rel_diff = np.mean(rel_diff) * 100
                ax.text(0.02, 0.02, f'Mean rel. diff: {mean_rel_diff:.1f}%',
                       transform=ax.transAxes, fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:  # Cross-correlation
            ax.plot(freq_bins, tr_mean, 'b-', label='TR mean', linewidth=2)
            ax.fill_between(freq_bins, tr_mean - tr_std, tr_mean + tr_std,
                           alpha=0.3, color='blue')
            ax.plot(freq_bins, normal_spectrum, 'r--', label='Normal spectra', linewidth=2)
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)

            # Calculate RMS difference
            rms_diff = np.sqrt(np.nanmean((tr_mean - normal_spectrum)**2))
            ax.text(0.02, 0.02, f'RMS diff: {rms_diff:.2e}',
                   transform=ax.transAxes, fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Frequency bin')
        ax.set_ylabel('Value')
        ax.set_title(PRODUCT_NAMES.get(prod, f'Product {prod}'))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.suptitle(f'TR Spectra vs Normal Spectra Detailed Comparison{title_suffix}')
    plt.tight_layout()
    plt.show()

    # Print comparison statistics
    print("\n" + "=" * 60)
    print("TR SPECTRA vs NORMAL SPECTRA COMPARISON")
    print("=" * 60)
    print(f"TR start: {metadata.get('tr_start', 'N/A')}")
    print(f"TR stop: {metadata.get('tr_stop', 'N/A')}")
    print(f"TR avg shift: {metadata.get('tr_avg_shift', 'N/A')}")
    print(f"TR length: {tr_length}")
    print(f"Navg2: {Navg2}")

    print("\nProduct-wise comparison:")
    for prod in range(min(n_products, 16)):
        tr_mean = np.nanmean(tr_spectra[:, prod, :, :], axis=(0, 1))
        normal_spectrum = normal_tr_data[prod, :] if prod < normal_tr_data.shape[0] else None

        if normal_spectrum is not None:
            valid = ~np.isnan(tr_mean) & ~np.isnan(normal_spectrum)
            if np.any(valid):
                if prod < 4:  # Autocorrelation
                    valid = valid & (tr_mean > 0) & (normal_spectrum > 0)
                    if np.any(valid):
                        rel_diff = np.abs(tr_mean[valid] - normal_spectrum[valid]) / normal_spectrum[valid]
                        print(f"  {PRODUCT_NAMES.get(prod, f'Product {prod}'):15s}: "
                              f"mean rel. diff = {np.mean(rel_diff)*100:.2f}%")
                else:  # Cross-correlation
                    rms_diff = np.sqrt(np.nanmean((tr_mean[valid] - normal_spectrum[valid])**2))
                    print(f"  {PRODUCT_NAMES.get(prod, f'Product {prod}'):15s}: "
                          f"RMS diff = {rms_diff:.3e}")

def plot_tr_evolution_with_normal(tr_spectra: np.ndarray, unique_ids: np.ndarray,
                                  normal_tr_data: Optional[np.ndarray],
                                  product: int = 0, navg2_idx: int = 0,
                                  metadata: dict = {}, title_suffix: str = ""):
    """Plot evolution of TR spectra with normal spectra reference."""
    n_time, n_products, Navg2, tr_length = tr_spectra.shape

    if product >= n_products:
        print(f"Product {product} out of range (0-{n_products-1})")
        return

    if navg2_idx >= Navg2:
        print(f"Navg2 index {navg2_idx} out of range (0-{Navg2-1})")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 2, 1], hspace=0.3)

    # Extract data for this product and Navg2 index
    data = tr_spectra[:, product, navg2_idx, :]

    # Waterfall plot
    ax1 = fig.add_subplot(gs[0])
    if product < 4:  # Autocorrelation
        data_plot = np.log10(data + 1e-10)
        cmap = 'viridis'
        cbar_label = 'log10(Power)'
    else:  # Cross-correlation
        data_plot = data
        cmap = 'RdBu_r'
        cbar_label = 'Value'

    im = ax1.imshow(data_plot, aspect='auto', cmap=cmap,
                   origin='lower', interpolation='nearest')
    ax1.set_xlabel('Frequency bin')
    ax1.set_ylabel('Time index')
    ax1.set_title(f'{PRODUCT_NAMES.get(product, f"Product {product}")} - '
                 f'Navg2 idx={navg2_idx} - Waterfall')

    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(cbar_label)

    # Plot specific time slices with normal spectra
    ax2 = fig.add_subplot(gs[1])
    time_indices = [0, n_time//2, n_time-1] if n_time > 2 else list(range(n_time))

    for t_idx in time_indices:
        spectrum = data[t_idx, :]
        if product < 4:  # Autocorrelation
            valid = (spectrum > 0) & ~np.isnan(spectrum)
            if np.any(valid):
                ax2.semilogy(spectrum[valid], alpha=0.7,
                           label=f't_idx={t_idx} (uid={unique_ids[t_idx]})')
        else:  # Cross-correlation
            ax2.plot(spectrum, alpha=0.7,
                    label=f't_idx={t_idx} (uid={unique_ids[t_idx]})')

    # Add normal spectra reference
    if normal_tr_data is not None and product < normal_tr_data.shape[0]:
        normal_spectrum = normal_tr_data[product, :]
        if product < 4:
            valid = (normal_spectrum > 0) & ~np.isnan(normal_spectrum)
            if np.any(valid):
                ax2.semilogy(normal_spectrum[valid], 'k--', linewidth=2,
                           alpha=0.8, label='Normal spectra (ref)')
        else:
            ax2.plot(normal_spectrum, 'k--', linewidth=2,
                    alpha=0.8, label='Normal spectra (ref)')
            ax2.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    ax2.set_xlabel('Frequency bin')
    ax2.set_ylabel('Value')
    ax2.set_title('Selected time slices vs Normal spectra')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Mean spectrum comparison
    ax3 = fig.add_subplot(gs[2])
    mean_spectrum = np.nanmean(data, axis=0)
    std_spectrum = np.nanstd(data, axis=0)
    freq_bins = np.arange(tr_length)

    if product < 4:  # Autocorrelation
        valid = (mean_spectrum > 0) & ~np.isnan(mean_spectrum)
        if np.any(valid):
            ax3.semilogy(freq_bins[valid], mean_spectrum[valid], 'b-', label='TR Mean')
            ax3.fill_between(freq_bins[valid],
                            np.maximum(mean_spectrum[valid] - std_spectrum[valid], 1e-10),
                            mean_spectrum[valid] + std_spectrum[valid],
                            alpha=0.3, color='blue')

        if normal_tr_data is not None and product < normal_tr_data.shape[0]:
            normal_spectrum = normal_tr_data[product, :]
            valid_normal = (normal_spectrum > 0) & ~np.isnan(normal_spectrum)
            if np.any(valid_normal):
                ax3.semilogy(freq_bins[valid_normal], normal_spectrum[valid_normal],
                           'r--', linewidth=2, label='Normal spectra')
    else:  # Cross-correlation
        ax3.plot(freq_bins, mean_spectrum, 'b-', label='TR Mean')
        ax3.fill_between(freq_bins,
                        mean_spectrum - std_spectrum,
                        mean_spectrum + std_spectrum,
                        alpha=0.3, color='blue')

        if normal_tr_data is not None and product < normal_tr_data.shape[0]:
            normal_spectrum = normal_tr_data[product, :]
            ax3.plot(freq_bins, normal_spectrum, 'r--', linewidth=2, label='Normal spectra')

        ax3.axhline(y=0, color='k', linestyle=':', alpha=0.3)

    ax3.set_xlabel('Frequency bin')
    ax3.set_ylabel('Mean value')
    ax3.set_title('Time-averaged spectrum comparison')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.suptitle(f'TR Spectra Evolution with Normal Spectra Reference{title_suffix}')
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_tr_spectra.py <hdf5_file> [group_index]")
        print("Example: python plot_tr_spectra.py session_000_20241105_112220.h5 0")
        sys.exit(1)

    hdf5_file = sys.argv[1]
    group_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Load TR spectra data
    try:
        tr_spectra, unique_ids, metadata = load_tr_spectra(hdf5_file, group_idx)
    except Exception as e:
        print(f"Error loading TR spectra: {e}")
        sys.exit(1)

    # Load normal spectra for comparison
    normal_spectra = load_normal_spectra(hdf5_file, group_idx)
    normal_tr_data = extract_tr_from_normal_spectra(normal_spectra, metadata)

    if normal_tr_data is not None:
        print("Successfully loaded normal spectra for comparison")
    else:
        print("Normal spectra not available for comparison")

    title_suffix = f" ({hdf5_file})"

    # Print statistics
    print("\n" + "=" * 60)
    print("TIME-RESOLVED SPECTRA STATISTICS")
    print("=" * 60)
    n_time, n_products, Navg2, tr_length = tr_spectra.shape
    print(f"Time samples: {n_time}")
    print(f"Products: {n_products}")
    print(f"Navg2: {Navg2}")
    print(f"TR length (frequency bins): {tr_length}")
    print(f"Unique ID range: {unique_ids.min()} to {unique_ids.max()}")

    print("\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Plot autocorrelations with normal spectra overlay
    plot_tr_autocorrelations_with_normal(tr_spectra, unique_ids, normal_tr_data,
                                         metadata, title_suffix)

    # Plot detailed comparison
    plot_tr_normal_comparison(tr_spectra, unique_ids, normal_tr_data,
                             metadata, title_suffix)

    # Plot evolution with normal spectra reference
    plot_tr_evolution_with_normal(tr_spectra, unique_ids, normal_tr_data,
                                  product=0, navg2_idx=0,
                                  metadata=metadata, title_suffix=title_suffix)

    # Interactive prompt for additional plots
    while True:
        print("\nOptions:")
        print("  1. Plot evolution with normal ref (enter: evolution <product> <navg2_idx>)")
        print("  2. Re-plot comparison (enter: compare)")
        print("  3. Quit (enter: q)")

        choice = input("Enter choice: ").strip().lower()

        if choice.startswith('q'):
            break
        elif choice.startswith('evolution'):
            parts = choice.split()
            if len(parts) >= 3:
                product = int(parts[1])
                navg2_idx = int(parts[2])
                plot_tr_evolution_with_normal(tr_spectra, unique_ids, normal_tr_data,
                                             product, navg2_idx,
                                             metadata, title_suffix)
        elif choice.startswith('compare'):
            plot_tr_normal_comparison(tr_spectra, unique_ids, normal_tr_data,
                                    metadata, title_suffix)

if __name__ == "__main__":
    main()
