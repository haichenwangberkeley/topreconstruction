#!/usr/bin/env python3
"""
Optimized script to plot truth_triplet_0 data from ttbar.root using uproot
Based on analysis: each event has exactly 3 values, ranging from -1 to 6
"""

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def plot_truth_triplet_0_optimized(filename="ttbar.root", max_events=50000):
    """
    Plot truth_triplet_0 data efficiently
    
    Args:
        filename: ROOT file to read
        max_events: Maximum number of events to process
    """
    try:
        print(f"Opening {filename}...")
        with uproot.open(filename) as file:
            tree = file['output']
            total_entries = len(tree)
            entries_to_read = min(max_events, total_entries)
            
            print(f"Total entries: {total_entries:,}")
            print(f"Analyzing first {entries_to_read:,} events...")
            
            # Read data efficiently
            truth_triplet_0 = tree['truth_triplet_0'].array(entry_stop=entries_to_read)
            
            print("Processing data...")
            
            # Convert to numpy arrays for efficient processing
            # Each event has exactly 3 values, so reshape accordingly
            all_data = np.array([list(vec) for vec in truth_triplet_0])  # Shape: (n_events, 3)
            
            print(f"Data shape: {all_data.shape}")
            
            # Create comprehensive plots
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Truth Triplet 0 Analysis ({entries_to_read:,} events)', fontsize=16)
            
            # Plot 1: Histogram of all values combined
            all_values = all_data.flatten()
            unique_vals, counts = np.unique(all_values, return_counts=True)
            
            axes[0, 0].bar(unique_vals, counts, alpha=0.8, edgecolor='black')
            axes[0, 0].set_xlabel('Truth triplet values')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of All Values')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for val, count in zip(unique_vals, counts):
                axes[0, 0].text(val, count + max(counts)*0.01, str(count), 
                               ha='center', va='bottom', fontsize=9)
            
            # Plot 2: Distribution by position in triplet
            colors = ['red', 'green', 'blue']
            positions = ['Position 0', 'Position 1', 'Position 2']
            
            for pos in range(3):
                pos_values, pos_counts = np.unique(all_data[:, pos], return_counts=True)
                axes[0, 1].bar(pos_values + pos*0.25 - 0.25, pos_counts, 
                              width=0.25, alpha=0.7, label=positions[pos], color=colors[pos])
            
            axes[0, 1].set_xlabel('Truth triplet values')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution by Position in Triplet')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Percentage of -1 values (missing/invalid)
            missing_counts = np.sum(all_data == -1, axis=1)  # Count -1s per event
            missing_unique, missing_freq = np.unique(missing_counts, return_counts=True)
            
            axes[0, 2].bar(missing_unique, missing_freq, alpha=0.8, color='orange', edgecolor='black')
            axes[0, 2].set_xlabel('Number of -1 values per event')
            axes[0, 2].set_ylabel('Number of events')
            axes[0, 2].set_title('Missing Values (-1) per Event')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Add percentages
            total_events = len(all_data)
            for count, freq in zip(missing_unique, missing_freq):
                pct = (freq / total_events) * 100
                axes[0, 2].text(count, freq + max(missing_freq)*0.01, f'{pct:.1f}%', 
                               ha='center', va='bottom', fontsize=9)
            
            # Plot 4: Valid triplets (excluding -1 values)
            valid_mask = all_data != -1
            valid_events = np.all(valid_mask, axis=1)  # Events with no -1 values
            valid_data = all_data[valid_events]
            
            if len(valid_data) > 0:
                valid_all = valid_data.flatten()
                valid_unique, valid_counts = np.unique(valid_all, return_counts=True)
                
                axes[1, 0].bar(valid_unique, valid_counts, alpha=0.8, color='green', edgecolor='black')
                axes[1, 0].set_xlabel('Truth triplet values')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title(f'Valid Values Only ({len(valid_data)} events)')
                axes[1, 0].grid(True, alpha=0.3)
                
                for val, count in zip(valid_unique, valid_counts):
                    axes[1, 0].text(val, count + max(valid_counts)*0.01, str(count), 
                                   ha='center', va='bottom', fontsize=9)
            else:
                axes[1, 0].text(0.5, 0.5, 'No valid triplets\n(all contain -1)', 
                               transform=axes[1, 0].transAxes, ha='center', va='center')
                axes[1, 0].set_title('Valid Values Only (No Data)')
            
            # Plot 5: Correlation heatmap between positions
            if len(valid_data) > 10:  # Need some data for correlation
                corr_matrix = np.corrcoef(valid_data.T)
                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_title('Correlation Between Positions')
                axes[1, 1].set_xticks([0, 1, 2])
                axes[1, 1].set_yticks([0, 1, 2])
                axes[1, 1].set_xticklabels(['Pos 0', 'Pos 1', 'Pos 2'])
                axes[1, 1].set_yticklabels(['Pos 0', 'Pos 1', 'Pos 2'])
                
                # Add correlation values to heatmap
                for i in range(3):
                    for j in range(3):
                        text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                             ha="center", va="center", color="black", fontweight='bold')
                
                plt.colorbar(im, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, 'Insufficient valid data\nfor correlation', 
                               transform=axes[1, 1].transAxes, ha='center', va='center')
                axes[1, 1].set_title('Correlation (Insufficient Data)')
            
            # Plot 6: Summary statistics
            axes[1, 2].axis('off')
            
            # Calculate statistics
            total_values = len(all_values)
            missing_values = np.sum(all_values == -1)
            valid_values = total_values - missing_values
            
            stats_text = f"""Summary Statistics:
            
Total Events: {entries_to_read:,}
Total Values: {total_values:,}
Missing (-1): {missing_values:,} ({missing_values/total_values*100:.1f}%)
Valid Values: {valid_values:,} ({valid_values/total_values*100:.1f}%)

Valid Value Range: {np.min(all_values[all_values != -1]):.0f} to {np.max(all_values[all_values != -1]):.0f}

Events with:
• 0 missing: {np.sum(missing_counts == 0):,} ({np.sum(missing_counts == 0)/total_events*100:.1f}%)
• 1 missing: {np.sum(missing_counts == 1):,} ({np.sum(missing_counts == 1)/total_events*100:.1f}%)
• 2 missing: {np.sum(missing_counts == 2):,} ({np.sum(missing_counts == 2)/total_events*100:.1f}%)
• 3 missing: {np.sum(missing_counts == 3):,} ({np.sum(missing_counts == 3)/total_events*100:.1f}%)

Most Common Values:
"""
            
            # Add most common values
            sorted_indices = np.argsort(counts)[::-1]
            for i, idx in enumerate(sorted_indices[:5]):
                val = unique_vals[idx]
                count = counts[idx]
                pct = count / len(all_values) * 100
                if val == -1:
                    stats_text += f"• {val} (missing): {count:,} ({pct:.1f}%)\n"
                else:
                    stats_text += f"• {val}: {count:,} ({pct:.1f}%)\n"
            
            axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                           fontsize=10, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            output_name = f'truth_triplet_0_analysis_{entries_to_read}.png'
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved as {output_name}")
            
            # Print summary to console
            print(f"\n=== Analysis Summary ===")
            print(f"Processed {entries_to_read:,} events")
            print(f"Each event has exactly 3 values")
            print(f"Value range: -1 (missing) and 0-6 (valid indices)")
            print(f"Missing values: {missing_values:,}/{total_values:,} ({missing_values/total_values*100:.1f}%)")
            print(f"Completely valid events: {np.sum(missing_counts == 0):,} ({np.sum(missing_counts == 0)/total_events*100:.1f}%)")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    filename = "ttbar.root"
    max_events = 50000  # Default to 50k events for good statistics but reasonable speed
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        max_events = int(sys.argv[2])
    
    plot_truth_triplet_0_optimized(filename, max_events)