#!/usr/bin/env python3
"""
Script to plot truth_triplet_0 data from ttbar.root using uproot
"""

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import sys

def plot_truth_triplet_0(filename="ttbar.root", max_events=10000):
    """
    Plot truth_triplet_0 data from ROOT file
    Args:
        filename: ROOT file to read
        max_events: Maximum number of events to process for efficiency
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return
    
    try:
        # Open the ROOT file and read the data
        print(f"Opening {filename}...")
        with uproot.open(filename) as file:
            tree = file['output']
            
            # Read the truth_triplet_0 branch (limit entries for efficiency)
            total_entries = len(tree)
            entries_to_read = min(max_events, total_entries)
            
            print(f"Total entries in file: {total_entries}")
            print(f"Reading first {entries_to_read} entries for analysis...")
            
            truth_triplet_0 = tree['truth_triplet_0'].array(entry_stop=entries_to_read)
            
            print(f"Loaded {len(truth_triplet_0)} events")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Truth Triplet 0 Analysis', fontsize=16)
            
            # Plot 1: Distribution of vector lengths
            lengths = [len(vec) for vec in truth_triplet_0]
            axes[0, 0].hist(lengths, bins=max(1, max(lengths) - min(lengths) + 1), 
                           alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Number of elements per event')
            axes[0, 0].set_ylabel('Number of events')
            axes[0, 0].set_title('Distribution of Vector Lengths')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {np.mean(lengths):.2f}\nStd: {np.std(lengths):.2f}\nMin: {min(lengths)}\nMax: {max(lengths)}'
            axes[0, 0].text(0.7, 0.7, stats_text, transform=axes[0, 0].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Plot 2: Distribution of actual values (flattened)
            all_values = []
            for vec in truth_triplet_0:
                all_values.extend(vec)
            
            if all_values:
                axes[0, 1].hist(all_values, bins=50, alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('Truth triplet values')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Distribution of All Values')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add statistics
                stats_text2 = f'Mean: {np.mean(all_values):.2f}\nStd: {np.std(all_values):.2f}\nMin: {min(all_values)}\nMax: {max(all_values)}\nTotal: {len(all_values)}'
                axes[0, 1].text(0.7, 0.7, stats_text2, transform=axes[0, 1].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            else:
                axes[0, 1].text(0.5, 0.5, 'No data to plot', transform=axes[0, 1].transAxes,
                               ha='center', va='center')
                axes[0, 1].set_title('Distribution of All Values (No Data)')
            
            # Plot 3: Event-by-event view (first 20 events)
            max_events_to_show = min(20, len(truth_triplet_0))
            event_numbers = []
            values_for_scatter = []
            
            for i in range(max_events_to_show):
                for val in truth_triplet_0[i]:
                    event_numbers.append(i)
                    values_for_scatter.append(val)
            
            if values_for_scatter:
                axes[1, 0].scatter(event_numbers, values_for_scatter, alpha=0.6)
                axes[1, 0].set_xlabel('Event number')
                axes[1, 0].set_ylabel('Truth triplet values')
                axes[1, 0].set_title(f'Values per Event (First {max_events_to_show} events)')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'No data to plot', transform=axes[1, 0].transAxes,
                               ha='center', va='center')
                axes[1, 0].set_title('Values per Event (No Data)')
            
            # Plot 4: Unique values histogram
            if all_values:
                unique_values, counts = np.unique(all_values, return_counts=True)
                axes[1, 1].bar(unique_values, counts, alpha=0.7, edgecolor='black')
                axes[1, 1].set_xlabel('Unique truth triplet values')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Frequency of Unique Values')
                axes[1, 1].grid(True, alpha=0.3)
                
                # Show top 5 most common values
                top_indices = np.argsort(counts)[-5:][::-1]
                top_values = unique_values[top_indices]
                top_counts = counts[top_indices]
                
                info_text = 'Top 5 values:\n' + '\n'.join([f'{val}: {count}' for val, count in zip(top_values, top_counts)])
                axes[1, 1].text(0.7, 0.7, info_text, transform=axes[1, 1].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
            else:
                axes[1, 1].text(0.5, 0.5, 'No data to plot', transform=axes[1, 1].transAxes,
                               ha='center', va='center')
                axes[1, 1].set_title('Frequency of Unique Values (No Data)')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save the plot
            output_name = 'truth_triplet_0_analysis.png'
            plt.savefig(output_name, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {output_name}")
            
            # Don't show plot in headless environment
            # plt.show()
            
            # Print some summary information
            print(f"\n=== Summary Statistics ===")
            print(f"Total events: {len(truth_triplet_0)}")
            print(f"Vector length statistics:")
            print(f"  Mean length: {np.mean(lengths):.2f}")
            print(f"  Min length: {min(lengths)}")
            print(f"  Max length: {max(lengths)}")
            
            if all_values:
                print(f"Value statistics:")
                print(f"  Total values: {len(all_values)}")
                print(f"  Unique values: {len(unique_values)}")
                print(f"  Value range: {min(all_values)} to {max(all_values)}")
                print(f"  Mean value: {np.mean(all_values):.2f}")
            
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Allow specifying filename as command line argument
    filename = "ttbar.root"
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    plot_truth_triplet_0(filename)