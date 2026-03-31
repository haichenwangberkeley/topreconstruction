#!/usr/bin/env python3
"""
Quick script to examine truth_triplet_0 data structure
"""

import uproot
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def quick_analysis():
    """
    Quick analysis of truth_triplet_0 with minimal data loading
    """
    try:
        print("Opening ttbar.root...")
        with uproot.open("ttbar.root") as file:
            tree = file['output']
            
            # Read just the first 100 events
            print("Reading first 100 events...")
            truth_triplet_0 = tree['truth_triplet_0'].array(entry_stop=100)
            
            print(f"Successfully loaded {len(truth_triplet_0)} events")
            
            # Quick analysis
            lengths = [len(vec) for vec in truth_triplet_0]
            all_values = []
            for vec in truth_triplet_0:
                all_values.extend(vec)
            
            print(f"\nQuick Stats:")
            print(f"Vector lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.2f}")
            if all_values:
                print(f"Values: min={min(all_values)}, max={max(all_values)}, total={len(all_values)}")
                print(f"Unique values: {sorted(set(all_values))}")
            
            # Simple plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot vector lengths
            ax1.hist(lengths, bins=max(1, max(lengths) - min(lengths) + 1), alpha=0.7)
            ax1.set_xlabel('Vector length')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Vector Lengths (100 events)')
            ax1.grid(True, alpha=0.3)
            
            # Plot values
            if all_values:
                unique_vals, counts = np.unique(all_values, return_counts=True)
                ax2.bar(unique_vals, counts, alpha=0.7)
                ax2.set_xlabel('Truth triplet values')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Value Distribution (100 events)')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('truth_triplet_0_quick.png', dpi=150, bbox_inches='tight')
            print(f"Quick plot saved as truth_triplet_0_quick.png")
            
            # Show some actual data
            print(f"\nFirst 5 events data:")
            for i in range(min(5, len(truth_triplet_0))):
                print(f"Event {i}: {list(truth_triplet_0[i])}")
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_analysis()