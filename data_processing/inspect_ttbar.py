#!/usr/bin/env python3
"""
Script to inspect branches in ttbar.root file using uproot
"""

import uproot
import os
import sys

def inspect_root_file(filename):
    """
    Inspect a ROOT file and print information about its contents
    """
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found!")
        return
    
    try:
        # Open the ROOT file
        print(f"Opening {filename}...")
        with uproot.open(filename) as file:
            print(f"\n=== ROOT File Contents ===")
            print(f"File: {filename}")
            print(f"Keys in file: {list(file.keys())}")
            
            # Iterate through all objects in the file
            for key in file.keys():
                print(f"\n--- Object: {key} ---")
                obj = file[key]
                
                # Check if it's a TTree
                if hasattr(obj, 'keys') and callable(obj.keys):
                    print(f"Type: TTree (or similar)")
                    print(f"Number of entries: {len(obj)}")
                    
                    # Get branches
                    branches = obj.keys()
                    print(f"Number of branches: {len(branches)}")
                    print("Branches:")
                    
                    for i, branch in enumerate(branches, 1):
                        try:
                            # Get branch info
                            branch_obj = obj[branch]
                            dtype = branch_obj.typename if hasattr(branch_obj, 'typename') else 'unknown'
                            print(f"  {i:2d}. {branch:<30} ({dtype})")
                        except Exception as e:
                            print(f"  {i:2d}. {branch:<30} (error reading: {e})")
                    
                    # Show some statistics for numeric branches
                    print(f"\n--- Sample data (first 5 entries) ---")
                    try:
                        # Get a few branches to show sample data
                        sample_branches = branches[:min(5, len(branches))]
                        data = obj.arrays(sample_branches, entry_stop=5)
                        
                        for branch in sample_branches:
                            values = data[branch]
                            print(f"{branch}: {list(values)}")
                            
                    except Exception as e:
                        print(f"Could not read sample data: {e}")
                        
                else:
                    print(f"Type: {type(obj)}")
                    try:
                        print(f"Content: {obj}")
                    except:
                        print("(Unable to display content)")
    
    except Exception as e:
        print(f"Error opening file: {e}")
        print("Make sure the file is a valid ROOT file and not corrupted.")

if __name__ == "__main__":
    # Check if ttbar.root exists in current directory
    filename = "ttbar.root"
    
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    inspect_root_file(filename)