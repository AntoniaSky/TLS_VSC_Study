""" 
Script to calculate various vegetation structural metrics based on entropy from the processed TLS point cloud data.
The script calculates the Vertical Complexity Index (VCI), Foliage Height Diversity (FHD), and Shannon Entropy at different height bins (1 m, 2 m, 5 m, 10 m, 15 m, 20 m) for the vegetation points in the point cloud data. 
The results are saved to a general results CSV file for further analysis and comparison across different public squares.

VCI and FHD are ways to normalize the entropy values to make them comparable across different sites and conditions but still formulas use different logarithm bases as Shannon Entropy.

The formulas are found in Jia et al. (2025) and height bins are suggested in Shokirov et al. (2023)

Sources:
- Jia, J., et al. (2025). “Vertical regulation of thermal stress by canopy structure in urban forests: The role of species composition”. In: Landscape and Urban Planning 264, p. 105495. DOI: 10.1016/j.landurbplan.2025.105495.
- Shokirov, S., et al. (2023). “Habitat highs and lows: Using terrestrial and UAV LiDAR for modelling avian species richness and abundance in a restored woodland”. In: Remote Sensing of Environment 285, p. 113326. DOI: 10.1016/j.rse.2022.113326.


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
Originally created Spring 2026, last updated June 2026
"""

import os
import laspy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.optimize import leastsq
from definitions_a import *
import math
import csv
from typing import Tuple, Dict




from definitions_a import assign_pointdata_with_cellkey,key_to_cells



    
    
    
    
def vci_calc(main_path: str, square: str,):
    """ Calculate VCI, FHD, and Entropy for the vegetation points in the point cloud data for the given square and save results to CSV
    Args:
        main_path: str, path to the main directory containing the square folders
        square: str, name of the square to process
        
    Returns:
        None: The function saves the calculated Skewness and Kurtosis in the general results CSV file (All_Results.csv) in the main path of the square. It also prints the results to the console.
    """
    print(f"Calculating VCI, FHD, and Entropy for square: {square} at main path: {main_path}")
    
    ######## Define paths for the current square
    laz_path = f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'
    memmap_path = f'{main_path}/{square}_FINAL/{square}_pointcelldata.memmap'
    # Load the point cloud data after preprocessing via definitions_a assign_pointdata_with_cellkey
    

    result = assign_pointdata_with_cellkey(
        laz_path=laz_path,
        memmap_path=memmap_path,
        cell_size=5,
        chunk_size=5_000_000,
        coord_dtype=np.float32,
    )
    
    
    arr = result["per_point"] 
    df = pd.DataFrame({
        "cell_key": arr["cell_key"],
        "X": arr["x"].astype(np.float64),
        "Y": arr["y"].astype(np.float64),
        "Z": arr["z"].astype(np.float64),
        "Intensity": arr["intensity"].astype(np.float32),
        "Number_of_Returns": arr["num_of_returns"].astype(np.uint8),
        "Return_Number": arr["return_number"].astype(np.uint8),
        "Classification": arr["classification"].astype(np.int32)
        })
    print(df.head())
    print(f"Total points loaded: {len(df)}")

    # Filter out ground points (Classification 2 and 3 are vegetation)
    veg_points = df[(df['Classification'] == 2) | (df['Classification'] == 3)] # | means OR
    print(f"Total vegetation points: {len(veg_points)}")

    
    ######## Calculate VCI FHD and Shannon Entropy for whole cloud
    print("Calculating VCI, FHD, and Entropy for the entire vegetation point cloud...")
    height_bins = [1, 2, 5, 10, 15, 20]
    vci_results = {}  
    #total vegetation points  
    total_veg_points = len(veg_points)
    
    for bin_height in height_bins:
        # Create height bins
        max_height = veg_points['Z'].max()
        bins = np.arange(1.3, max_height + bin_height, bin_height)
        
        # Calculate amount of points in each bin via histogram
        hist, _ = np.histogram(veg_points['Z'], bins=bins)
        
        # Calculate pi (Probability of points in each bin) for the formulas
        pi = hist / total_veg_points
        pi = pi[pi > 0]  # Remove zero entries to avoid log(0)
        
        # Calculate Shannon Entropy (caluclated with log with base 2) as in Jia et al. (2025)
        # Entropy = -∑(pi * log2(pi)) (Jia et al., 2025)
        entropy = -np.sum(pi * np.log2(pi))
        vci_results[f'entropy_{bin_height}m'] = entropy
        print(f"Entropy for {bin_height}m bins: {entropy:.4f}")
        
        # Calculate Foliage Height Diversity (FHD)
        # FHD = -∑(pi * log(pi)) * log(Hmax) (Jia et al., 2025) 
        fhd = -np.sum(pi * np.log(pi)) * np.log(max_height)
        vci_results[f'fhd_{bin_height}m'] = fhd
        print(f"FHD for {bin_height}m bins: {fhd:.4f}")
        
        # Calculate VCI
        # VCI = -∑(pi * log(pi)) / log(n-bins) (Jia et al., 2025)
        vci = -np.sum(pi * np.log(pi)) / np.log(len(bins) - 1) 
        vci_results[f'vci_{bin_height}m'] = vci
        print(f"VCI for {bin_height}m bins: {vci:.4f}")

    
    
    ########################### Combine metrics and add to General Results CSV
    general_results_path = f'{main_path}/All_Results.csv'
    
    new_row = {'Square': square,
                f'entropy_1m': vci_results['entropy_1m'],
                f'entropy_2m': vci_results['entropy_2m'],
                f'entropy_5m': vci_results['entropy_5m'],
                f'entropy_10m': vci_results['entropy_10m'],
                f'entropy_15m': vci_results['entropy_15m'],
                f'entropy_20m': vci_results['entropy_20m'],
                f'fhd_1m': vci_results['fhd_1m'],
                f'fhd_2m': vci_results['fhd_2m'],
                f'fhd_5m': vci_results['fhd_5m'],
                f'fhd_10m': vci_results['fhd_10m'],
                f'fhd_15m': vci_results['fhd_15m'],
                f'fhd_20m': vci_results['fhd_20m'],
                f'vci_1m': vci_results['vci_1m'],
                f'vci_2m': vci_results['vci_2m'],
                f'vci_5m': vci_results['vci_5m'],
                f'vci_10m': vci_results['vci_10m'],
                f'vci_15m': vci_results['vci_15m'],
                f'vci_20m': vci_results['vci_20m']
                }
    
    if os.path.exists(general_results_path):
        general_results_df = pd.read_csv(general_results_path)
        # check if the columns exist, if not create them
        for col in new_row.keys():
            if col not in general_results_df.columns:
                general_results_df[col] = np.nan
    else:
        general_results_df = pd.DataFrame(columns=new_row.keys())
    
    general_results_df.loc[len(general_results_df)] = new_row
    general_results_df.to_csv(general_results_path, index=False)
    print(f"General results updated at {general_results_path}")

if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square_list = ['rundfunk']
    square_list = ['konigsplatz','alpenplatz','rundfunk','nikolai', 'miesbacher', "jakobgelb"]
    square_list = ["elisabeth"]
    for square in square_list:  
        vci_calc(main_path, square)
    
