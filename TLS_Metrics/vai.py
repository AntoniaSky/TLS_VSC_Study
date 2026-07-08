"""Vegetation Area Index (VAI) and Vegetation Area Density (VAD) (based on formula from Jia et al., 2025)
Estimated by summing the Vegetation Area Density (VAD) across all vertical height layers and using the Beer-Lambert law to account for the attenuation of LiDAR pulses as they penetrate the canopy. 

In Jia et al. (2025): VAI is calculated using the formula:
VAI= Summe_z-start_to_z-max(-1/k *ln((n_belowZ+∆z/N_belowZ))
"Where z0 is the starting height for VAI calculation, usually set to 3 m to exclude ground vegetation, Δz is vertical bin size (layer thickness),
typically set to 1 m, zmax is maximum canopy height in the point cloud, k is extinction coefficient, a parameter from the Beer-Lambert law; commonly
set to 0.5 based on empirical studies, Nbelow(z) is the number of LiDAR points above the height z; used to estimate vertical penetration and leaf density."

We adapted the formula to TLS data (instead of ALS) and changed n_belowZ to n_aboveZ as we are interested in the number of points above the height layer (instead of below) to estimate the vertical penetration of liDAR pulses and leaf density. 
We also start from 0.1m to include understorey vegetation, but exclude grass only vegetation. This however can confuse the alogrithm as no light no light penetration is expected with understorey but with mutliple scanning positions we can have light penetration in the understorey layer.

Source:
- Jia, J., et al. (2025). “Vertical regulation of thermal stress by canopy structure in urban forests: The role of species composition”. In: Landscape and Urban Planning 264, p. 105495. DOI: 10.1016/j.landurbplan.2025.105495.


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT) adapted from equations from Jia et al. (2025) 
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
from TLS_Metrics.VCI import vci_calc
from definitions_a import *
import math
import csv
from typing import Tuple, Dict




from definitions_a import assign_pointdata_with_cellkey,key_to_cells



    
    
    
def vai_calc(main_path: str, square: str):
    """Calculate the Vegetation Area Index (VAI) for a given square using the point cloud data and the method described in Jia et al. (2025).
    Saves the calculated VAI values to the general results CSV file.

    Args:
        main_path (str): The path to the main directory containing the data and the folders to the final processed point cloud data.
        square (str): The identifier for the square area.
    Returns:
        None: The function saves the calculated VAI values to the general results CSV file and does not return any value.   
    
    """
    
    print(f"Calculating VAI for square: {square}")
    
    ############### Define paths and parameters    
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

    
    ############### Calculate VAI for whole cloud
    print("Calculating VAI for the entire vegetation point cloud...")
    height_bins = [1, 2, 5, 10, 15, 20] # different bin heights to test the effect of layer thickness on VAI calculation as in Shokirov et al. (2023)
    vai_results = {}  
    
    #total vegetation points  
    total_veg_points = len(veg_points)
    
    # Define k
    k = 0.5  # extinction coefficient from the Beer-Lambert law, commonly set to 0.5 based on empirical studies (Jia et al., 2025)
    
    # loop over the different height bins sizes (e.g. different layer thickness aka ∆z)
    for bin_height in height_bins:
        # delta z = bin_height
        delta_z = bin_height
        
        # Create height bins
        max_height = veg_points['Z'].max()
        bins = np.arange(0.1, max_height + bin_height, bin_height) #3 m original to exclude ground vegetation but we want to include also understorey so start from 0.1m to exclude grass only
        
        # Calculate amount of points in each bin via histogram
        hist, _ = np.histogram(veg_points['Z'], bins=bins)
        
        # number of points above the current height layer
        n_above = total_veg_points - np.cumsum(hist)
        
        # number of points above Z = z + delta z aka n_above(z + ∆z) aka number of points above the current and the next layer
        n_above_dz = np.roll(n_above, -1)
        n_above_dz[-1] = 0  # last bin has no layer above it
        
        print(f"n_above: {n_above}")
        print(f"n_above_dz: {n_above_dz}")
        
        # Calculate VAI using the formula
        valid = (n_above > 0) & (n_above_dz > 0)
        vai = np.sum(-(1/k) * np.log(n_above_dz[valid] / n_above[valid]))
        
        vai_formula = "VAI= Σ -1/k * ln((N_above(z + ∆z)/N_above(z))) over all height layers" #N_above(z + ∆z) is smaler than N_above(z) as points of one more layer are included so lesser points above
        print(f"VAI formula used: {vai_formula}")
    
        vai_results[f'vai_{bin_height}m'] = vai
        print(f"VAI for {bin_height}m bins: {vai:.4f}")

    
    
    ########################### Combine metrics and add to General Results CSV
    general_results_path = f'{main_path}/All_Results.csv'
    
    new_row = {'Square': square,
                f'vai_1m_corr': vai_results['vai_1m'],
                f'vai_2m_corr': vai_results['vai_2m'],
                f'vai_5m_corr': vai_results['vai_5m'],
                f'vai_10m_corr': vai_results['vai_10m'],
                f'vai_15m_corr': vai_results['vai_15m'],
                f'vai_20m_corr': vai_results['vai_20m']
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


##############################################################################################################################
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square_list = ['konigsplatz','rundfunk','nikolai', 'miesbacher', "jakobgelb", "elisabeth"]

    for square in square_list:
        vai_calc(main_path, square)

    
