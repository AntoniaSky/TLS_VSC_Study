# -*- coding: utf-8 -*-
"""Skewness and Kurtosis of the vertical distribution of vegetation points per cloud.
Skewness and Kurtosis calculated from TLS data of a public square. 
Followed as in Shokirov et al. (2023) 

Source:
- Shokirov, S., et al. (2023). “Habitat highs and lows: Using terrestrial and UAV LiDAR for modelling avian species richness and abundance in a restored woodland”. In: Remote Sensing of Environment 285, p. 113326. DOI: 10.1016/j.rse.2022.113326.


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
Originally created Dec 2025, last updated June 2026
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
from scipy.stats import skew, kurtosis

from definitions_a import assign_pointdata_with_cellkey,key_to_cells

    
    
def skew_kurt(main_path: str, square: str):
    """Calculate the Skewness and the Kurtosis of the Z data of the vegetation of the Public Square. 
    Derived from TLS Point Clouds
    Saves results in csv.
    

    Args:
        main_path (str): Path to the main directory where the final-folder of the square is located (after preprocessing and metric calculation per cell)
        square (str): Name (shortform) of the square to process (e.g. "alpenplatz", "nikolai", "rundfunk", "konigsplatz", "miesbacher", "jakobgelb")
        
    Returns:
        None: The function saves the calculated Skewness and Kurtosis in the general results CSV file (All_Results.csv) in the main path of the square. It also prints the results to the console.
    """
    
    
    print(f"Calculating Skewness and Kurtosis for square: {square}")
    
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
    
    #dtype = create_point_dtype(np.float32, True)  # from your writer
    #per_point = np.memmap("mycloud_pointdata.memmap", mode="r", dtype=dtype, shape=(total_points,))
    
    
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

    ######## Calculate Skewness and Kurtosis for whole cloud
    print("Calculating Skewness and Kurtosis for the entire vegetation point cloud...")
    skew_kurt_results = {}  
    #total vegetation points  
    total_veg_points = len(veg_points)
    
    # Skewness
    skewness = skew(veg_points['Z'], bias=False)
    skew_kurt_results['skewness'] = skewness
    print(f"Skewness of vegetation Z of {square}: {skewness}")
    
    # Kurtosis
    kurt = kurtosis(veg_points['Z'], bias=False)
    skew_kurt_results['kurtosis'] = kurt
    print(f"Kurtosis of vegetation Z of {square}: {kurt}")
    
   
    
    
    
    ######## Combine metrics and add to General Results CSV
    general_results_path = f'{main_path}/All_Results.csv'
    columns = ['Square', 
               f'skewness', f'kurtosis'
               ]
    
    new_row = {'Square': square,
                f'skewness': skew_kurt_results['skewness'],
                f'kurtosis': skew_kurt_results['kurtosis']
                }
    
    if os.path.exists(general_results_path):
        general_results_df = pd.read_csv(general_results_path)
        # check if the columns exist, if not create them
        for col in columns:
            if col not in general_results_df.columns:
                general_results_df[col] = np.nan
    else:
        general_results_df = pd.DataFrame(columns=columns)
    
    general_results_df.loc[len(general_results_df)] = new_row
    general_results_df.to_csv(general_results_path, index=False)
    print(f"General results updated at {general_results_path}")



########################################
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square = 'miesbacher'
    square_list = ['jakobgelb',  'miesbacher',
                   'konigsplatz', 'rundfunk',
                   'nikolai', "elisabeth"]
    for square in square_list:
        skew_kurt(main_path, square)
    
