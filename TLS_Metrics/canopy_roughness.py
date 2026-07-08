#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates the canopy roughness metric (CR) based on the method described in Huerta et al. (2020) cited in Shokirov et al. (2023). 
The canopy roughness is calculated by fitting a local plane to the neighborhood of each canopy point and measuring the distance from the point to the plane. 
The CR metric is then derived from the distribution of these distances.

Use it after the preprocessing steps in PreprocessingPointClouds.py 

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)

Sources:
- Herrero-Huerta, M., et al. (2020). “Canopy Roughness: A New Phenotypic Trait to Estimate Aboveground Biomass from Unmanned Aerial System”. In: Plant Phenomics 2020, p. 6735967. DOI: 10.34133/2020/6735967.
- Shokirov, S., et al. (2023). “Habitat highs and lows: Using terrestrial and UAV LiDAR for modelling avian species richness and abundance in a restored woodland”. In: Remote Sensing of Environment 285, p. 113326. DOI: 10.1016/j.rse.2022.113326.

Originally created Spring 2026, last updated June 2026
"""


#IMPORTS
from sklearn.neighbors import NearestNeighbors
from scipy.stats import iqr
import os
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
from typing import Tuple, Dict

from yaml import scan

from definitions_a import assign_pointdata_with_cellkey

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize



import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.stats import iqr
from joblib import Parallel, delayed
import warnings



##################
# Definition & helper functions
##################
def calculate_canopy_roughness(canopy_points: pd.DataFrame, search_radius: float = 0.10, 
                               height_threshold: float = 0.5, n_jobs: int = -1,
                               batch_size: int = 25_000) -> tuple[float, np.array]:
    """
    Calculate canopy roughness using local plane fitting (OPTIMIZED VERSION).
    
    Parameters:
    -----------
    canopy_points : DataFrame
        Point cloud with X, Y, Z columns (Z should be height above ground)
        already filtered to include only canopy points (e.g. Classification 2 and 3)
    search_radius : float
        Radius for neighborhood search in meters (default: 0.10m) 
    height_threshold : float
        Minimum height to consider as canopy (default: 0.5m)
    n_jobs : int
        Number of parallel jobs (-1 for all cores, default: -1)
    batch_size : int
        Process points in batches to reduce memory usage (default: 25_000)
    
    Returns:
    --------
    canopy_roughness : float
        CR metric (IQR**median) in m^2
    point_roughness : array
        Roughness value for each point in meters
    """        
    
    print(f"Calculating canopy roughness with {search_radius}m radius...")
    
    if len(canopy_points) < 10:
        print("Not enough canopy points for roughness calculation")
        return np.nan, np.array([])
    
    # Extract XYZ coordinates as contiguous array for better performance
    coords = np.ascontiguousarray(canopy_points[['X', 'Y', 'Z']].values, dtype=np.float32)
    n_points = len(coords)
    
    print(f"Processing {n_points:,} canopy points...")
    
    # Build KD-tree
    print("Building KD-tree...")
    nbrs = NearestNeighbors(radius=search_radius, algorithm='kd_tree', n_jobs=n_jobs).fit(coords)
    
    # Process in batches to avoid memory issues
    print("Calculating point roughness in parallel batches...")
    point_roughness = np.zeros(n_points, dtype=np.float32)
    
    n_batches = (n_points + batch_size - 1) // batch_size
    
    start_time = pd.Timestamp.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_points)
        
        print(f"  Batch {batch_idx + 1}/{n_batches}: points {start_idx:,} to {end_idx:,}")
        
        # Get batch coordinates
        batch_coords = coords[start_idx:end_idx]
        
        # Find neighbors for this batch
        distances, indices = nbrs.radius_neighbors(batch_coords)
        
        # Process batch in parallel
        batch_roughness = Parallel(n_jobs=n_jobs, backend='threading')(
            delayed(calculate_point_roughness)(coords[i + start_idx], coords[neighbor_indices])
            for i, neighbor_indices in enumerate(indices)
        )
        
        point_roughness[start_idx:end_idx] = batch_roughness
    
    end_time = pd.Timestamp.now()
    print(f"Finished point roughness calculation at {end_time.strftime('%Y-%m-%d %H:%M:%S')}, Duration: {(end_time - start_time)}")
    
    # Remove NaN values
    valid_roughness = point_roughness[~np.isnan(point_roughness)]
    
    if len(valid_roughness) == 0:
        print("No valid roughness values calculated")
        return np.nan, point_roughness
    
    # Calculate canopy roughness: CR = IQR**median (FIXED: was using ^ instead of **)
    median_roughness = np.median(valid_roughness)
    iqr_roughness = iqr(valid_roughness)
    
    if median_roughness == 0:
        print("Warning: Median roughness is zero")
        canopy_roughness = np.nan
    else:
        canopy_roughness = iqr_roughness ** median_roughness  # FIXED: ** not ^
    
    print(f"Median point roughness: {median_roughness:.4f} m")
    print(f"IQR of point roughness: {iqr_roughness:.4f} m")
    print(f"Canopy Roughness (CR): {canopy_roughness:.4f} m^2")
    
    return canopy_roughness, point_roughness


def calculate_point_roughness(point, neighbors):
    """
    Calculate roughness for a single point (optimized for parallel processing).
    
    Parameters:
    -----------
    point : array (3,)
        The query point [x, y, z]
    neighbors : array (N, 3)
        Neighbor points including the query point
    
    Returns:
    --------
    roughness : float
        Distance from point to fitted plane
    """
    if len(neighbors) < 3:
        return np.nan
    
    return fit_plane_and_get_distance(point, neighbors)

def fit_plane_and_get_distance(point, neighbors):
    """
    Fit a plane to neighbor points and calculate distance from point to plane.
    Optimized version with better numerical stability.
    """
    # Center the neighbors
    centroid = np.mean(neighbors, axis=0)
    centered = neighbors - centroid
    
    # Use SVD to find the plane normal
    try:
        # rcond=None for better compatibility
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2, :]  # Normal vector of the plane
        
        # Distance from point to plane
        distance = np.abs(np.dot(normal, point - centroid))
        
        return distance
    
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

def visualize_canopy_roughness(canopy_points, point_roughness, main_path, square, 
                                height_threshold=0.5, max_roughness_display=0.05):
    """
    Visualize point-level roughness values.
    
    Parameters:
    -----------
    canopy_points : DataFrame
        Canopy point cloud - should already be filtered to include only canopy points (e.g. Classification 2 and 3)
    point_roughness : array
        Roughness values for canopy points
    main_path : str
        Base path for saving
    square : str
        Square identifier
    height_threshold : float
        Height threshold used for canopy filtering
    max_roughness_display : float
        Maximum roughness value for color scaling (for better visualization)
    """
    
    canopy_points['roughness'] = point_roughness
    
    # Remove NaN values for plotting
    
    valid_points = canopy_points[~np.isnan(canopy_points['roughness'])]
    
    if len(valid_points) == 0:
        print("No valid points to visualize")
        return
    
    
    #sample 300_000 points for visualization if there are more than that
    if len(valid_points) > 300_000:
        valid_points = valid_points.sample(300_000, random_state=42)
            # Clip roughness for better visualization (optional, adjust as needed)
        roughness_display = np.clip(valid_points['roughness'], 0, max_roughness_display)
    
    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Top view (X-Y) colored by roughness
    scatter1 = axes[0, 0].scatter(valid_points['X'], valid_points['Y'], 
                                   c=roughness_display, cmap='RdYlGn_r', 
                                   s=1, alpha=0.6)
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('Top View - Point Roughness')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(scatter1, ax=axes[0, 0], label='Roughness (m)')
    
    # Plot 2: Side view (X-Z) colored by roughness
    scatter2 = axes[0, 1].scatter(valid_points['X'], valid_points['Z'], 
                                   c=roughness_display, cmap='RdYlGn_r', 
                                   s=1, alpha=0.6)
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Z (Height, m)')
    axes[0, 1].set_title('Side View - Point Roughness')
    plt.colorbar(scatter2, ax=axes[0, 1], label='Roughness (m)')
    
    # Plot 3: Histogram of roughness values
    axes[1, 0].hist(valid_points['roughness'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.median(valid_points['roughness']), 
                       color='red', linestyle='--', linewidth=2, label='Median')
    q25 = np.percentile(valid_points['roughness'], 25)
    q75 = np.percentile(valid_points['roughness'], 75)
    axes[1, 0].axvline(q25, color='orange', linestyle='--', linewidth=1, label='Q25')
    axes[1, 0].axvline(q75, color='orange', linestyle='--', linewidth=1, label='Q75')
    axes[1, 0].set_xlabel('Roughness (m)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Point Roughness')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    axes[1, 1].axis('off')
    median_r = np.median(valid_points['roughness'])
    iqr_r = iqr(valid_points['roughness'])
    cr = iqr_r**median_r if median_r > 0 else np.nan
    
    stats_text = f"""
    Canopy Roughness Statistics
    {'='*40}
    
    Number of points: {len(valid_points):,}
    Search radius: 0.10 m
    
    Point Roughness (m):
      - Median: {median_r:.4f}
      - Mean: {np.mean(valid_points['roughness']):.4f}
      - Std: {np.std(valid_points['roughness']):.4f}
      - Min: {np.min(valid_points['roughness']):.4f}
      - Max: {np.max(valid_points['roughness']):.4f}
      - IQR: {iqr_r:.4f}
    
    Canopy Roughness (CR):
      CR = IQR ^ median = {cr:.4f} m² #python operation is ** not ^ but here is just text so it doesn't matter
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
    
    plt.tight_layout()
    metrics_dir = f'{main_path}/0_metrics/{square}_Metrics'
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
    path = f'{metrics_dir}/{square}_CanopyRoughness.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved canopy roughness visualization to {path}")

##################
# main functions
##################
def canopy_roughness_metric(main_path, square, height_threshold=0.5):
    """Calculate canopy roughness metric for a given square. 
    Saves results to a general CSV and visualizes point-level roughness values.
    (Duration: potentially long for large squares)

    Args:
        main_path (str): The path to the main directory containing the data and the folders to the final processed point cloud data.
        square (str): The identifier for the square area.
        height_threshold: float, Minimum height to consider as canopy (default: 0.5m)

    Returns:
        None: The function saves the calculated VAI values to the general results CSV file and does not return any value.

    """
    
    # Load the point cloud data 
    print("Loading point cloud data...")
    laz_path = f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'
    memmap_path = f'{main_path}/{square}_FINAL/{square}_pointcelldata.memmap'    

    result = assign_pointdata_with_cellkey(
        laz_path=laz_path,
        memmap_path=memmap_path,
        cell_size=20,
        chunk_size=5_000_000,
        coord_dtype=np.float32,
    )
        
    arr = result["per_point"] 
    pd_points = pd.DataFrame({
        "cell_key": arr["cell_key"],
        "X": arr["x"].astype(np.float64),
        "Y": arr["y"].astype(np.float64),
        "Z": arr["z"].astype(np.float64),
        #"Intensity": arr["intensity"].astype(np.float32),
        #"Number_of_Returns": arr["num_of_returns"].astype(np.uint8),
        #"Return_Number": arr["return_number"].astype(np.uint8),
        "Classification": arr["classification"].astype(np.int32)
        })
    print(pd_points.head())
    print(f"Total points loaded: {len(pd_points)}")
    
    
    # Filter for canopy points only
    if "Classification" in pd_points.columns:
        canopy_points = pd_points[pd_points['Classification'].isin([2, 3])].copy()
        del pd_points
    else: 
        print("Warning: No Classification column found, using height threshold")
        canopy_points = pd_points[pd_points['Z'] > height_threshold].copy()
    
    
    ############################ Calculate Canopy Roughness
    print("\n" + "="*60)
    print("CALCULATING CANOPY ROUGHNESS")
    print("="*60)
    
    canopy_roughness, point_roughness = calculate_canopy_roughness(
        canopy_points, 
        search_radius=0.10,  # 10 cm as specified in the paper
        height_threshold=height_threshold  # Adjust based on your data
    )
    
    
    # dict and then saving
    
    
        #%% Save results
    # Save results
    result_dict = {
        'Square': square,
        'canopy_roughness_m2': canopy_roughness,
        'median_point_roughness_m': np.median(point_roughness[~np.isnan(point_roughness)]),
        'iqr_point_roughness_m': iqr(point_roughness[~np.isnan(point_roughness)])   
    }
    
    print(f"RESULTS FOR CANOPY ROUGHNESS for {square}:")
    print(result_dict)
    
    
    #save to General Results CSV
    #general result_csv_path
    general_result_csv_path = f'{main_path}/All_Results.csv'
    
    if os.path.exists(general_result_csv_path):
        result_df = pd.read_csv(general_result_csv_path)
        # check if the columns exist, if not create them
        for col in result_dict.keys():
            if col not in result_df.columns:
                result_df[col] = np.nan
    else:
        result_df = pd.DataFrame(columns=result_dict.keys())
        
    print("Updating general results CSV...")
    print(result_dict)
    # Append new results
    result_df = pd.concat([result_df, pd.DataFrame([result_dict])], ignore_index=True)
    result_df.to_csv(general_result_csv_path, index=False)


    print(f"start visualization of canopy roughness for {square}...")
    
    # Visualize
    visualize_canopy_roughness(
        canopy_points, 
        point_roughness, 
        main_path, 
        square,
        height_threshold=height_threshold
    )
    

    print(f"CANOPY ROUGHNESS calculation completed for square: {square}")

#####################################        
if __name__ == "__main__":
    main_path = "/Volumes/T7_Shield_A/msc/"
    square_list = [#"alpenplatz", "nikolai", "rundfunk", "konigsplatz", "miesbacher", "jakobgelb", 
                   "elisabeth"]  # Replace with your square identifiers
    for square in square_list:
        canopy_roughness_metric(main_path, square)
    print("CANOPY ROUGHNESS - All done!")