#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Dec 2025, last updated June 2026

This script calculates various vegetation structure metrics from the processed TLS Point Cloud Data on a per cell basis.
- Height metrics based on cells
- Rumple and Rugosity metrics based on cells


Saves the results in a CSV file for each square and also updates a general results CSV file with the metrics for all squares.
Saves plots

Use it after the preprocessing steps in PreprocessingPointClouds.py

Sources for metrics: 
- Shokirov, S. et al. (2023). “Habitat highs and lows: Using terrestrial and UAV LiDAR for modelling avian species richness and abundance in a restored woodland”. In: Remote Sensing of Environment 285, p. 113326. DOI: 10.1016/j.rse.2022.113326.
- LaRue, E. et al. (2020). “Compatibility of Aerial and Terrestrial LiDAR for Quantifying Forest Structural Diversity". In: Remote Sensing 12, p. 1407. DOI: 10.3390/rs12091407



Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
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




### helper functions for surface area calculation
def calculate_surface_area_from_cells(x_coords, y_coords, z_coords, cell_size):
    """
    Calculate 3D surface area from gridded cells using triangulation.
    
    Parameters:
    -----------
    x_coords : array
        X coordinates of cell centers
    y_coords : array
        Y coordinates of cell centers
    z_coords : array
        Z values (height) at each cell
    cell_size : float
        Size of each grid cell
    
    Returns:
    --------
    surface_area : float
        Total 3D surface area
    """
    from scipy.spatial import Delaunay
    
    # Remove NaN values
    mask = ~np.isnan(z_coords)
    x = x_coords[mask]
    y = y_coords[mask]
    z = z_coords[mask]
    
    if len(x) < 3:
        return np.nan
    
    try:
        # Create 2D points for triangulation
        points_2d = np.column_stack([x, y])
        
        # Triangulate
        tri = Delaunay(points_2d)
        
        # Calculate area of each triangle in 3D
        total_area = 0.0
        for simplex in tri.simplices:
            p1 = np.array([x[simplex[0]], y[simplex[0]], z[simplex[0]]])
            p2 = np.array([x[simplex[1]], y[simplex[1]], z[simplex[1]]])
            p3 = np.array([x[simplex[2]], y[simplex[2]], z[simplex[2]]])
            
            # Calculate triangle area using cross product
            v1 = p2 - p1
            v2 = p3 - p1
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            total_area += area
        
        return total_area
    
    except Exception as e:
        print(f"Error in triangulation: {e}")
        # Fallback: approximate using grid cell method
        # Each cell contributes approximately cell_size^2 to horizontal area
        # For sloped surfaces, we adjust based on local slope
        return len(x) * cell_size * cell_size





#### Main function to calculate metrics per cell for each square
def metrics_per_cell(main_path, square, cell_size=1):
    """Calculate various vegetation structure metrics from the processed TLS Point Cloud Data on a per cell basis for a given square.
    Saves the results in a CSV file for the square and also updates a general results CSV file with the metrics for all squares.
    Saves some plots of the metrics per cell.

    Args:
        main_path (str): The main path to the data directory where the folder of square_FINAL is located
        square (str): The square identifier
        cell_size (int, optional): The size of each grid cell. Defaults to 1 meter.
    Returns:
        None: The function saves the calculated Skewness and Kurtosis in the general results CSV file (All_Results.csv) in the main path of the square. It also prints the results to the console.

    """
    print(f"Calculating metrics per {cell_size}m cell for square: {square}")
    
    ##### Create output folder for metrics if it doesn't exist (for each square for plots)
    metrics_folder = f'{main_path}/0_metrics/{square}_Metrics'
    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)
    
    ##### Fast loading of point cloud
    laz_path = f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'
    memmap_path = f'{main_path}/{square}_FINAL/{square}_pointcelldata.memmap'
    

    result = assign_pointdata_with_cellkey(
        laz_path=laz_path,
        memmap_path=memmap_path,
        cell_size=cell_size,
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
        #"Number_of_Returns": arr["num_of_returns"].astype(np.uint8),
        #"Return_Number": arr["return_number"].astype(np.uint8),
        "Classification": arr["classification"].astype(np.int32)
        })
    print(df.head())
    print(f"Total points loaded: {len(df)}")

    # To convert cell_key back to cell_x/cell_y:
    keys = df["cell_key"].to_numpy(dtype=np.int64)
    cx, cy = key_to_cells(keys)
    
    pd_points = df.copy()
    pd_points["cell_x"] = cx
    pd_points["cell_y"] = cy
    
    print(pd_points.head())
    
    ########################### Calculate Intensity stats per cell
    intensity_stats = df.groupby("cell_key")["Intensity"].agg( "mean").reset_index()
     # To convert cell_key back to cell_x/cell_y:
    keys = intensity_stats["cell_key"].to_numpy(dtype=np.int64)
    cx, cy = key_to_cells(keys)
    intensity_stats["cell_x"] = cx
    intensity_stats["cell_y"] = cy
    
    print(intensity_stats.head())
    print(intensity_stats.head())
    
    
    # Test via visualization
    # plot the grid with mean intensity per cell
    plt.figure(figsize=(10, 8))
    plt.scatter(intensity_stats['cell_x'], intensity_stats['cell_y'], c=intensity_stats['Intensity'], cmap='viridis', s=100)
    plt.colorbar(label='Mean Intensity')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title(f'Mean Intensity per {cell_size}m Cell')
    path = f'{metrics_folder}/{square}_MeanIntensity_per_{cell_size}m_Cell.png'
    plt.savefig(path)
    plt.close()
    #plt.show()

    

    ############################ Calculate basic Z metrics per cell 
    print("Calculating Z metrics per cell...")
    # calculate the number of points per cell
    point_count = pd_points.groupby(['cell_key']).size().reset_index(name='point_count')

    # calculate the mean z value per cell
    mean_Z = pd_points.groupby(['cell_key'])['Z'].mean().reset_index()
    min_Z = pd_points.groupby(['cell_key'])['Z'].min().reset_index()
    max_Z = pd_points.groupby(['cell_key'])['Z'].max().reset_index()
    std_Z = pd_points.groupby(['cell_key'])['Z'].std().reset_index()


    # group pd_points by cell_x and cell_y
    cell_info = mean_Z.copy()
    cell_info.columns = ['cell_key', 'mean_Z']

    # merge the min/max Z values
    cell_info = cell_info.merge(min_Z, on=['cell_key'], suffixes=('', '_min'))
    cell_info = cell_info.merge(max_Z, on=['cell_key'], suffixes=('', '_max'))
    cell_info = cell_info.merge(std_Z, on=['cell_key'], suffixes=('', '_std'))

    # merge the point count
    cell_info = cell_info.merge(point_count, on=['cell_key'])

    cell_info.columns = ['cell_key', 'mean_Z', 'min_Z', 'max_Z', 'std_Z', 'point_count']
    
    # To convert cell_key back to cell_x/cell_y:
    keys = cell_info["cell_key"].to_numpy(dtype=np.int64)
    cx, cy = key_to_cells(keys)
    cell_info["cell_x"] = cx
    cell_info["cell_y"] = cy
    
    # visualize mean Z per cell
    plt.figure(figsize=(10, 8))
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['mean_Z'], cmap='viridis', s=100)
    plt.colorbar(label='Mean Z')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title(f'Mean Z per {cell_size}m Cell {square}')
    #save as pdf file 
    print(f"Saving Mean Z per {cell_size}m Cell plot to PDF...")
    path = f'{metrics_folder}/{square}_MeanZ_per_{cell_size}m_Cell.png'
    plt.savefig(path)
    plt.close()
    #plt.show()
    
    ########################### Canopy Height Model (CHM)
    print("Calculating Canopy Height Model (CHM)...")
    # filter points for Classification 2 or 3
    veg_points = pd_points[pd_points['Classification'].isin([2, 3])]
    
    #calculate average max Z per cell for vegetation points only
    veg_mean_Z = veg_points.groupby(['cell_x', 'cell_y'])['Z'].mean().reset_index()
    veg_mean_Z.columns = ['cell_x', 'cell_y', 'veg_mean_Z']
    cell_info = cell_info.merge(veg_mean_Z, on=['cell_x', 'cell_y'], how='left')
    

    
    # calculate the max Z per cell for vegetation points only
    veg_max_Z = veg_points.groupby(['cell_key'])['Z'].max().reset_index() #LOCH - Leaf On Canopy Height OUTER CANOPY HEIGHT
    veg_max_Z.columns = ['cell_key', 'veg_max_Z']
    cell_info = cell_info.merge(veg_max_Z, on=['cell_key',], how='left')
    cell_info['LOCH'] = cell_info['veg_max_Z']
    
    # calculate the std of height in each cell for vegetation points only
    veg_std_Z = veg_points.groupby(['cell_x', 'cell_y'])['Z'].std().reset_index() # for rugosity
    veg_std_Z.columns = ['cell_x', 'cell_y', 'veg_std_Z']
    cell_info = cell_info.merge(veg_std_Z, on=['cell_x', 'cell_y'], how='left')
    
    print(cell_info.head())
    
    
    # Visualize the Canopy Height Model
    plt.figure(figsize=(10, 8))
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['veg_max_Z'], cmap='Greens', s=100)
    plt.colorbar(label='Canopy Height (m)')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title(f'Canopy Height Model per {cell_size}m Cell')
    path = f'{metrics_folder}/{square}_CHM_per_{cell_size}m_Cell.png'
    plt.savefig(path)
    plt.close()
    #plt.show()
    
    # calculate rough understorey max Z (points below 2 m)
    rough_understorey_points = veg_points[veg_points['Z'] < 2]
    rough_understorey_max_Z = rough_understorey_points.groupby(['cell_key'])['Z'].max().reset_index()
    rough_understorey_max_Z.columns = ['cell_key', 'rough_understorey_max_Z']
    cell_info = cell_info.merge(rough_understorey_max_Z, on=['cell_key'], how='left')
    
    rough_understorey_mean_Z = rough_understorey_points.groupby(['cell_x', 'cell_y'])['Z'].mean().reset_index()
    rough_understorey_mean_Z.columns = ['cell_x', 'cell_y', 'rough_understorey_mean_Z']
    cell_info = cell_info.merge(rough_understorey_mean_Z, on=['cell_x', 'cell_y'], how='left')
    

    # visualize rough understorey max Z per cell
    plt.figure(figsize=(10, 8))
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['rough_understorey_max_Z'], cmap='Oranges', s=100)
    plt.colorbar(label='Rough Understorey Max Z (m)')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title(f'Rough Understorey Max Z per {cell_size}m Cell')
    path = f'{metrics_folder}/{square}_RoughUnderstoreyMaxZ_per_{cell_size}m_Cell.png'
    plt.savefig(path)
    plt.close()
    #plt.show()
    
    ########################### Calculate Vegetation Density Metric (VDM)
    print("Calculating Vegetation Density Metric (VDM)...")
    # VDM = (number of vegetation points (class 2 or 3) / total number of points) * 100
    veg_point_count = veg_points.groupby(['cell_key']).size().reset_index(name='veg_point_count')
    cell_info = cell_info.merge(veg_point_count, on=['cell_key'], how='left')
    cell_info['veg_point_count'] = cell_info['veg_point_count'].fillna(0)
    cell_info['VDM'] = (cell_info['veg_point_count'] / cell_info['point_count']) * 100  
    # visualize VDM
    plt.figure(figsize=(10, 8))
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['VDM'], cmap='YlGn', s=100)
    plt.colorbar(label='Vegetation Density Metric (VDM)')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')        
    plt.title(f'Vegetation Density Metric per {cell_size}m Cell')
    path = f'{metrics_folder}/{square}_VDM_per_{cell_size}m_Cell.png'
    plt.savefig(path)
    plt.close()
    #plt.show()
    
    
    ######################### Canopy cover of square
    print("Calculating Canopy Cover of the square...")
    # aka how many pixel have veg of total pixels
    total_cells = cell_info.shape[0]
    cells_with_veg = cell_info[cell_info['veg_point_count'] > 0].shape[0]
    canopy_cover_percentage = (cells_with_veg / total_cells) * 100
    print(f"Canopy Cover of the square: {canopy_cover_percentage:.2f} %")
    
    # understorey cover of square
    cells_with_understorey = cell_info[cell_info['rough_understorey_max_Z'] > 0].shape[0]
    understorey_cover_percentage = (cells_with_understorey / total_cells) * 100
    print(f"Understorey Cover of the square: {understorey_cover_percentage:.2f} %")
    
    
    cells_with_canopy = cell_info[cell_info['veg_max_Z'] > 2].shape[0]
    canopy_cover_percentage_2m = (cells_with_canopy / total_cells) * 100
    print(f"Canopy Cover of the square (above 2m): {canopy_cover_percentage_2m:.2f} %")
    
    
    
    
    ########################## Canopy Relief Ratio (CRR) and Canopy Cover (CC) per cell
    print("Calculating Canopy Relief Ratio (CRR) and Canopy Cover (CC) per cell...")
    # CRR
    # CRR = zmean − zmin / zmax − zmin .
    cell_info['CRR'] = (cell_info['mean_Z'] - cell_info['min_Z']) / (cell_info['max_Z'] - cell_info['min_Z'])

    #plot histogramm of CRR
    plt.figure(figsize=(10, 6))
    plt.hist(cell_info['CRR'].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Canopy Relief Ratio (CRR)')
    plt.xlabel('CRR')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    path = f'{metrics_folder}/{square}_CRR_Histogram.png'
    plt.savefig(path)
    plt.close()
    #plt.show()  

    # CC
    # We used the structural metric CC which is the proportion of the ground covered by forest canopy (above 2 m)
    # in units of percent. CC is approximately the inverse of gap fraction, another commonly estimated forest structural property. 
    # CC was calculated for each pixel using lidar returns as:
    # CC = (1 - (sum (i=1 to n) Zi whereZ>2) / n) * 100
    # n calculate number of points above 2 m per cell via pd_points
    number_points_above_2 = pd_points[pd_points['Z'] > 2].groupby(['cell_key']).size().reset_index(name='number_points_above_2')
    # sum zi caclulate sum of heights of points above 2 m
    sum_points_above_2 = pd_points[pd_points['Z'] > 2].groupby(['cell_key'])['Z'].sum().reset_index(name='sum_points_above_2')
    cell_info = cell_info.merge(number_points_above_2, on=['cell_key'], how='left')
    cell_info = cell_info.merge(sum_points_above_2, on=['cell_key'], how='left')
    cell_info['CoverPercentage'] = ((cell_info['number_points_above_2'] / cell_info['point_count'])) * 100
  
    cell_info['CC'] = (1 - (cell_info['sum_points_above_2'] / cell_info['point_count'])) * 100
    cell_info["CC_corr"] = (1- (cell_info['number_points_above_2'] / cell_info['point_count'])) * 100
    #visualize the CRR and CC
    plt.figure(figsize=(16, 6))

    # Subplot for CRR
    plt.subplot(1, 3, 1)
    plt.hist(cell_info['CRR'].dropna(), bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Canopy Relief Ratio (CRR)')
    plt.xlabel('CRR')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Subplot for CC
    plt.subplot(1, 3, 2)
    plt.hist(cell_info['CC'].dropna(), bins=30, color='green', alpha=0.7)
    plt.title('Histogram of Canopy Cover (CC)')
    plt.xlabel('CC')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    # Subplot for CC_cor
    plt.subplot(1, 3, 3)
    plt.hist(cell_info['CC_corr'].dropna(), bins=30, color='green', alpha=0.7)
    plt.title('Histogram of Canopy Cover (CC_corr)')
    plt.xlabel('CC_corr')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    plt.tight_layout()
    
    path = f'{metrics_folder}/{square}_CRR_CC_Histograms.png'
    plt.savefig(path)
    plt.close()
    #plt.show()

    # and now visualise as xy grid

    plt.figure(figsize=(16, 6))

    # Subplot for CRR
    plt.subplot(1, 2, 1)
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['CRR'], cmap='viridis', s=100)
    plt.colorbar(label='CRR')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title('Canopy Relief Ratio (CRR)')

    # Subplot for CC
    plt.subplot(1, 2, 2)
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['CC'], cmap='viridis', s=100)
    plt.colorbar(label='CC')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title('Canopy Cover (CC)')
    
    # Subplot for CC_corr
    plt.subplot(1, 2, 2)
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['CC_corr'], cmap='viridis', s=100)
    plt.colorbar(label='CC_corr')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title('Canopy Cover (CC_corr)')
    
    #save
    path = f'{metrics_folder}/{square}_CRR_CC_ScatterPlots.png'
    plt.savefig(path)
    plt.tight_layout()
    plt.close()
    #plt.show()


    #plot the xy of cover
    plt.figure(figsize=(12, 6))
    plt.scatter(cell_info['cell_x'], cell_info['cell_y'], c=cell_info['CoverPercentage'], cmap='viridis', s=100)
    plt.colorbar(label='Cover Percentage')
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title('Canopy Cover Percentage')
    path = f'{metrics_folder}/{square}_CanopyCoverPercentage_ScatterPlot.png'
    plt.savefig(path)
    plt.close()
    #plt.show()
    
    
    
    ############################## Rugosity  metrics per cell
    print("Calculating Rugosity ...")
    #Standard deviation of LOCH
    top_rugosity = cell_info['LOCH'].std() 
    
    # std of height in each cell and of that the standard deviation is the rugosity
    rugosity_std = cell_info['veg_std_Z'].std()  # this methods is from MacArthur and MacArthur 1961, but it is not very good as it is influenced by the number of points in each cell and the height of the canopy, so it is not a good measure of canopy roughness as it can be high in areas with low vegetation cover and low in areas with high vegetation cover.
    rugosity_max = top_rugosity # also try with max height per cell as measure of canopy roughness
    print(f"Top Rugosity (std of LOCH): {top_rugosity:.3f} m")
    print(f"Rugosity (std of veg std Z): {rugosity_std:.3f} m")
    print(f"Rugosity (std of veg max Z): {rugosity_max:.3f} m")
    
    ############################## Rumple  metrics per cell
    print("Calculating Rumple Index ...")
    # Create actual X,Y coordinates for cells (convert back from cell indices)
    cell_info['X_coord'] = cell_info['cell_x'] * cell_size + cell_size / 2
    cell_info['Y_coord'] = cell_info['cell_y'] * cell_size + cell_size / 2

    # For rumple index with height-above-ground data:
    # Ground surface is flat at Z=0, so ground_surface_area is just the 2D horizontal area
    # Canopy surface uses the LOCH values

    # Calculate 3D canopy surface area
    canopy_surface_area = calculate_surface_area_from_cells(
        cell_info['X_coord'].values,
        cell_info['Y_coord'].values,
        cell_info['LOCH'].values,  # Use LOCH since data is already height above ground
        cell_size
    )

    # Ground surface area (flat at Z=0)
    # This is simply the 2D horizontal area = number_of_cells * cell_size^2
    valid_cells = cell_info['LOCH'].notna().sum()
    ground_surface_area = valid_cells * cell_size * cell_size

    rumple_index = canopy_surface_area / ground_surface_area
    print(f"Rumple Index (Canopy SA / Ground SA): {rumple_index:.3f}")

    # Add metrics to cell_info
    cell_info['top_rugosity'] = top_rugosity
    cell_info['rugosity_std'] = rugosity_std
    cell_info['rugosity_max'] = rugosity_max
    cell_info['rumple_index'] = rumple_index
    
    ############################## POROSITY metrics per cell 
    # number of cells without vegetation points divided by total number of cells
    total_cells = cell_info.shape[0]
    cells_without_veg = cell_info[cell_info['veg_point_count'] == 0].shape[0]
    porosity = (cells_without_veg / total_cells) * 100
    print(f"Porosity of the square: {porosity:.2f} %")



    ############################## prepare final cell_info DataFrame for saving
    cell_info["cell_size"] = cell_size
    
    print(cell_info.head())

    #save the cell_info DataFrame to a CSV file
    metrics_path = f'{metrics_folder}/{square}_cell_info_{cell_size}.csv'
    if not os.path.exists(metrics_folder):
        os.makedirs(metrics_folder)
    cell_info.to_csv(metrics_path, index=False)
    
    

    print(f"Metrics per cell saved to {metrics_path}")
    
    
    # Combine metrics and add to General Results CSV
    general_results_path = f'{main_path}/All_Results.csv'

    new_row = {'Square': square,
                f'Canopy_Coverage_Percentage_{cell_size}': canopy_cover_percentage,
                f'Understorey_Coverage_Percentage_{cell_size}': understorey_cover_percentage,
                f"Canopy_Coverage_Percentage_Above2m_{cell_size}": canopy_cover_percentage_2m,
                f'Mean_CRR_{cell_size}': cell_info['CRR'].mean(),
                f'Min_CRR_{cell_size}': cell_info['CRR'].min(),
                f'Max_CRR_{cell_size}': cell_info['CRR'].max(),
                f'Mean_CC_{cell_size}': cell_info['CC'].mean(),
                f'Min_CC_{cell_size}': cell_info['CC'].min(),
                f'Max_CC_{cell_size}': cell_info['CC'].max(),
                f"Mean_CC_corr_{cell_size}": cell_info['CC_corr'].mean(),
                f"Min_CC_corr_{cell_size}": cell_info['CC_corr'].min(),
                f"Max_CC_corr_{cell_size}": cell_info['CC_corr'].max(),
                f"Median_CC_corr_{cell_size}": cell_info['CC_corr'].median(),
                f'Mean_Z_{cell_size}': cell_info['mean_Z'].mean(),
                f'MinZ_{cell_size}': cell_info['min_Z'].min(),
                f'MaxZ_{cell_size}': cell_info['max_Z'].max(),
                f'Std_Z_{cell_size}': cell_info['std_Z'].mean(),
                f'Mean_VDM_{cell_size}': cell_info['VDM'].mean(),
                f'Min_VDM_{cell_size}': cell_info['VDM'].min(),
                f'Max_VDM_{cell_size}': cell_info['VDM'].max(),
                f'Mean_Points_per_Cell_{cell_size}': cell_info['point_count'].mean(),
                f'Mean_Veg_Points_per_Cell_{cell_size}': cell_info['veg_point_count'].mean(),
                f'Max_Veg_Height_{cell_size}': cell_info['veg_max_Z'].max(),
                f'Mean_Veg_Height_{cell_size}': cell_info['veg_mean_Z'].mean(),
                f'Max_Understorey_Height_{cell_size}': cell_info['rough_understorey_max_Z'].max(),
                f'Mean_Understorey_Height_{cell_size}': cell_info['rough_understorey_mean_Z'].mean(),
                f"MOCH_{cell_size}": cell_info['LOCH'].mean(),
                f'Top Rugosity_{cell_size}': top_rugosity,
                f'Rugosity_VegStdZ_{cell_size}': rugosity_std,
                f"Rugosity_VegMaxZ_{cell_size}": rugosity_max,
                f'Rumple_Index_{cell_size}': rumple_index,
                f'Porosity_{cell_size}': porosity
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



################################################
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square = 'nikolai'
    square_list = [ "nikolai", "rundfunk", "konigsplatz", "miesbacher", "jakobgelb","elisabeth"]  # Replace with your square identifier
    for square in square_list:
        metrics_per_cell(main_path, square, cell_size=1)
    
