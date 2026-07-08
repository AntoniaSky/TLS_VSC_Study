# -*- coding: utf-8 -*-
"""
This script estimates the volume of the vegetation based on the formula of Shokirov et al. (2023) and the processed TLS point cloud data.

Total vegetation volume (m3): number of 0.5 m3 voxels divided by 8 (ground points excluded)

Source:
- Shokirov, S., et al. (2023). “Habitat highs and lows: Using terrestrial and UAV LiDAR for modelling avian species richness and abundance in a restored woodland”. In: Remote Sensing of Environment 285, p. 113326. DOI: 10.1016/j.rse.2022.113326.


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
Originally created Jan 2026, last updated July 2026
"""


#Total vegetation volume (m3) – number of 0.5 m3 voxels divided by 8 (ground points excluded)

##import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar
import seaborn as sns
import open3d as o3d

##Main function to calculate the vegetation volume
def volumeVegetation(square: str, main_path: str):
    """Calculate the vegetation volume for a given square using TLS point cloud data.
    Total vegetation volume (m3): number of 0.5 m3 voxels divided by 8 (ground points excluded)

    Args:
        square (str): square identifier (e.g., 'alpenplatz', 'nikolai', etc.)
        main_path (str): path to the directory containing the final preprocessed TLS point cloud data
        
    Returns:
        None: Saves the results to a CSV file and generates plots for vegetation volume per height class and cumulative vegetation volume.
    """
    print(f"Processing square: {square}")
    print(f"Main path: {main_path}")


    # Load the point cloud data after preprocessing
    file_path = f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'
    laz, pd_laz = load_laz_file(file_path)
    
    # folder and paths
    metrics_folder = f'{main_path}/{square}_Metrics'
    os.makedirs(metrics_folder, exist_ok=True)

    #general result_csv_path
    result_csv_path = f'{main_path}/All_Results.csv'
    
    columns = ['veg_volume', 'volume_voxel_size',
               'square_area', 'vol_to_area', 
               'Square']
    if os.path.exists(result_csv_path):
        result_df = pd.read_csv(result_csv_path)
        # check if the columns exist, if not create them
        for col in columns:
            if col not in result_df.columns:
                result_df[col] = np.nan
    else:
        result_df = pd.DataFrame(columns=columns)

    # below 0 is genullt z=0
    pd_laz.loc[pd_laz['Z'] < 0, 'Z'] = 0

    # make classification 3 to 2 (low veg to high veg so all veg is 2)
    pd_laz.loc[pd_laz['Classification'] == 3, 'Classification'] = 2

    #only keep vegetation points
    pd_laz = pd_laz[pd_laz['Classification'] == 2]
    
    #Calculate the Vegetation Volume
    voxel_volume = 0.5 # m3 (Shokirov)
    voxel_size = voxel_volume ** (1/3)  # edge length of the voxel cube
    print(f"Voxel size: {voxel_size} m")

    # Assign points to voxel indices
    print("Calculating vegetation volume...")
    pd_laz['vx'] = (pd_laz['X'] // voxel_size).astype(int)
    pd_laz['vy'] = (pd_laz['Y'] // voxel_size).astype(int)
    pd_laz['vz'] = (pd_laz['Z'] // voxel_size).astype(int)
    
    # Get number of voxels 
    veg_voxels_unique = pd_laz[['vx', 'vy', 'vz']].drop_duplicates()
    

    # Calculate total vegetation volume
    veg_volume = len(veg_voxels_unique) / 8
    print(f"Vegetation volume for voxel size {voxel_size} m: {veg_volume:.3f} m³")
    
    # ------------ make volume to area ratio --------------
    # Calculate area of the square # todo still needs to be adapted for non-rectangular areas
    print("Calculating square area...")
    x_range = pd_laz['X'].max() - pd_laz['X'].min()
    y_range = pd_laz['Y'].max() - pd_laz['Y'].min()
    square_area = x_range * y_range
    print(f"Square area: {square_area:.2f} m²")

    # Volume-to-area ratio
    vol_to_area = veg_volume / square_area
    print(f"Vegetation volume per m²: {vol_to_area:.4f} m³/m²")



    # lets do a line plot with the vegetation volume per height class
    # Create height classes
    #height of layers 
    height_of_layers = 1.0  # 1 meter height classes
    height_bins = np.arange(pd_laz['Z'].min(), pd_laz['Z'].max(), height_of_layers)
    bin_center = (height_bins[:-1] + height_bins[1:]) / 2
    bin_minimum = height_bins[:-1]
    bin_maximum = height_bins[1:]
    pd_laz['height_class'] = pd.cut(pd_laz['Z'], bins=height_bins,
                                    labels=bin_center, include_lowest=True, right=False)

    # Group by height class and count vegetation points
    veg_volume_per_height = pd_laz[pd_laz['Classification'] == 2].groupby('height_class').size().reset_index(name='veg_count')
    # Calculate volume for each height class
    veg_volume_per_height['veg_volume'] = veg_volume_per_height['veg_count'] * voxel_volume

    veg_volume_per_height['height_class_min'] = bin_minimum
    veg_volume_per_height['height_class_max'] = bin_maximum
    #add a column for height of layers
    veg_volume_per_height['height_of_layer'] = height_of_layers


    ##
    # Plotting a line plot of vegetation volume per height class
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=veg_volume_per_height, y='veg_volume', x='height_class', marker='o')
    plt.title(f'{square} - Vegetation Volume per Height Class')
    plt.xlabel('Height Class (m)')
    plt.ylabel('Vegetation Volume (m³)')
    plt.grid(True)
    plt.savefig(f'{main_path}/{square}_Metrics/{square}_VegVolume_height{height_of_layers}.pdf')
    plt.show()  
    # %% now lets do commulative sum of vegetation volume per height class
    veg_volume_per_height['cumulative_volume'] = veg_volume_per_height['veg_volume'].cumsum()
    # Plotting a line plot of cumulative vegetation volume per height class
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=veg_volume_per_height, y='cumulative_volume', x='height_class_min', marker='o')
    plt.title(f'{square} - Cumulative Vegetation Volume per Height Class')
    plt.xlabel('Height Class (m)')
    plt.ylabel('Cumulative Vegetation Volume (m³)')
    plt.grid(True)
    #save the figure
    plt.savefig(f'{main_path}/{square}_Metrics/{square}_VegVolume_Cumulative_height{height_of_layers}.pdf')
    plt.show()

    # %% Save the results
    # Save the vegetation volume per height class to a CSV file
    new_row = {
        'veg_volume': veg_volume,
        'volume_voxel_size': voxel_size,
        
        #'square_area': square_area,
        #'vol_to_area': vol_to_area,
        #'veg_volume_per_height': veg_volume_per_height.to_dict(orient='records'),
        #'file': os.path.basename(file_path),
        #'height_of_layers': height_of_layers,
        'Square': square,
    }
    result_df.loc[len(result_df)] = new_row
    result_df.to_csv(result_csv_path, index=False)
    print(f"Results saved to {result_csv_path}")

##################################################################################
if __name__ == "__main__":
    square = 'alpenplatz'  # Replace with your square identifier
    square_list = [ "alpenplatz", "nikolai", "rundfunk", "konigsplatz", "miesbacher", "jakobgelb"]  # Replace with your square identifier#
    square_list = ["elisabeth"]
    for square in square_list:
        main_path = f'/Volumes/T7_Shield_A/msc'
        volumeVegetation(square, main_path)