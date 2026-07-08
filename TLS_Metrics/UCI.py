#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates the UCI (Understorey Complexity Index) from the processed TLS point cloud data.
UCI is calculated based fractal dimension of points in the understorey layer (between 0.8 m and 1.8 m height) based on the method described in Willim et al. (2019).

Source:
- Willim, K., et al. (2019). “Assessing Understory Complexity in Beech-dominated Forests (Fagus sylvatica L.) in Central Europe—From Managed to Primary Forests”. In.: Sensors 19.7, p. ​​1684​. DOI: 10.3390/s19071684.
- Seidel, D., et al. (2021). “Quantifying Understory Complexity in Unmanaged Forests Using TLS and Identifying Some of Its Major Drivers”. In: Remote Sensing 13.8, p. 1513. DOI: 10.3390/rs13081513.


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT) adapted from Method description (Willim et al., 2019, Seidel et al., 2021) with modifications to fit into the workflow of this project (error might be introduced due to modifications)

Originally created Spring 2026, last updated June 2026
"""
#script for UCI calculation (based on: Assessing Understory Complexity in Beech-dominated Forests (Fagus sylvatica L.) 
# in Central Europe—From Managed to Primary Forests -  Katharina Willim et al. 2019)

# import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar, assign_pointdata_with_cellkey,key_to_cells
import seaborn as sns
from shapely.geometry import Polygon
#from shapely.ops import unary_union
import open3d as o3d
from definitions_a import assign_pointdata_with_cellkey,key_to_cells


################ 
# Help functions
################
def cart2sph(x, y, z):
    """Convert Cartesian coordinates to Spherical coordinates."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)      # inclination
    phi = np.arctan2(y, x)        # azimuth
    return theta, phi, r


#################################
# Main UCI calculation
##################################
def uci_calc(main_path: str, square: str):
    """Calculate the Understory Complexity Index (UCI) for a given square.
    UCI is calculated based on the fractal dimension of the points in the understorey layer (between 0.8 m and 1.8 m height) following the method described in Willim et al. (2019) and Seidel et al. (2021).

    Function saves results to csv file and saves and viaualizes the polygon of the understorey vegetation points used for the UCI calculation.
    
    Args:
        main_path (str): The path to the main directory containing the data and the folders to the final processed point cloud data.
        square (str): The identifier for the square area.
    Returns:
        None: The function saves the calculated VAI values to the general results CSV file and does not return any value.
    """
    print("Starting UCI calculation...")
    print(f"Processing square: {square}")
    print(f"Main path: {main_path}")


    # Load the point cloud data with the assign_pointdata_with_cellkey function (defined in definitions_a.py)
    print("Loading point cloud data...")
    laz_path = f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'
    memmap_path = f'{main_path}/{square}_FINAL/{square}_pointcelldata.memmap'
    # Load the point cloud data after preprocessing via definitions_a assign_pointdata_with_cellkey
    

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
    
    #### Filter vegetation points only of understorey height in within the two given tresholds
    print("Filtering vegetation points within height thresholds...")
    bottom_height_tresh = 0.8  # meters
    top_height_tresh = 1.8    # meters
    pd_veg = pd_points[(pd_points['Classification'].isin([2, 3])) & 
                       (pd_points['Z'] >= bottom_height_tresh) & 
                       (pd_points['Z'] <= top_height_tresh)].copy()
    print(f"Total vegetation points within threshold: {len(pd_veg)}")

    ##### Convert to polar coordinates (from scanner origin) for Sorting points # see helper function cart2sph
    pd_veg["Z"] = 0  # set Z to zero for 2D polar conversion
    print("Converting to polar coordinates...")
    theta, phi, r = cart2sph(pd_veg['X'].values, pd_veg['Y'].values, pd_veg['Z'].values)
    pd_veg['theta'] = theta
    pd_veg['phi'] = phi
    pd_veg['r'] = r
    pd_veg['theta_deg'] = np.degrees(theta) # polar angle in degrees
    pd_veg['phi_deg'] = np.degrees(phi) # azimut angle in degrees
    print(f"Columns of pd_veg: {pd_veg.columns.tolist()}")   
    
    # sort azimut angles
    pd_veg = pd_veg.sort_values(by=['phi_deg', 'r']).reset_index(drop=True)
    
    ###### Reconvert to cartesian for area and perimeter calculation
    points_2D = pd_veg[['X', 'Y']].values
    # create polygon
    polygon_veg = Polygon(points_2D) 
    # calculate area, perimeter and fractal dimension
    area_veg = polygon_veg.area
    perimeter_veg = polygon_veg.length
    fractal_dimension_veg = (2 * (np.log(0.25 * perimeter_veg))) / np.log(area_veg) if area_veg > 0 else np.nan
    print(f"Vegetation area: {area_veg}, Perimeter: {perimeter_veg}, Fractal Dimension: {fractal_dimension_veg}")
    
    ##### Visualize the polygon and the points in the understorey layer
    plt.figure(figsize=(8, 8))
    x, y = polygon_veg.exterior.xy
    plt.plot(x, y, color='blue', alpha=0.7, linewidth=2, solid_capstyle='round', zorder=2)
    plt.scatter(pd_veg['X'], pd_veg['Y'], color='green', s=1, zorder=1)
    plt.title(f'Vegetation Polygon for {square} (Height: {bottom_height_tresh}-{top_height_tresh} m)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(f'{main_path}/{square}_FINAL/{square}_UCI_Vegetation_Polygon.png', dpi=300)
    plt.show() 
    #plt.close()
    
    
    
    ##### Save results
    result_dict = {
        'Square': square,
        f'UCI_area_veg_{bottom_height_tresh}_{top_height_tresh}': area_veg,
        f'UCI_Perimeter_veg_{bottom_height_tresh}_{top_height_tresh}': perimeter_veg,
        f'UCI_Fractal Dimension_veg_{bottom_height_tresh}_{top_height_tresh}_phiR': fractal_dimension_veg}
        
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


    print(f"UCI calculation completed for square: {square}")



##########################################
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'  # Replace with your main path /Volumes/T7_Shield_A/msc/alpenplatz_FINAL
    
    square_list = [ "alpenplatz", "nikolai", "rundfunk", "konigsplatz", "miesbacher", "jakobgelb", "elisabeth"]

    for square in square_list:
      uci_calc(main_path, square)
