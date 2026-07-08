#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates the Stand Structural Complexity Index (SSCI) and Effective Number of Layers (2D-ENL) from the processed TLS point cloud data, 
following the method described in Ehbrecht et al. (2016) and Ehbrecht et al. (2017).

2d-ENL: 2nd Hill number (Inverse Simpson Index) as a measure of diversity of layers. 


The script includes the following steps:
1. Load the processed point cloud data (after preprocessing and manual cleaning) for each square.
2. Filter to vegetation points only (Classification 2 and 3).
3. Calculate ENL (Effective Number of Layers) using the method of Ehbrecht et al. (2016) using voxelization (20 cm) and stratification into vertical layers.
4. Convert the vegetation points to spherical coordinates (from scanner origin) and simulate beam directions. (Since different Scanner than in original study, we use the angles to simulate beam directions)
5. Construct cross-sectional polygons based on opposite beam directions and calculate their area, perimeter, and fractal dimension.
6. Calculate SSCI (Stand Structural Complexity Index) using the mean fractal dimension and ENL.
7. Save the results to a CSV file and print the results.

Source:
- Ehbrecht, M., et al. (2016). “Effective number of layers: A new measure for quantifying three-dimensional stand structure based on sampling with terrestrial LiDAR”. In: Forest Ecology and Management 380, pp. 212–223. DOI: 10.1016/j.foreco.2016.09.003.
- Ehbrecht, M., et al. (2017). “Quantifying stand structural complexity and its relationship with forest management, tree species diversity and microclimate”. In: Agricultural and Forest Meteorology 242, pp. 1–9. DOI: 10.1016/j. agrformet.2017.04.012.


Author: Antonia Hostlowsky (assisted by ChatGPT and CoPilot) adapted from original code by Ehbrecht et al. (2017) uploaded in github repository of the project https://github.com/ehbrechtetal/Stand-structural-complexity-index---SSCI (June 2026)

Originally created Spring 2026, last updated June 2026
"""

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
# Convert to polar coordinates (from scanner origin)
def cart2sph_R_enl(x, y, z):
    """
    Convert Cartesian to Spherical coordinates.
    Matches R's pracma:: cart2sph convention.
    
    Args:
        x, y, z:  Cartesian coordinates (numpy arrays or scalars)
    
    Returns:
        theta: azimuthal angle in radians [-π, π] (angle in XY plane from X-axis)
        phi: elevation angle in radians [-π/2, π/2] (angle from XY plane)
        r: radial distance
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)                    # azimuth (horizontal angle)
    phi = np.arctan2(z, np.sqrt(x**2 + y**2))   # elevation (vertical angle)
    
    return theta, phi, r

def sph2cart_R_enl(theta, phi, r):
    """
    Convert Spherical to Cartesian coordinates.
    
    Args:
        theta:  azimuthal angle in radians
        phi: elevation angle in radians
        r: radial distance
    
    Returns:
        x, y, z: Cartesian coordinates
    """
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z

def calculate_fractal_dimension(perimeter, area):
    """
    Calculate fractal dimension using the perimeter-area relationship.
    Formula: D = 2 * log(0.25 * P) / log(A)
    
    Args:
        perimeter: polygon perimeter
        area: polygon area
    
    Returns:
        fractal dimension
    """
    if area <= 0 or perimeter <= 0:
        return np.nan
    frac = 2 * np.log(0.25 * perimeter) / np.log(area)
    return frac

def compute_polygon_metrics(points_2d):
    """
    Compute area, perimeter, and fractal dimension for a 2D polygon.
    
    Args:
        points_2d: array of shape (n, 2) with x, z coordinates
    
    Returns: 
        dict with area, perimeter, and fractal dimension
    """
    if len(points_2d) < 3:
        return {'area': 0, 'perimeter': 0, 'fractal_dim': np.nan}
    
    try:
        # Create polygon (using convex hull or all points)
        # Option 1: Use all points to create polygon
        poly = Polygon(points_2d) 
        
        # Option 2: Use convex hull (simpler, faster)
        # hull = ConvexHull(points_2d)
        # poly = Polygon(points_2d[hull.vertices])
        
        if not poly.is_valid:
            poly = poly.buffer(0)  # Fix invalid polygons
        
        area = poly.area
        perimeter = poly.length
        fractal_dim = calculate_fractal_dimension(perimeter, area)
        
        # debbug figure
        # import matplotlib.pyplot as plt
        # x, y = poly.exterior.xy
        # plt.figure()
        # plt.plot(x, y)
        # plt.scatter(points_2d[:,0], points_2d[:,1], color='red', s=1)
        # plt.title(f"Area: {area:.2f}, Perimeter: {perimeter:.2f}, Fractal Dim: {fractal_dim:.2f}")
        # plt.show() 
        
        
        return {
            'area': round(area, 1),
            'perimeter': round(perimeter, 1),
            'fractal_dim': fractal_dim
        }
    except:
        return {'area':  0, 'perimeter':  0, 'fractal_dim': np.nan}

def calculate_enl(pointcloud, cell_radius=20, voxel_size=0.2, layer_height=1.0):
    """
    Calculate Effective Number of Layers (ENL) - Ehbrecht et al. 2016.
    With the 2nd Hill number (Inverse Simpson Index) as a measure of diversity of layers. 
    
    Args:
        pointcloud: DataFrame with X, Y, Z columns
        cell_radius: radius of circular plot (default 20m)
        voxel_size: voxel size in meters (default 0.2m = 1/5)
        layer_height: height of vertical layers in meters (default 1m)
    
    Returns:
        dict with ENL, top_height, and number of layers
    """
    # Filter to circular plot of radius 20m
    #no filter for now
    pc_subset = pointcloud.copy()
    
    if len(pc_subset) == 0:
        return {'enl': np.nan, 'top_height': 0, 'n_layers': 0}
    
    # Voxelization:  round coordinates to voxel_size grid
    # R code: round(x*5)/5 means voxel_size = 0.2 m
    pc_subset['x_voxel'] = np.round(pc_subset['X'] / voxel_size) * voxel_size
    pc_subset['y_voxel'] = np.round(pc_subset['Y'] / voxel_size) * voxel_size
    pc_subset['z_voxel'] = np.round(pc_subset['Z'] / voxel_size) * voxel_size + 0.5
    
    # Keep only unique voxels
    voxels = pc_subset[['x_voxel', 'y_voxel', 'z_voxel']].drop_duplicates()
    
    # Stratify into 1m vertical layers
    voxels['z_layer'] = np.round(voxels['z_voxel'] / layer_height) * layer_height
    
    # Count voxels per layer
    layer_counts = voxels['z_layer'].value_counts()
    sum_voxels = layer_counts.sum()
    
    # Calculate ENL using Inverse Simpson Index
    proportions = layer_counts / sum_voxels
    enl = 1 / np.sum(proportions ** 2)
    
    top_height = voxels['z_layer'].max()
    n_layers = len(layer_counts)
    
    #free memory
    del pc_subset
    del voxels
    del layer_counts
    del proportions
    
    return {
        'enl': enl,
        'top_height': top_height,
        'n_layers': n_layers
    }



#################################
# Main Function 
##################################
def enl_ssci(main_path, square):
    """
    Calculate Stand Structural Complexity Index (SSCI) and Effective Number of Layers (ENL)
    following Ehbrecht et al. (2017).
    
    Args:
        main_path (str): The path to the main directory containing the data and the folders to the final processed point cloud data.
        square (str): The identifier for the square area.
    Returns:
        None: The function saves the calculated VAI values to the general results CSV file and does not return any value.

    """
    print("Starting ENL and SSCI calculation...")
    print(f"Processing square: {square}")
    print(f"Main path: {main_path}")
    ################################################################################################
    ############ Load the point cloud data 
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
    
    # Create DataFrame from memmap
    arr = result["per_point"] 
    pd_points = pd.DataFrame({
        "cell_key": arr["cell_key"],
        "X": arr["x"].astype(np.float64),
        "Y": arr["y"].astype(np.float64),
        "Z": arr["z"].astype(np.float64),
        "Classification": arr["classification"].astype(np.int32)
    })
    print(f"Total points loaded: {len(pd_points)}")
    
    ############ Remove points below 0 m
    pd_points = pd_points[pd_points['Z'] > 0]
    print(f"Points after Z>0 filter: {len(pd_points)}")
    
    ############ Filter vegetation points only (Classification 2 and 3)
    pd_veg = pd_points[pd_points['Classification'].isin([2, 3])]
    print(f"Total vegetation points: {len(pd_veg)}")
    ############################################################################################################
    ############ Calculate ENL (Effective Number of Layers)
    print("\nCalculating Effective Number of Layers (ENL)...")
    enl_results = calculate_enl(pd_veg, voxel_size=0.2, layer_height=1.0)
    
    print(f"  ENL:  {enl_results['enl']:.3f}")
    print(f"  Top height: {enl_results['top_height']:.1f} m")
    print(f"  Number of layers: {enl_results['n_layers']}")
    ############################################################################################################
    ############ Calculation of SSCI (Stand Structural Complexity Index)
    ######## Convert to spherical coordinates (from scanner origin at 0,0)     ## Convert to spherical coordinates (equivalent to R: pracma::cart2sph)
    print("Converting to spherical coordinates...")
    theta, phi, r = cart2sph_R_enl(
        pd_veg['X'].values, 
        pd_veg['Y'].values, 
        pd_veg['Z'].values
    )
    
    # Store spherical coordinates
    pd_veg['theta'] = theta          # Azimuthal angle (radians) - horizontal
    pd_veg['phi'] = phi              # Elevation angle (radians) - vertical
    pd_veg['r'] = r                  # Radial distance
    
    # Convert to degrees (equivalent to R: *180/pi)
    pd_veg['theta_deg'] = np.degrees(theta)  
    pd_veg['phi_deg'] = np.degrees(phi)
    
    # Use calculated angles directly (since no beam_direction data)
    pd_veg['angle_vertical'] = pd_veg['phi_deg']
    pd_veg['angle_horizontal'] = (pd_veg['theta_deg'] + 360) % 360  # Convert to 0-360° #the % 360 ensures angles are within 0-360 range 
    
    ######## Simulate beam directions (since you don't have them)
    # The R code uses beam directions with step 0.03515625°
    # Total horizontal beams: 360 / 0.03515625 = 10236
    # Total vertical beams:  300 / 0.03515625 = 8533
    
    # We'll discretize your angles to match this resolution
    angular_step = 0.03515625
    pd_veg['beam_direction_horizontal'] = np.round(pd_veg['angle_horizontal'] / angular_step).astype(int)
    pd_veg['beam_direction_vertical'] = np.round(pd_veg['angle_vertical'] / angular_step).astype(int)
    
    print(f"Horizontal beam range: {pd_veg['beam_direction_horizontal'].min()} to {pd_veg['beam_direction_horizontal'].max()}")
    print(f"Vertical beam range: {pd_veg['beam_direction_vertical'].min()} to {pd_veg['beam_direction_vertical'].max()}")
    
    ######## Construct cross-sectional polygons
    print("\nConstructing cross-sectional polygons...")
    
    # R code works with every 4th beam (16th of original resolution)
    # Create pairs of opposite horizontal beams (180° apart)
    # horizontal_1: 0 to 5116 (step 4) = 0° to ~180°
    # horizontal_2: 5120 to 10236 (step 4) = ~180° to 360°
    
    horizontal_1 = np.arange(0, 5117, 4)
    horizontal_2 = np.arange(5120, 10237, 4)
    opposite_pairs = pd.DataFrame({
        'horizontal_1': horizontal_1, #left beam directions
        'horizontal_2': horizontal_2 #right side beam directions
    })
    
    print(f"Number of cross-sectional polygons: {len(opposite_pairs)}")
    
    # Filter to valid beam directions
    pd_veg = pd_veg[pd_veg['beam_direction_horizontal'] <= 10236]
    
    # Calculate modified angles for 2D projection
    print("Calculating modified angles for 2D projection...")
    pd_veg['angle_vertical_mod'] = np.where(
        pd_veg['angle_horizontal'] >= 180,
        180 - pd_veg['angle_vertical'],
        pd_veg['angle_vertical']
    ) #results in angles between 0 and 180 degrees
    pd_veg['angle_horizontal_mod'] = 180 - pd_veg['angle_horizontal'] # results in angles between -180 and 180 degrees? 
    
    print("after angle modifications:")
    print(pd_veg[['angle_vertical_mod', 'angle_horizontal_mod']].head())
    
    # Sort by beam direction
    print("Sorting vegetation points by beam directions...")
    pd_veg = pd_veg.sort_values(['beam_direction_horizontal', 'angle_vertical_mod'])
    
    # Convert modified angles to 2D coordinates using spherical to Cartesian conversion
    print("Converting to 2D Cartesian coordinates...")
    theta_2d = np.where(
        pd_veg['angle_vertical_mod'] > 90,
        np.radians(180),
        np.radians(0)
    )
    phi_2d = np.radians(pd_veg['angle_vertical'])
    
    x2d, y2d, z2d = sph2cart_R_enl(theta_2d, phi_2d, pd_veg['r'].values)
    
    pd_veg['x2D'] = x2d
    pd_veg['y2D'] = y2d
    pd_veg['z2D'] = z2d
    
    print("2D coordinates calculated:" )    
    print(pd_veg[['x2D', 'y2D', 'z2D']].head())
    
    ## Calculate polygon metrics for each cross-section
    print("\nCalculating polygon metrics...")
    
    results = []
    
    for i, row in opposite_pairs.iterrows():
        # Get points for this pair of opposite beam directions
        mask = pd_veg['beam_direction_horizontal'].isin([row['horizontal_1'], row['horizontal_2']])
        subset = pd_veg[mask]
        
        if len(subset) < 3:
            continue
        
        # Use x2D and z2D for 2D polygon
        points_2d = subset[['x2D', 'z2D']].values
        
        # Calculate metrics
        metrics = compute_polygon_metrics(points_2d) 
        
        # Store results with mean values
        result_dict = {
            'polygon_id': i,
            'angle_horizontal_mean': subset['angle_horizontal'].mean(),
            'angle_vertical_mean':  subset['angle_vertical'].mean(),
            'n_points': len(subset),
            'area': metrics['area'],
            'perimeter': metrics['perimeter'],
            'fractal_dim': metrics['fractal_dim']
        }
        
        results.append(result_dict)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(opposite_pairs)} polygons...")
    
    df_polygons = pd.DataFrame(results)
    print(f"\nTotal valid polygons: {len(df_polygons)}")
    print(f"Mean fractal dimension: {df_polygons['fractal_dim'].mean():.3f}")
    
   
    
    ## Calculate SSCI (Stand Structural Complexity Index)
    mean_fractal = df_polygons['fractal_dim'].mean()
    ssci = mean_fractal ** np.log(enl_results['enl'])
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"{'='*60}")
    print(f"Mean Fractal Dimension: {mean_fractal:.4f}")
    print(f"2D-ENL: {enl_results['enl']:.4f}")
    print(f"SSCI: {ssci:.4f}")
    print(f"{'='*60}")
    
    ## Prepare output
    output = {
        'Square': square,
        'area_mean': df_polygons['area'].mean(),
        'perimeter_mean': df_polygons['perimeter'].mean(),
        'fractal_dim':  mean_fractal,
        'top_height': enl_results['top_height'],
        'n_layers': enl_results['n_layers'],
        'enl': enl_results['enl'],
        'ssci':  ssci
    }
    

    #save to General Results CSV
    #general result_csv_path
    general_result_csv_path = f'{main_path}/All_Results.csv'
    
    if os.path.exists(general_result_csv_path):
        result_df = pd.read_csv(general_result_csv_path)
        # check if the columns exist, if not create them
        for col in output.keys():
            if col not in result_df.columns:
                result_df[col] = np.nan
    else:
        result_df = pd.DataFrame(columns=output.keys())
        
    print("Updating general results CSV...")
    print(output)
    # Append new results
    result_df = pd.concat([result_df, pd.DataFrame([output])], ignore_index=True)
    result_df.to_csv(general_result_csv_path, index=False)


    print(f"ENL and SSCI calculation completed for square: {square}")


####################################################################
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'  # Replace with your main path e.g. /Volumes/T7_Shield_A/msc/alpenplatz_FINAL
    square_list = [ 
                   #"alpenplatz",
                   "nikolai", 
                   "rundfunk", 
                   "konigsplatz", 
                   "miesbacher",
                   "jakobgelb", 
                   "elisabeth"
                   ]  # Replace with your square identifier
    for square in square_list:  
       enl_ssci(main_path, square)
       
