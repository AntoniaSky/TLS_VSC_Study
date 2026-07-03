#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Dec 2025, last updated June 2026

This script normalizes TLS Point Cloud Data via CSF (Cloth Simulation Filter) and a lowest-point grid DEM approach. 
Use after Downsampling and Noise Removal (PreprocessingPointClouds.py) and before manual cleaning in CloudCompare.
1. CSF (Zhang et al., 2016, DOI: https://doi.org/10.3390/rs8060501), implementation in Python by library CSF (
2. DEM Generation: Lowest Point per Grid Cell
3. Smoothing of DEM
4. Normalization
5. Visualization of DEM and Normalized Point Cloud
6. Exception Case: Elisabethplatz with Delaunay interpolation for DEM generation (to avoid large gaps in DEM)

Sources: 
- CSF: https://doi.org/10.3390/rs8060501


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
"""


#import necessary libraries
import laspy
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import open3d as o3d
import datetime
import CSF #import CSF library 
from scipy.spatial import cKDTree #import cKDTree for fast nearest neighbor search
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator 
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter, uniform_filter

from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar

from StyleMaps import quads

##################
#Helper functions for CSF and DEM generation and normalization
##################
def run_csf(input_laz_path, output_ground_laz_path, overwrite=False):
    """Run CSF (Cloth Simulation Filter) on a LAS/LAZ file to classify ground points.
    Saves the ground points to a new LAS/LAZ file and the parameters used to a text file.

    Args:
        input_laz_path (str): Path to the input LAS/LAZ file.
        output_ground_laz_path (str): Path to the output file containing ground points.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to False.
    """
    
    if not overwrite and os.path.exists(output_ground_laz_path):
        print(f"Ground points file {output_ground_laz_path} already exists. Skipping CSF processing.")
        return
    else: 
        print(f"Running CSF on {input_laz_path}...")
        
        # Load the LAS/LAZ file
        laspy_file = laspy.read(input_laz_path)
        points = laspy_file.points # get all points from the las file with all attributes
        xyz = np.vstack((laspy_file.x, laspy_file.y, laspy_file.z)).transpose() # extract x, y, z and put into a list

        
        # CSF used to classify ground points (more details about parameter: http://ramm.bnu.edu.cn/projects/CSF/download/)
        csf = CSF.CSF()
        
        ##### CSF SETTINGS: you can try different values for the parameters to see how it affects the result
        resolution = 8 # grid size in meter #larger easier to detect ground, but less accurate
        csf.params.bSloopSmooth = True # smooth the slope 
        csf.params.cloth_resolution = resolution # grid size in meter
        csf.params.rigidness = 10  #higher value makes the cloth more rigid so that it is less likely to be bended by small objects
        csf.params.time_step = 0.8 # higher values make the cloth detect the ground faster, but too high values may cause the cloth to blow up
        csf.params.class_threshold = 0.40 # distance threshold to classify ground points
        #csf.params.iterations = 500 # number of iterations # the higher the number, the longer it takes, but the more accurate the result is

        print("parameters set for CSF:")
        print(f"bSloopSmooth: {csf.params.bSloopSmooth}")
        print(f"cloth_resolution: {csf.params.cloth_resolution}")
        print(f"rigidness: {csf.params.rigidness}")
        print(f"time_step: {csf.params.time_step}") 
        print(f"class_threshold: {csf.params.class_threshold}")
        #print(f"iterations: {csf.params.iterations}")
        
        csf.setPointCloud(xyz)
        ground = CSF.VecInt()  # a list to indicate the index of ground points after calculation
        non_ground = CSF.VecInt() # a list to indicate the index of non-ground points after calculation
        csf.do_filtering(ground, non_ground) # do actual filtering.

        # Save ground points as laz file
        outFile = laspy.LasData(laspy_file.header) # create a new las file to save the ground points based on the original las file
        print(f"Scale of las file: {laspy_file.header.scale}") # should be same as las thus 0.00025
        outFile.points = points[np.array(ground)] # extract ground points, and save it to a las file.
        outFile.write(output_ground_laz_path)
        
        #save parameters to a text file
        param_file_path = output_ground_laz_path.replace('.laz', '_CSF_parameters.txt')
        with open(param_file_path, 'w') as f:
            f.write("CSF parameters used:\n")
            f.write(f"bSloopSmooth: {csf.params.bSloopSmooth}\n")
            f.write(f"cloth_resolution: {csf.params.cloth_resolution}\n")
            f.write(f"rigidness: {csf.params.rigidness}\n")
            f.write(f"time_step: {csf.params.time_step}\n")
            f.write(f"class_threshold: {csf.params.class_threshold}\n")
            #f.write(f"iterations: {csf.params.iterations}\n")

        print(f"CSF finished. Ground points are saved to {output_ground_laz_path}.")
        
def dem_find_lowest_point_grid(pd_ground: pd.DataFrame, dem_res: float = 0.20):
    """
    Generate DEM with the lowest point per given resolution of grid of DEM 

    Args:
        pd_ground (pd.DataFrame): DataFrame containing ground points with columns X Y Z
        dem_res (float, optional): DEM resolution in meters. Defaults to 0.20 m (20 cm)

    Returns:
        Z_grid_before (np.ndarray): The DEM grid before smoothing
        X_centers (np.ndarray): X coordinates of grid cell centers
        Y_centers (np.ndarray): Y coordinates of grid cell centers
        dem_res_cm (float): DEM resolution in centimeters
        dem_res (float): DEM resolution in meters
    """
    
    # --- DEM Generation: Lowest Point per Grid Cell ---
    dem_res_cm = dem_res * 100  # convert to cm if needed
    print(f"Generating DEM with resolution {dem_res_cm} cm...")
    
    # Compute grid indices relative to min coordinates 
    x_min, y_min = pd_ground['X'].min(), pd_ground['Y'].min()
    x_idx = np.floor((pd_ground['X'] - x_min) / dem_res).astype(int)
    y_idx = np.floor((pd_ground['Y'] - y_min) / dem_res).astype(int)
    pd_ground['x_idx'] = x_idx
    pd_ground['y_idx'] = y_idx

    # Group by grid cell and take the minimum Z (lowest ground elevation)
    df_min = pd_ground.groupby(['x_idx', 'y_idx'], as_index=False)['Z'].min()

    # Build explicit grid coordinate arrays
    x_unique = np.sort(df_min['x_idx'].unique())
    y_unique = np.sort(df_min['y_idx'].unique())
    X_centers = x_min + (x_unique + 0.5) * dem_res
    Y_centers = y_min + (y_unique + 0.5) * dem_res

    # Initialize DEM grid and fill with NaN
    Z_grid = np.full((len(y_unique), len(x_unique)), np.nan)
    
    # Fill DEM grid with Z values
    for _, row in df_min.iterrows():
        xi = np.where(x_unique == row['x_idx'])[0][0]
        yi = np.where(y_unique == row['y_idx'])[0][0]
        Z_grid[yi, xi] = row['Z']
    # copy for visualization
    Z_grid_before = Z_grid.copy()
    
    return Z_grid_before, X_centers, Y_centers, dem_res_cm, dem_res
        
def smooth_dem(Z_grid, X_centers, Y_centers, smoothing_method="gaussian", smooth_strength=3, dem_png_path="", dem_res_cm=20,square="Public Square"):
    """
    Apply smoothing filter to the DEM grid to avaoid jumps due to objects occluding the ground 
    Visualize the result.
    Choose your smoothing strategy:
         - Gaussian: preserves shape, smooths gently
         - Uniform: simple mean filter (good for small noise)


    Args:
        Z_grid (np.ndarray): The DEM grid before smoothing
        X_centers (np.ndarray): X coordinates of grid cell centers
        Y_centers (np.ndarray): Y coordinates of grid cell centers
        smoothing_method (str, optional): "gaussian" or "uniform". Defaults to "gaussian".
        smooth_strength (int, optional): Strength of smoothing. Larger = smoother. Defaults to 3.
        dem_png_path (str, optional): Path to save the DEM visualization. Defaults to "".
        dem_res_cm (float, optional): DEM resolution in centimeters for visualization title. Defaults to 20.

    Returns:
        df_dem_smooth (pd.DataFrame): DataFrame with columns X_center, Y_center, Z of the smoothed DEM grid
    """
    
    # --- Apply smoothing filter to the interpolated DEM --- 
    #  - Gaussian: preserves shape, smooths gently
    #  - Uniform: simple mean filter (good for small noise)

    if smoothing_method == "gaussian":
        Z_smooth = gaussian_filter(Z_grid, sigma=smooth_strength) # sigma is the standard deviation for Gaussian kernel
    else:
        Z_smooth = uniform_filter(Z_grid, size=smooth_strength) # smooth size is the window size for uniform filter # common values are 2, 3, 5
    
    
    # --- Visualization ---
    plt.figure(figsize=(10, 8))
    plt.imshow(
        Z_smooth,
        cmap='viridis',
        extent=(X_centers.min(), X_centers.max(), Y_centers.min(), Y_centers.max()),
        origin='lower',
        aspect='equal'
    )
    plt.colorbar(label='Ground Elevation (m)')
    plt.title(f'Smoothed Lowest-Point Ground DEM ({square})')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.savefig(dem_png_path.replace('.png', f'{dem_res_cm}cm_smoothed.png'), dpi=300)
    plt.close()
    
    
    # --- Flatten smoothed grid back into DataFrame ---
    xx, yy = np.meshgrid(X_centers, Y_centers)
    df_dem_smooth = pd.DataFrame({
        'X_center': xx.ravel(),
        'Y_center': yy.ravel(),
        'Z': Z_smooth.ravel()
    }).dropna(subset=['Z'])

    print("DEM lowest-point grid head:")
    print(df_dem_smooth.head())
    
    return df_dem_smooth
    
def normalize_heights(square,input_laz_path, ground_laz_path, output_normalized_laz_path, dem_png_path, overwrite=False):
    """
    Normalize the heights of the input point cloud by subtracting the smoothed DEM derived from the CSF ground points.

    Args:
        square (str): Name of the square being processed (for logging)
        input_laz_path (str): Path to the input LAS/LAZ file to be normalized
        ground_laz_path (str): Path to the LAS/LAZ file containing CSF-classified ground points
        output_normalized_laz_path (str): Path to save the normalized LAS/LAZ file
        dem_png_path (str): Path to save the DEM visualization PNG
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to False.

    Returns:
        Saves the normalized point cloud to output_normalized_laz_path and a visualization of the DEM to dem_png_path.
    """
    
    if not overwrite and os.path.exists(output_normalized_laz_path):
        print(f"Normalized file {output_normalized_laz_path} already exists. Skipping normalization.")
        return
    else: 
        print(f"Normalizing heights for {input_laz_path}...")
        print(f"Loading ground points from {ground_laz_path}...")
        
        # --- Load the input point cloud to be normalized 
        las, pd_laz = load_laz_file(input_laz_path)

        # --- Load the CSF-derived ground points
        ground_laz, pd_ground = load_laz_file(ground_laz_path)
        print(f"Number of ground points: {len(pd_ground)}")
        
        
        # --- DEM Generation: Lowest Point per Grid Cell ---
        Z_grid_before, X_centers, Y_centers, dem_res_cm, dem_res = dem_find_lowest_point_grid(pd_ground, dem_res=0.20)
        
       
        # --- Interpolate gaps before smoothing via nearest neighbor ---
        Z_grid_filled = Z_grid_before.copy()       
        yy, xx = np.meshgrid(Y_centers, X_centers, indexing='ij')
        known = ~np.isnan(Z_grid_before)
        if np.any(~known):
            filled = griddata(
                (xx[known], yy[known]),
                Z_grid_filled[known],
                (xx[~known], yy[~known]),
                method='nearest',
                fill_value=561.500000  # nearest neighbor interpolation
            )
            Z_grid_filled[~known] = filled
        else:
            print("No missing data in DEM grid; skipping interpolation.")
            
        # --- Visualize initial DEM ---
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(Z_grid_before, cmap='terrain', origin='lower')
        plt.title('Before interpolation')

        plt.subplot(1,2,2)
        plt.imshow(Z_grid_filled, cmap='terrain', origin='lower')
        plt.title('After interpolation (nearest neighbor)')
        plt.savefig(dem_png_path.replace('.png', f'{dem_res_cm}cm_interpolated.png'), dpi=300)
        plt.close()
        
        
        # --- Apply smoothing filter to the interpolated DEM --- 
        smoothing_method = "gaussian"  # or "uniform"
        smooth_strength = 3  # larger = smoother; typical range 1–5
        df_dem_smooth = smooth_dem(Z_grid_filled, X_centers, Y_centers, smoothing_method, smooth_strength, dem_png_path, dem_res_cm, square=square)
        
        
        # --- Normalize the full LAS file (subtract interpolated ground)
        pd_las = pd_laz.copy()
        print(f"Type of pd_las: {type(pd_las)}")

        # Convert LAS coordinates safely to NumPy arrays
        X = np.asarray(pd_las['X'], dtype=float)
        Y = np.asarray(pd_las['Y'], dtype=float)
        points_xy = np.column_stack((X, Y))

        # Build KD-tree using DEM centers
        tree_points = np.column_stack((
            np.asarray(df_dem_smooth['X_center'], dtype=float),
            np.asarray(df_dem_smooth['Y_center'], dtype=float)
            ))
        tree = cKDTree(tree_points)
        # Query nearest DEM cell for each LAS point
        dist, idx = tree.query(points_xy, k=1)

        # Retrieve corresponding ground elevations
        ground_z = np.asarray(df_dem_smooth['Z'], dtype=float)[idx]
        # Compute normalized heights
        pd_las['normalized_Z'] = np.asarray(pd_las['Z'], dtype=float) - ground_z

        #exchange Z with normalized_Z
        pd_las['Z'] = pd_las['normalized_Z']
        pd_las = pd_las.drop(columns=['normalized_Z'])

                
        print(pd_las.head())

        # save the normalized las file
        save_laz_file(pd_points=pd_las, file_path=output_normalized_laz_path)
        
        print(f"SUCCESS: Normalized point cloud saved to {output_normalized_laz_path}.")
        #print 5 empty lines to separate from next output
        print("\n" * 5) 

##################
#Main processing pipeline for all squares
##################
def process_csf(square, main_path, overwrite=False):
    """
    Applies CSF-based ground classification and lowest-point grid DEM normalization to the point cloud data for a given square.

    Args:
        square (str): Name of the square being processed (for logging)
        main_path (str): Base path where the input and output files are located
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to False.

    Returns:
        Saves the normalized point cloud to output_normalized_laz_path and a visualization of the DEM to dem_png_path.
    """
    
    # --- load the data
    if os.path.exists(f"{main_path}/{square}_merged_noClip/{square}_all_points_DS5mm.laz"):
        quads = ["all_points"]
    else:
        quads = quads # from StyleMaps.py 
    print(f"Processing square: {square} with quadrants: {quads}")
    
    # --- Loop through quadrants and normalize each one separately (to avoid memory issues with large clouds)
    for quad in quads:        
        input_laz = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm.laz"
        output_ground_laz = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_Groundpoints.laz"
        output_normalized_laz = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH.laz"

        if not overwrite and os.path.exists(output_normalized_laz):
            print(f"File with relH for {quad} already exists. Skipping processing.")
            continue
        elif not os.path.exists(input_laz): 
            print(f"{quad} does not exist. Skipping processing.")
            continue
        else:
            print(f"Processing {quad}...")
            dem_png_path = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_Groundpoints_interpolatedGroundSurface_DEM.png"

            # Run CSF to extract ground points
            run_csf(input_laz, output_ground_laz, overwrite=overwrite)
            
            # Run normalization using the extracted ground points
            normalize_heights(square, input_laz, output_ground_laz, output_normalized_laz, dem_png_path, overwrite=overwrite)
            
            
            # delete variables to free memory
            del input_laz, output_ground_laz, output_normalized_laz, dem_png_path
            
    print(f"CSF PROCESSING COMPLETED for square: {square}.")
    print("\n" * 3)
                


##################
# Exception case: Elisabethplatz with Delaunay interpolation for DEM generation (to avoid large gaps in DEM)
##################
def interpolate_dem_delaunay(Z_grid, X_centers, Y_centers, max_edge_length=1.0, fill_value=561.5):
    """
    Helper Function for Cases like Elisabethplatz:
    Interpolate DEM using Delaunay triangulation with max edge length constraint.
    Similar to CloudCompare's Delaunay interpolation method.
    
    Parameters:
    -----------
    Z_grid : array with NaN for empty cells
    X_centers, Y_centers : 1D arrays of grid centers
    max_edge_length : maximum edge length in grid units (CloudCompare's "triangular max length")
    fill_value : value to use for cells beyond max edge length
    """
    
    # Create meshgrid
    yy, xx = np.meshgrid(Y_centers, X_centers, indexing='ij')
    
    # Find known points
    known = ~np.isnan(Z_grid)
    
    if not np.any(~known):
        print("No missing data in DEM grid; skipping interpolation.")
        return Z_grid
    
    print("Interpolating DEM using Delaunay triangulation with max edge length constraint...")
    # Extract known point coordinates and values
    known_points = np.column_stack([xx[known].ravel(), yy[known].ravel()])
    known_values = Z_grid[known].ravel()
    
    # Create standard LinearNDInterpolator (don't modify the Delaunay object)
    interpolator = LinearNDInterpolator(known_points, known_values, fill_value=fill_value)
    print("LinearNDInterpolator created successfully")
    
    # Create Delaunay triangulation for edge length filtering
    tri = Delaunay(known_points)
    
    # Get unknown points
    unknown = ~known
    unknown_points = np.column_stack([xx[unknown].ravel(), yy[unknown].ravel()])
    
    # Interpolate all unknown points first
    interpolated_values = interpolator(unknown_points)
    
    # Now apply max edge length constraint as a post-processing step
    # Find which simplex each unknown point belongs to
    simplex_indices = tri.find_simplex(unknown_points)
    
    # Check edge lengths for each point's simplex
    for i, (point, simplex_idx) in enumerate(zip(unknown_points, simplex_indices)):
        if simplex_idx == -1:
            # Point is outside convex hull
            interpolated_values[i] = fill_value
            continue
            
        # Get the vertices of the simplex
        simplex = tri.simplices[simplex_idx]
        vertices = known_points[simplex]
        
        # Calculate edge lengths
        edge1 = np.linalg.norm(vertices[1] - vertices[0])
        edge2 = np.linalg.norm(vertices[2] - vertices[1])
        edge3 = np.linalg.norm(vertices[0] - vertices[2])
        
        # If ANY edge exceeds max length, use fill value
        if edge1 > max_edge_length or edge2 > max_edge_length or edge3 > max_edge_length:
            interpolated_values[i] = fill_value
    
    # Fill in the grid
    Z_grid_filled = Z_grid.copy()
    Z_grid_filled[unknown] = interpolated_values
    
    valid_interp_count = np.sum(interpolated_values != fill_value)
    print(f"DEM interpolation completed. Interpolated {valid_interp_count}/{len(interpolated_values)} points within max edge length.")
    return Z_grid_filled


def normalize_heights_elisabethplatz(square, main_path, overwrite=False):
    """
    Applies normalization of heights for Cases like Elisabethplatz using Delaunay interpolation for DEM generation to avoid large gaps in the DEM due to occlusions.
    Issue at Elisabethplatz: large containers occlude ground underneath canopy, leading to large gaps in the DEM when using lowest-point grid method. 
    Delaunay interpolation with max edge length constraint can help fill these gaps more realistically. 
    Values estimated with CloudCompare's Delaunay interpolation method with max edge length of 1m and fill value of 561.55 (based on visual inspection of DEM) 

    Args:
        square (str): Name of the square being processed (for logging)
        main_path (str): Base path where the input and output files are located
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to False.

    Returns:
        Saves the normalized point cloud to output_normalized_laz_path and a visualization of the DEM to dem_png_path.
    """
    
    quads = quads # from StyleMaps.py
    
    for quad in quads:
        #################### define paths to file and ground points
        non_normalized_laz_path = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm.laz"
        ground_laz_path = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_Groundpoints.laz"
        output_normalized_laz_path = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH.laz"
        dem_png_path = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_Groundpoints_interpolatedGroundSurface_DEM.png"
        
        #check if input file exists
        if not os.path.exists(non_normalized_laz_path):
            print(f"Input file {non_normalized_laz_path} does not exist. Skipping.")
            continue #with next quad

        if not os.path.exists(ground_laz_path):
            print(f"Ground file {ground_laz_path} does not exist. Skipping.")
            continue #with next quad
        
        if not overwrite and os.path.exists(output_normalized_laz_path):
            print(f"Normalized file {output_normalized_laz_path} already exists. Skipping normalization.")
            continue #with next quad
        else: 
            print(f"Normalizing heights for {non_normalized_laz_path}...")
            print(f"Loading ground points from {ground_laz_path}...")
        
            #################### load ground laz file
            las_ground, pd_ground = load_laz_file(ground_laz_path) 
            
            #################### Build DEM step 1 find lowest point per grid cell
            Z_grid_before, X_centers, Y_centers, dem_res_cm, dem_res = dem_find_lowest_point_grid(pd_ground, dem_res=0.20)     
            
            #del pd_ground, las_ground to free memory
            del pd_ground, las_ground

            #################### interpolate like in cloud compare: interpolation method 561.500000
            Z_grid_filled = interpolate_dem_delaunay(Z_grid_before, X_centers, Y_centers, max_edge_length=1.0, fill_value=561.55)
            
            """ # use linear interpolation griddata
            yy, xx = np.meshgrid(Y_centers, X_centers, indexing='ij')
            known = ~np.isnan(Z_grid_filled)
            if np.any(~known):
                filled = griddata(
                    (xx[known], yy[known]),
                    Z_grid_filled[known],
                    (xx[~known], yy[~known]),
                    method='linear',
                    fill_value=561.5500000  # linear interpolation
                )
                Z_grid_filled[~known] = filled """
        
            # --- Visualize initial DEM ---
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.imshow(Z_grid_before, cmap='terrain', origin='lower')
            plt.title('Before interpolation')

            plt.subplot(1,2,2)
            plt.imshow(Z_grid_filled, cmap='terrain', origin='lower')
            plt.title('After interpolation (delauney (Fill value 561.55))')
            plt.savefig(dem_png_path.replace('.png', f'{dem_res_cm}cm_interpolated_delauney56155.png'), dpi=300)
            plt.close()

            print(f"Saved interpolated DEM image to {dem_png_path.replace('.png', f'{dem_res_cm}cm_interpolated_delauney56155.png')}")

            # smooth DEM
            smoothing_method = "gaussian"  # or "uniform"
            smooth_strength = 3  # larger = smoother; typical range 1–5
            df_dem_smooth = smooth_dem(Z_grid_filled, X_centers, Y_centers, smoothing_method, smooth_strength, dem_png_path, dem_res_cm, square=square)
            
            #################### normalize heights - Normalize the full LAS file (subtract interpolated ground)
            las, pd_laz = load_laz_file(non_normalized_laz_path)
            
            #print type of pd_laz
            print(f"Type of pd_laz: {type(pd_laz)}")

            # Convert LAS coordinates safely to NumPy arrays
            X = np.asarray(pd_laz['X'], dtype=float)
            Y = np.asarray(pd_laz['Y'], dtype=float)
            points_xy = np.column_stack((X, Y))

            # Build KD-tree using DEM centers
            tree_points = np.column_stack((
                np.asarray(df_dem_smooth['X_center'], dtype=float),
                np.asarray(df_dem_smooth['Y_center'], dtype=float)
                ))
            tree = cKDTree(tree_points)
            # Query nearest DEM cell for each LAS point
            dist, idx = tree.query(points_xy, k=1)

            # Retrieve corresponding ground elevations
            ground_z = np.asarray(df_dem_smooth['Z'], dtype=float)[idx]
            # Compute normalized heights
            pd_laz['normalized_Z'] = np.asarray(pd_laz['Z'], dtype=float) - ground_z

            #exchange Z with normalized_Z
            pd_laz['Z'] = pd_laz['normalized_Z']
            pd_laz = pd_laz.drop(columns=['normalized_Z'])

                    
            print(pd_laz.head())

            # save the normalized las file
            save_laz_file(pd_points=pd_laz, file_path=output_normalized_laz_path)
            
            print(f"SUCCESS: Normalized point cloud saved to {output_normalized_laz_path}.")
            #print 5 empty lines to separate from next output
            print("\n" * 5) 
            
            #free memory
            del las, pd_laz, las_ground, pd_ground, Z_grid_filled, Z_grid_before, Z_grid, X_centers, Y_centers, df_dem_smooth, pd_laz


######################################################################################################################
if __name__ == "__main__":
    square_list = ["elisabeth"]  # manually enter the square names used for the following scripts
    for square in square_list:
        if square == "elisabeth":
            main_path = f'tls_data/{square}_reg'  # manually enter the main path used for the script
            normalize_heights_elisabethplatz(square, main_path, overwrite=False)
        else:
            main_path = f'tls_data/{square}_reg'  # manually enter the main path used for the script
            process_csf(square, main_path, overwrite=False)
        
        
        