#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Autum 2025, last updated June 2026

This script downsamples Point Clouds with Open3D library. It includes two functions:
- Option 1: downsample_into_quadrants() - takes one point cloud per site, separates it into quadrants, downsamples each quadrant and saves the results (for smaller clouds). 
- Option 2: downsample_from_quadrants() - takes one point cloud per quadrant, downsamples it and saves the result (for larger clouds).

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
"""

#import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import open3d as o3d    
import datetime

from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar, downsample_pointcloud

from StyleMaps import quads #quad_list


def downsample_into_quadrants(square, main_path, voxel_size: float, overwrite=False, device=o3d.core.Device('CPU:0'), dtype=o3d.core.float32):
    """
    Input: One Point Cloud per Site
    
    Per point cloud downsampling into downsampled clouds of quadrant (part) with given voxel size and saving the resulst
    It uses Open3D for point cloud processing.
    
    Parameters:
    - square: Name of the square (str)
    - main_path: Main path where the orignal point cloud data is stored (str)
    - overwrite: Whether to overwrite existing files (bool)
    - voxel_size: Voxel size for downsampling (float, in meters)
    - device: Open3D device to use (default is CPU)
    - dtype: Data type for Open3D tensors (default is float32)
    
    Returns:
    - None (saves downsampled point clouds to files)
    """
    
    print("#" * 20)
    print("DOWNSAMPLING INTO QUADRANTS")
    print("#" * 20)

   
    #time stamp
    time_start = datetime.datetime.now()
    print(f"Script started at: {time_start}")

    # Check voxel_size
    if voxel_size is None or not isinstance(voxel_size, (int, float)):
        raise ValueError("voxel_size must be provided as a float or int")

    # Convert voxel_size to mm safely for file naming
    voxel_size_mm = int(float(voxel_size) * 1000)

    print(f"Processing square: {square}")
    print(f"voxel_size_mm: {voxel_size_mm} mm")

    ######## Load the merged point cloud data
    # check if variable all_points is defined
    if 'all_points' not in locals():
        # load the merged point cloud data
        file_path = f'{main_path}/{square}_merged_noClip/{square}_all_points.laz'
        all_points, pd_all_points = load_laz_file(file_path) 
        print(f"finishing loading at: {datetime.datetime.now()} ")
    else:
        print("Variable all_points already defined. Skipping loading step for Time efficiency.")

    ######### separate the point cloud into the four quadrants for reduced data amount    
    quadrants = {
        'quad1': pd_all_points[(pd_all_points['X'] >= 0) & (pd_all_points['Y'] >= 0)],
        'quad2': pd_all_points[(pd_all_points['X'] < 0) & (pd_all_points['Y'] >= 0)],
        'quad3': pd_all_points[(pd_all_points['X'] < 0) & (pd_all_points['Y'] < 0)],
        'quad4': pd_all_points[(pd_all_points['X'] >= 0) & (pd_all_points['Y'] < 0)],
    }

    ####### Downsample each quadrant and save the results
    for name, quad_df in quadrants.items():
        print(f"Number of points in {name}: {len(quad_df)}")

        # Convert to numpy arrays for the tensors
        point_set = quad_df[['X', 'Y', 'Z']].values.astype(np.float32)
        intensity = quad_df['Intensity'].values.astype(np.float32)
        number_of_returns = quad_df['NumberOfReturns'].values.astype(np.uint8)
        return_number = quad_df['ReturnNumber'].values.astype(np.uint8)

        if 'Classification' in quad_df.columns:
            classification = quad_df['Classification'].values.astype(np.uint8)
        else:
            print("Classification column not found in the point cloud data.")
            classification = None


        # Create an Open3D point cloud
        pcd = o3d.t.geometry.PointCloud()

        # Set the points of the point cloud
        print("Setting points of the point cloud...")
        pcd.point.positions = point_set
        pcd.point.intensity = intensity.reshape(-1, 1)  # Reshape to ensure correct dimensions
        pcd.point.return_number = return_number.reshape(-1, 1)  # Reshape to ensure correct dimensions
        pcd.point.number_of_returns = number_of_returns.reshape(-1, 1)  # Reshape to ensure correct dimensions
        if classification is not None:
            pcd.point.classification = classification.reshape(-1, 1)  # Reshape to ensure correct dimensions    

        # Print the point cloud to verify
        print(pcd, "\n")

        # Downsample the point cloud using voxel downsampling
        print(f"Starting to downsample the point cloud at {datetime.datetime.now()} ... (voxel size: {voxel_size_mm} mm)")
        downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Ending to downsample the point cloud at {datetime.datetime.now()} ...")

        # Debugging: Visualize the downsampled point cloud
        #print("starting to visualize the downsampled point cloud...")
        #o3d.visualization.draw_geometries([downpcd.to_legacy()])



        ####### Save the downsampled point cloud to a new file
        #get the attributes of the downsampled point cloud from open3d objet
        print("Saving the downsampled point cloud to a new file...")
        positions = downpcd.point.positions
        intensity = downpcd.point.intensity
        return_number = downpcd.point.return_number
        number_of_returns = downpcd.point.number_of_returns

        # turn tensors into numpy arrays for saving
        positions = positions.numpy()
        intensity = intensity.numpy().flatten()  # Flatten to ensure correct shape
        return_number = return_number.numpy().flatten()  # Flatten to ensure correct shape
        number_of_returns = number_of_returns.numpy().flatten()  # Flatten to ensure correct shape

        # Create a pandas DataFrame from the downsampled point cloud
        downsampled_points = pd.DataFrame({ 
            'X': positions[:, 0],
            'Y': positions[:, 1],
            'Z': positions[:, 2],
            'Intensity': intensity,
            'ReturnNumber': return_number,
            'NumberOfReturns': number_of_returns
        })

        print(downsampled_points.head())  # Print the first few rows of the downsampled point cloud

        path = f'{main_path}/{square}_merged_noClip/{square}_{name}_DS{voxel_size_mm}mm.laz'
        save_laz_file(path, downsampled_points)

        print(f"Downsampled point cloud saved to {path}")
        print(f"Number of points after downsampling in {name}: {len(downsampled_points)}")
        print(f"Finished processing {name} at {datetime.datetime.now()} \n")

    # time stamp 
    time_end = datetime.datetime.now()
    print(f"Script ended at: {time_end}")
    print(f"Total processing time for {square}: {time_end - time_start}")
    
    
    
#################
# Define the downsample function for when points in quadrant point clouds
def downsample_from_quadrants(square, main_path, overwrite=False, device=o3d.core.Device('CPU:0'), dtype=o3d.core.float32, voxel_size=0.005):
    """
    Downsample merged point cloud data from Quadrants and save the result (Input One Point Cloud per Quadrant).
    Used Open3D for point cloud processing.

    Parameters:
        square: str, name of the square (e.g., 'jakobgelb')
        main_path: str, path to the main directory containing point cloud data
        voxel_size: float, size of the voxel for downsampling (default is 0.001 m)
        overwrite: bool, whether to overwrite existing files (default is False)
    
    Returns:
    None
    """
    
    #print start of new function with 20*#
    print("#" * 10)
    print("DOWNSAMPLING FROM QUADRANTS for square: ", square)


    #time stamp
    time_start = datetime.datetime.now()
    print(f"Script started at: {time_start}")

    
    #voxel size in mm without decimal 
    voxel_size_mm = int(voxel_size * 1000)

    #print parameters
    print(f"overwrite: {overwrite}")
    print(f"voxel_size_mm: {voxel_size_mm} mm")
    print(f"processing square: {square}")

    ###### Load the merged point cloud data per quadrant
    for quad in quads: # stylemaps.py
        print(f"Processing {quad}...")
        
        all_points_quad_path = f'{main_path}/{square}_merged_noClip/{square}_all_points_{quad}.laz'
        save_path = f'{main_path}/{square}_merged_noClip/{square}_{quad}_DS{voxel_size_mm}mm.laz'

        if not os.path.exists(save_path) or overwrite:
            if not os.path.exists(all_points_quad_path):
                print(f"the quad {quad} does not exist. Skipping {quad}.")
                continue  # Skip to the next quadrant if the file does not exist
            else:
                print(f"Loading merged point cloud data for {quad}...")
                all_points_quad, pd_all_points_quad = load_laz_file(all_points_quad_path) 

                numberOfPoints = len(pd_all_points_quad)
                print(f"Number of points in {quad}: {numberOfPoints}")
                
                if numberOfPoints > 150_000_000:
                    print(f"Warning: The point cloud for {quad} has more than 150 million points. This may lead to high memory usage and long processing times.")
                    print("Thus splitting the point cloud into two halves for downsampling.")
                    median_x = pd_all_points_quad['X'].median()
                    pd_first_half = pd_all_points_quad[pd_all_points_quad['X'] <= median_x]
                    pd_second_half = pd_all_points_quad[pd_all_points_quad['X'] > median_x]
                    downsampled_first_half = downsample_pointcloud(pd_first_half, voxel_size, save=False, output_path=None)
                    downsampled_second_half = downsample_pointcloud(pd_second_half, voxel_size, save=False, output_path=None)
                    downsampled_pd_points = pd.concat([downsampled_first_half, downsampled_second_half], ignore_index=True)
                    print(f"Number of points after downsampling in {quad}: {len(downsampled_pd_points)}")
                    # save the downsampled point cloud
                    save_laz_file(save_path, downsampled_pd_points)
                    print(f"Downsampled point cloud saved to {save_path}")
                else:
                    #downsample the merged point cloud data
                    print(f"Downsampling the merged point cloud data for {quad} with voxel size {voxel_size_mm} mm...")
                    downsampled_pd_points = downsample_pointcloud(pd_all_points_quad, voxel_size, save=True, output_path=save_path)
                    print(f"Number of points after downsampling in {quad}: {len(downsampled_pd_points)}")
                    print(f"Downsampled point cloud saved to {save_path}")
                    print("\n"*2)
        else:
            print(f"Downsampled point cloud file for {quad} already exists. Skipping loading step.")
            continue  # Skip to the next quadrant if the file already exists

    # time stamp
    time_end = datetime.datetime.now()
    print(f"Script ended at: {time_end}")
    print(f"Total duration: {time_end - time_start}")
    
    print(f"DOWNSAMPLING COMPLETED for square: {square}.")
    print("\n" * 3)
    
    
# If this script is run as the main module, execute the function with example parameters
if __name__ == "__main__":
    #for small clouds
    #downsample_into_quadrants(square='jakobgelb', main_path='data/jakobgelb/jakobgelb_msa', 
       #                       overwrite=False, device=o3d.core.Device('CPU:0'), dtype=o3d.core.float32, voxel_size=0.001)
    #for large clouds
    downsample_from_quadrants(square='jakobgelb', main_path='tls_data/jakobgelb_reg',
                              overwrite=False, device=o3d.core.Device('CPU:0'), dtype=o3d.core.float32, 
                              voxel_size=0.005)