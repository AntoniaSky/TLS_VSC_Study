#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Denoising the point cloud data using Open3D's statistical outlier removal.
nb_neighbors=10, std_ratio=8.0: less aggressive parameters (tested on nikolai miesbacher and visually checked

Author: Antonia Hostlowsky (assisted by ChatGPT, CoPILOT)

Originally created Autum 2025, last updated June 2026
"""

#import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import open3d as o3d    

from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar

from StyleMaps import quads
    

##### Denoise the point cloud with Open3D's statistical outlier removal (nb_neighbors=10, std_ratio=8.0) - less aggressive parameters (tested on nikolai miesbacher and visually checked)
def denoise_point_cloud(square: str, main_path: str, overwrite: bool = False):
    """Denoise the point cloud data for a given square using Open3D's statistical outlier removal (nb_neighbors=10, std_ratio=8.0).

    Args:
        square (str): _name of the square to process_
        main_path (str): _path to the square's data directory_
        overwrite (bool, optional): _whether to overwrite existing denoised files_. Defaults to False.
    """

    print(f"square: {square}")
    print(f"main_path: {main_path}")
    print(f"overwrite: {overwrite}")

        
    for quad in quads:
        print(f"Denoising quadrant: {quad}")
        
        # Define paths
        path_before_denoised = f'{main_path}/{square}_CC/{square}_{quad}_DS5mm_origCC_VegGround.laz' # merged quadrants all points denoised
        path_after_denoised = f'{main_path}/{square}_CC/{square}_{quad}_DS5mm_origCC_DN.laz' # denoised single quadrant

        # Check if input paths exist
        if not os.path.exists(path_before_denoised):
            print(f"Original file {path_before_denoised} not found. Skipping this quadrant.")
            continue

        # Check if denoised file already exists if not do denoising
        if not overwrite and os.path.exists(path_after_denoised):
            print(f"Denoised file {path_after_denoised} already exists. Skipping denoising of {quad}.")
        else:
            print("Starting denoising process...")
            
            # Load the point cloud data
            points, pd_points = load_laz_file(path_before_denoised)

        
            print("Preparation for Denoising the point cloud...")
            
            #print max of z
            print(f"Max Z value: {pd_points['Z'].max()}")

            #denoise the point cloud
            # Filter out noise points with open3d statistical outlier removal
            # Convert into o3d object thus PC data to a numpy array #make a position tensor
            print("Converting point cloud data to numpy array for tensor prep...")
            point_set = pd_points[['X', 'Y', 'Z']].values.astype(np.float32)
            intensity = pd_points['Intensity'].values.astype(np.float32)
            number_of_returns = pd_points['NumberOfReturns'].values.astype(np.uint8)
            return_number = pd_points['ReturnNumber'].values.astype(np.uint8)
            if 'Classification' in pd_points.columns:
                classification = pd_points['Classification'].values.astype(np.uint8)
            else:
                print("Classification column not found in the point cloud data.")
                classification = None

            # Create an Open3D point cloud
            pcd = o3d.t.geometry.PointCloud()

            # Set the points of the point cloud
            print("Setting points of the point cloud...")
            pcd.point.positions = point_set  # Reshape to ensure correct dimensions
            pcd.point.intensity = intensity.reshape(-1, 1)  # Reshape to ensure correct dimensions
            pcd.point.return_number = return_number.reshape(-1, 1)  # Reshape to ensure correct dimensions
            pcd.point.number_of_returns = number_of_returns.reshape(-1, 1)  # Reshape to ensure correct dimensions
            if classification is not None:
                pcd.point.classification = classification.reshape(-1, 1)
                
            # Apply statistical outlier removal
            print("Applying statistical outlier removal...")
            # pcd_denoised, ind = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0) # original parameters but too aggressive in removing tree canopy points
            pcd_denoised, ind = pcd.remove_statistical_outliers(nb_neighbors=10, std_ratio=8.0) # less aggressive parameters (tested on nikolai miesbacher and visually checked)
            

            #Convert the denoised point cloud tensors back to a numpy arrays
            denoised_points = pcd_denoised.point.positions.numpy()
            denoised_intensity = pcd_denoised.point.intensity.numpy().flatten()
            denoised_return_number = pcd_denoised.point.return_number.numpy().flatten()
            denoised_number_of_returns = pcd_denoised.point.number_of_returns.numpy().flatten()
            if classification is not None:
                denoised_classification = pcd_denoised.point.classification.numpy().flatten()   

            # Create a DataFrame from the denoised point cloud data
            print("Creating DataFrame from denoised point cloud data...")
            if classification is not None:
                print("Including classification in the denoised DataFrame.")
                pd_points_denoised = pd.DataFrame({
                    'X': denoised_points[:, 0],
                    'Y': denoised_points[:, 1],
                    'Z': denoised_points[:, 2],
                    'Intensity': denoised_intensity,
                    'ReturnNumber': denoised_return_number,
                    'NumberOfReturns': denoised_number_of_returns,
                    'Classification': denoised_classification
                })
            else:
                pd_points_denoised = pd.DataFrame({
                'X': denoised_points[:, 0],
                'Y': denoised_points[:, 1],
                'Z': denoised_points[:, 2],
                'Intensity': denoised_intensity,
                'ReturnNumber': denoised_return_number,
                'NumberOfReturns': denoised_number_of_returns,
                'Classification': denoised_classification
            })

            #print max of z after denoising
            print(f"Max Z value after denoising: {pd_points_denoised['Z'].max()}")
            print(f"Number of points before vs after denoising: {pd_points.shape[0]} vs {pd_points_denoised.shape[0]}")

            #save the denoised point cloud to a new file
            save_laz_file(path_after_denoised, pd_points_denoised)

    print("Denoising process completed for all quadrants.")
# %%
if __name__ == "__main__":
    square = 'nikolai'
    main_path = f'tls_data/{square}_reg'
    overwrite = False  # Set to True to overwrite the existing files
    
    # specify the device and dtype for Open3D
    device = o3d.core.Device('CPU:0') #meaning to use the CPU for processing # 0 stands for the first CPU
    dtype = o3d.core.float32
    denoise_point_cloud(square=square, main_path=main_path, overwrite=overwrite)
    
    
