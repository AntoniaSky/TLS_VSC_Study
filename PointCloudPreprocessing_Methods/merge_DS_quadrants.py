#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Autum 2025, last updated June 2026

This script merges multiple (downsampled) point cloud files and saves the result.
(Applicable only for smaller Sites (e.g., Nikolaiplatz))

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
"""

#import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import open3d as o3d    

from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar

from StyleMaps import quads

    
def merge_DS_quadrants(square, main_path, voxelsize, overwrite=False):
    """
    Merges downsampled point cloud files into one point cloud.

    Parameters:
        square (str): name of Square
        main_path (str): path to the main directory
        voxelsize (float): voxel size from downsampling (for naming the files)
        overwrite (bool, optional): whether to overwrite existing files. Defaults to False.
    
    Returns:       
    None
    """
    
    
    
    print(f"square: {square}")
    print(f"main_path: {main_path}")
    print(f"overwrite: {overwrite}")

    voxelsize_mm = int(voxelsize * 1000)  # convert to mm

    ######## stack the point clouds into one point cloud
    # Check for merged point cloud data otherwise merge the point clouds

    merged_directory = f'{main_path}/{square}_merged_noClip'
    all_points_ds_path = f'{merged_directory}/{square}_all_points_DS{voxelsize_mm}mm.laz'

    list_of_filepaths = [
        f'{merged_directory}/{square}_quad1_DS{voxelsize_mm}mm.laz',
        f'{merged_directory}/{square}_quad2_DS{voxelsize_mm}mm.laz',
        f'{merged_directory}/{square}_quad3_DS{voxelsize_mm}mm.laz',
        f'{merged_directory}/{square}_quad4_DS{voxelsize_mm}mm.laz'
    ]
    
    if not os.path.exists(all_points_ds_path) and not overwrite:
        print(f"Preparation of Merging points of {square}...")
        # merge all the point clouds in /{square}_merged_noClip folder into one point cloud via np.stack
        # Load the point cloud data from the merged folder
        
        # number of files to merge
        num_files = len(list_of_filepaths)
        print(f"Number of files to merge: {num_files}")
        counter = 1
        
        # Initialize an empty array to hold all points
        all_points_ds = np.empty((0, 6))  # Assuming 6 columns: X, Y, Z, Intensity, ReturnNumber, NumberOfReturns
        
        # Loop through each file path and load the point cloud data 
        for file_path in list_of_filepaths:
            print(f"Loading point cloud {counter} of {num_files} from {file_path}...")
            points, pd_points = load_laz_file(file_path)
            
            # Stack the points to the all_points array
            all_points_ds = np.vstack((all_points_ds, points))
            
            # Increment the counter
            counter += 1

            
        # Print the shape of the all_points array
        print(f"All points shape: {all_points_ds.shape}")
        # Print the first few points for verification
        print("First few points of all points:")
        print(all_points_ds[:10])

        # Convert the point cloud data to a pandas DataFrame for saving
        pd_all_points = pd.DataFrame(all_points_ds, 
                                    columns=['X', 'Y', 'Z', 
                                            'Intensity', 'ReturnNumber', 'NumberOfReturns'])
        # save
        save_laz_file(all_points_ds_path, pd_all_points)

        
        # Visualize the merged point cloud
        # visualize_xy_z(all_points_ds, sample_size=100000, save=False)

    else:
        print("Merged point cloud file already exists. Skipping merging step.")
        


############################
#define the main path if the script is main
if __name__ == "__main__":
    square = 'miesbacher'
    main_path = f'tls_data/{square}_reg'
    overwrite = False  # Set to True to overwrite the existing CSV file
