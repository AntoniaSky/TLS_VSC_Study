#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Autum 2025, last updated June 2026

This script merges multiple point cloud files and saves the result as .laz file

Option 1 (for smaller point clouds): Merge all point clouds into one file (example: Alpenplatz)
Option 2 (for larger point clouds): Merge the point clouds into four quadrants and save them separately to avoid memory issue (example: Königsplatz)
The quadrants are defined as follows:
- Quadrant 1: X >= 0 and Y >= 0
- Quadrant 2: X < 0 and Y >= 0
- Quadrant 3: X < 0 and Y < 0
- Quadrant 4: X >= 0 and Y < 0

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
""" 

# import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import open3d as o3d    
from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar
import datetime

from StyleMaps import quads #quad_list

# def merge_to_one
def merge_to_one(square, main_path, overwrite=False):
    """
    Merges multiple point cloud files into one cloud per Site and saves the result.
    
    Parameters:
    - square: Name of the area/square being processed.
    - main_path: Main directory path where data is stored.
    - overwrite: If True, existing files will be overwritten.
    
    Returns:
    - None (saves the merged point cloud to a file)
    """

    print(f"square: {square}")
    print(f"main_path: {main_path}")
    print(f"overwrite: {overwrite}")

    ###### time stamp
    time_start = datetime.datetime.now()
    print(f"Script started at: {time_start} for square: {square}")

    ###### Path for the final merged point cloud file
    all_points_path = f'{main_path}/{square}_merged_noClip/{square}_all_points.laz'
    all_points_folder = f'{main_path}/{square}_merged_noClip'
    os.makedirs(all_points_folder, exist_ok=True)

    ###### Merge all the point clouds in /{square}_merged_noClip folder into one point cloud via np.stack
    if not os.path.exists(all_points_path) or overwrite:
        print(f"Preparation of Merging points of {square}...")
        
        # Load the point cloud data from the merged folder
        ScanPos_directory = f'{main_path}/{square}_merged_singlePos/'
        merged_file_paths = [os.path.join(ScanPos_directory, f) for f in os.listdir(ScanPos_directory) if f.endswith('.laz')]

        # number of files to merge
        num_files = len(merged_file_paths)

        #exeption: do not list the file 'SQUARE_all_points.laz' as it is the output file (In case it exists but overwrite is True)
        merged_file_paths = [f for f in merged_file_paths if f'{square}_all_points.laz' not in f] 
        
        # number of files to merge
        num_files = len(merged_file_paths)

        # Debugging Option: load first laz file and visualize it in 2d for estimation of AOI
        #points, pd_points = load_laz_file(merged_file_paths[0])
        #picture_path = f'{main_path}/{square}_merged_noClip/{square}_FirstXYPlane.png'
        #visualize_xy_z(points, sample_size=100000, save=True, path=picture_path)
        
        # Manually check the extent of AOI from the 2d visualization & Define the extent of the area of interest (AOI) based on the visualization
        x_min, x_max = -40, 40
        y_min, y_max = -40, 40

        print(f"AOI defined: X from {x_min} to {x_max}, Y from {y_min} to {y_max}")


        counter = 1
        print(f"Number of files to merge: {len(merged_file_paths)}")
        
        # load offset from center data
        if os.path.exists('data/Offset_from_Center.csv'):
            offset_df = pd.read_csv('data/Offset_from_Center.csv')
        else:
            offset_df = pd.DataFrame() 

        # Initialize an empty array to hold all points
        all_points = np.empty((0, 6))  # Assuming 6 columns: X, Y, Z, Intensity, ReturnNumber, NumberOfReturns
        
        # MERGING STEP: Loop through each file path and load the point cloud data 
        for file_path in merged_file_paths:
            print(f"Loading point cloud {counter} of {num_files} from {file_path}...")
            points, pd_points = load_laz_file(file_path)
            
            # Get the offsets from the CSV file
            offsets = offset_df[offset_df['Square'] == square]
            if not offsets.empty:
                x_offset = offsets['x_offset_center'].values[0]
                y_offset = offsets['y_offset_center'].values[0]
                # Apply the offsets to the point cloud
                pd_points['X'] -= x_offset
                pd_points['Y'] -= y_offset
                print(f"Applied offsets: x_offset={x_offset}, y_offset={y_offset}")
            else:
                print("No offsets provided.")

            # Filter points based on the rough AOI
            pd_points = pd_points[(pd_points['X'] >= x_min) & (pd_points['X'] <= x_max) & 
                                    (pd_points['Y'] >= y_min) & (pd_points['Y'] <= y_max)]
            
            # Convert the DataFrame back to a numpy array
            points = pd_points.to_numpy()   
            
            # Stack the points to the all_points array
            all_points = np.vstack((all_points, points))
            
            # Increment the counter
            counter += 1
            
        # Print the shape of the all_points array
        print(f"All points shape: {all_points.shape}")
        # Print the first few points for verification
        print("First few points of all points:")
        print(all_points[:10])  
        
        # Convert the point cloud data to a pandas DataFrame for saving
        pd_all_points = pd.DataFrame(all_points, 
                                    columns=['X', 'Y', 'Z', 
                                            'Intensity', 'ReturnNumber', 'NumberOfReturns'])
        # save with the correct scale
        save_laz_file(all_points_path, pd_all_points)

        # Debugging step: visualize in 2d
        #visualize_xy_z(all_points, sample_size=100000, save=False)

        # delete cached variables
        del all_points
        del pd_all_points
        del points
        del pd_points
        
    else:
        print("All points file already exists. Skipping merging step.")

    # time stamp
    time_end = datetime.datetime.now()
    print(f"Script ended at: {time_end}")
    print(f"Total duration: {time_end - time_start}")
    
    
def merge_quadrantwise(square, main_path, overwrite=False):
    """
    Merges multiple point cloud files into one cloud per Quadrant and saves the result.
    
    Parameters:
    - square: Name of the area/square being processed.
    - main_path: Main directory path where data is stored.
    - overwrite: If True, existing files will be overwritten.
    
    Returns:
    - None (saves the merged point cloud to files per quadrant)
    """
    
    print(f"square: {square}")
    print(f"main_path: {main_path}")
    print(f"overwrite: {overwrite}")

    # time stamp
    time_start = datetime.datetime.now()
    print(f"Script started at: {time_start} for square: {square}")

    # stack the point clouds into one point cloud
    # Check for merged point cloud data otherwise merge the point clouds
    ScanPos_directory = f'{main_path}/{square}_merged_singlePos/'
    output_directory = f'{main_path}/{square}_merged_noClip/'
    os.makedirs(output_directory, exist_ok=True)
    all_points_quad4_path = f'{output_directory}/{square}_all_points_quad4.laz'


    if not os.path.exists(all_points_quad4_path) or overwrite:
        print(f"Preparation of Merging points of {square}...")
        
        ###### Prepare data frame for loading the point clouds
        merged_file_paths = [os.path.join(ScanPos_directory, f) for f in os.listdir(ScanPos_directory) if f.endswith('.laz')]

        #exception: do not list the file 'SQUARE_all_points.laz' as it is the output file
        merged_file_paths = [f for f in merged_file_paths if f'{square}_all_points.laz' not in f] 
        merged_file_paths = [f for f in merged_file_paths if f'{square}_all_points_quad1.laz' not in f] 
        merged_file_paths = [f for f in merged_file_paths if f'{square}_all_points_quad2.laz' not in f] 
        merged_file_paths = [f for f in merged_file_paths if f'{square}_all_points_quad3.laz' not in f] 
        merged_file_paths = [f for f in merged_file_paths if f'{square}_all_points_quad4.laz' not in f] 
        
        # number of files to merge
        num_files = len(merged_file_paths)
        
        ###### AOI: enter the extent of the AOI based on the visualization here
        x_min, x_max = -55, 55
        y_min, y_max = -55, 55
        print(f"AOI defined: X from {x_min} to {x_max}, Y from {y_min} to {y_max}")

        quads_basic = ["quad1", "quad2", "quad3", "quad4"]

        ###### Offset: load offset from center data
        if os.path.exists('data/Offset_from_Center.csv'):
            offset_df = pd.read_csv('data/Offset_from_Center.csv')
        else:
            offset_df = None
        
        ####### Existing Merged Quads: Make a list to check which quadrants are in the output folder
        existing_files = [f for f in os.listdir(output_directory) if f.endswith('.laz') and not f.startswith(".")] 

        ###### Loop: Merge the point clouds into quadrants and save them separately to avoid memory issue
        for quad in quads_basic:
            ###### Output path
            all_points_quad_path = f'{output_directory}/{square}_all_points_{quad}.laz'
            
            # check if a file for a part of the quadrant already exists in the output folder
            already_exists = any(f'{quad}' in s for s in existing_files)
            
            # check if the file exists or if overwrite is True or if any existing files contains the quadrant name
            if (not os.path.exists(all_points_quad_path) and not already_exists) or overwrite:
                print(f"Processing {quad}...")
                # Initialize an empty array to hold all points
                all_points_quad = np.empty((0, 6))  # Assuming 6 columns: X, Y, Z, Intensity, ReturnNumber, NumberOfReturns
                
                counter = 1
                
                # Loop through each file path of the Scanning Positionsand load the point cloud data 
                for file_path in merged_file_paths:
                    print(f"Loading point cloud {counter} of {num_files} from {file_path}...")
                    points, pd_points = load_laz_file(file_path)
                    
                    # Apply Offset: Get the offsets from the CSV file loaded earlier
                    offsets = offset_df[offset_df['Square'] == square] if offset_df is not None else pd.DataFrame()
                    x_offset = offsets['x_offset_center'].values[0] if not offsets.empty else 0
                    y_offset = offsets['y_offset_center'].values[0] if not offsets.empty else 0
                    if pd.isna(x_offset) or pd.isna(y_offset):
                        # print a warning message if the offsets are not provided
                        raise ValueError(f"Warning: No offsets provided. Cloud is not moved from orignal position. Check if Cloud is Centered around the center of Square")
                    else:
                        # Apply the offsets to the point cloud  
                        pd_points['X'] -= x_offset  
                        pd_points['Y'] -= y_offset
                        print(f"Applied offsets: x_offset={x_offset}, y_offset={y_offset}")

                    # Apply AOI: Filter points based on the rough distance based AOI
                    pd_points = pd_points[(pd_points['X'] >= x_min) & (pd_points['X'] <= x_max) & 
                                            (pd_points['Y'] >= y_min) & (pd_points['Y'] <= y_max)]
                    
                    # Apply Quadrant Filter: filter the points based on the quadrant
                    if quad == "quad1":
                        pd_points = pd_points[(pd_points['X'] >= 0) & (pd_points['Y'] >= 0)]
                    elif quad == "quad2":   
                        pd_points = pd_points[(pd_points['X'] < 0) & (pd_points['Y'] >= 0)]
                    elif quad == "quad3":   
                        pd_points = pd_points[(pd_points['X'] < 0) & (pd_points['Y'] < 0)]
                    elif quad == "quad4":   
                        pd_points = pd_points[(pd_points['X'] >= 0) & (pd_points['Y'] < 0)]
                    
                    # Convert the DataFrame back to a numpy array
                    points = pd_points.to_numpy()   
                    
                    # Stack the points to the all_points array
                    all_points_quad = np.vstack((all_points_quad, points))
                    print(f"Current shape of {quad} points: {all_points_quad.shape}")
                    print(f"First few points of {quad}:")
                    print(all_points_quad[:5])

                    # Increment the counter
                    counter += 1
                    
                    """
                    # Debugging Option: save quad to laz to test which file causes problems
                    if counter > 11:
                        pd_all_points = pd.DataFrame(all_points_quad, 
                                                columns=['X', 'Y', 'Z', 
                                                        'Intensity', 'ReturnNumber', 'NumberOfReturns'])
                        test_path = f'/Volumes/T7_Shield_A/{square}_merged_noClip/{square}_all_points_{quad}_{counter}.laz'

                        save_laz_file(test_path, pd_all_points)
                    """
                    
                # Print the shape of the all_points array
                print(f"All points shape: {all_points_quad.shape}")
                # Print the first few points for verification
                print("First few points of all points:")
                print(all_points_quad[:10])

                # Convert the point cloud data to a pandas DataFrame for saving
                pd_all_points = pd.DataFrame(all_points_quad, 
                                            columns=['X', 'Y', 'Z', 
                                                    'Intensity', 'ReturnNumber', 'NumberOfReturns'])
                
                # if the points are too many >150 mio points, split into multiple files using quantiles
                if pd_all_points.shape[0] > 300_000_000:
                    print(f"Point cloud for {quad} has {pd_all_points.shape[0]} points, splitting into multiple files...")
                    files_needed = pd_all_points.shape[0] // 150_000_000 + 1
                    quantiles = np.linspace(0, 1, files_needed + 1)
                    
                    for i in range(files_needed):
                        print(f"Processing part {i+1} of {files_needed}...")
                        lower_quantile = quantiles[i]
                        upper_quantile = quantiles[i + 1]
                        lower_threshold = pd_all_points['Y'].quantile(lower_quantile)
                        upper_threshold = pd_all_points['Y'].quantile(upper_quantile)
                        if i == files_needed - 1:
                            pd_split = pd_all_points[pd_all_points['Y'] >= lower_threshold]
                        else:
                            pd_split = pd_all_points[(pd_all_points['Y'] >= lower_threshold) & 
                                                    (pd_all_points['Y'] < upper_threshold)]
                        split_path = f'{main_path}/{square}_merged_noClip/{square}_all_points_{quad}_part{i+1}.laz'
                        print(f"Saving part {i+1} to {split_path}...")
                        save_laz_file(split_path, pd_split)
                        
                elif pd_all_points.shape[0] > 150_000_000:
                    print(f"Point cloud for {quad} has {pd_all_points.shape[0]} points, splitting into two files...")
                    median_quantile = pd_all_points['X'].quantile(0.5)
                    pd_all_points_lower = pd_all_points[pd_all_points['X'] <= median_quantile]
                    pd_all_points_upper = pd_all_points[pd_all_points['X'] > median_quantile]
                    
                    split_path_lower = f'{main_path}/{square}_merged_noClip/{square}_all_points_{quad}_lower.laz'
                    split_path_upper = f'{main_path}/{square}_merged_noClip/{square}_all_points_{quad}_upper.laz'
                    print(f"Saving lower half to {split_path_lower}...")
                    save_laz_file(split_path_lower, pd_all_points_lower)
                    print(f"Saving upper half to {split_path_upper}...")
                    save_laz_file(split_path_upper, pd_all_points_upper)
                else:
                    print(f"Saving merged point cloud for {quad} to {all_points_quad_path}...")
                    # save the merged point cloud per quadrant as is whithout splitting
                    all_points_quad_path = f'{main_path}/{square}_merged_noClip/{square}_all_points_{quad}.laz'
                    save_laz_file(all_points_quad_path, pd_all_points)
                    
                # delete cached variables
                del all_points_quad
                del pd_all_points
                del points
                del pd_points

            else:
                print(f"All points file for {quad} already exists. Skipping merging step.")
                continue  # Skip to the next quadrant if the file already exists
    else:
        print("All quadrants processed.")
            


    # time stamp
    time_end = datetime.datetime.now()
    print(f"Script ended at: {time_end}")
    print(f"Total duration: {time_end - time_start}")

    # Debugging: visualize in 2d
    #visualize_xy_z(all_points, sample_size=100000, save=False)
    
    
    print(f"MERGING INTO QUADRANTS COMPLETED for square: {square}.")
    print("\n" * 3)


############################################################################################################
# Use the following to run the script for on its own for testing - use the loop in PreprocessingPointClouds.py for the whole processing pipeline
if __name__ == "__main__":
    square_list = ['elisabeth']
    overwrite = False  # Set to True to overwrite the existing CSV file

    for square in square_list:
        main_path = f'tls_data/{square}_reg' # path to square folder for raw, registered data
        merge_to_one(square=square, main_path=main_path, overwrite=overwrite)
        #merge_quadrantwise(square=square, main_path=main_path, overwrite=overwrite)