#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Dec 2025, last updated June 2026

Use after Classification with RandLa-Net and manual cleaning in CloudCompare (Script: PointCloudPreprocessing_Methods/torch_loadersem3d_working.py)

This script 
- fills the classification back into the original file with all points and attributes (number of returns, return number, intensity) after the classification process in CloudCompare.
- filters the points to keep only ground and vegetation classes (0-3) and saves the new file with classification as a LAZ file.

Assumptions:
- The classified LAZ files from CloudCompare are in the folder {square}_CC and end with _CC.laz


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
from scipy.interpolate import griddata
from scipy.spatial import cKDTree #import cKDTree for fast nearest neighbor search
from scipy.ndimage import gaussian_filter, uniform_filter


from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar

from StyleMaps import quads

# load the orig cloud per quadrant DS with all points and get classification back in
def fill_classification_back(square: str, main_path: str, overwrite: bool = False):
    """
    Fill the classification back into the original file with all points and attributes (number of returns, return number, intensity) after the classification process in CloudCompare.
    Filters the points to keep only ground and vegetation classes (0-3) and saves the new file with classification as a LAZ file.

    Args:
        square (str): The square name
        main_path (str): The main path where the files are located
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    Returns:
        None - Writes the LAZ file to the specified location
    """
    
    print(f"square: {square}")
    print(f"main_path: {main_path}")
    print(f"overwrite: {overwrite}")
    time_stamp_start = datetime.datetime.now()

             
    for quad in quads:
        print(f"Processing quadrant: {quad}")
        
        # -------- Define paths of classified file, original file and output file based on quadrant
        class_file = f'{main_path}/{square}_CC/{square}_{quad}_DS5mm_CC.laz' # classified single quadrant or all points

        if not os.path.exists(class_file):
            print(f"Class file {class_file} not found. Skipping this quadrant.")
            continue
        
        # ------- Find original file path based on quadrant
        if quad in ["quad1_lower", "quad2_lower", "quad3_lower", "quad4_lower",
                    "quad1_upper", "quad2_upper", "quad3_upper", "quad4_upper"]:
            if os.path.exists(f'{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH5m4Ch04m.laz'):
                orig_file = f'{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH5m4Ch04m.laz' # original single quadrant
            else:
                separator = "_"
                base_quad = quad.split(separator)[0]  # e.g., "quad1"
                orig_file = f'{main_path}/{square}_merged_noClip/{square}_{base_quad}_DS5mm_relH5m4Ch04m.laz' # original single quadrant part
        elif quad == "all_points_lower" or quad == "all_points_upper":
            if os.path.exists(f'{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH5m4Ch04m.laz'):
                orig_file = f'{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH5m4Ch04m.laz' # original single quadrant
            else:
                orig_file = f'{main_path}/{square}_merged_noClip/{square}_all_points_DS5mm_relH5m4Ch04m.laz' # original all points part
        else:
            orig_file = f'{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH5m4Ch04m.laz' # original single quadrant 

        output_path = f'{main_path}/{square}_CC/{square}_{quad}_DS5mm_origCC_VegGround.laz' # output file with classification filled back in (next to number of returns and return number and intensity)

        # -------- Check if file already exists or needs to be overwritten
        if not overwrite and os.path.exists(output_path):
            print(f"Output file {output_path} already exists. Skipping classification fill back.")
            continue
        else:
            print("Starting classification fill back process...")
            time_before = datetime.datetime.now()
            # -------- Load original and classified files
            orig_points, orig_df = load_laz_file(orig_file) # cloud with all points incl return attr. but no classification
            class_points, class_df = load_laz_file(class_file) # cloud with classification

            # print the extent of both files
            print(f"Original file extent: X({orig_df['X'].min()}, {orig_df['X'].max()}), Y({orig_df['Y'].min()}, {orig_df['Y'].max()})")
            print(f"Classified file extent: X({class_df['X'].min()}, {class_df['X'].max()}), Y({class_df['Y'].min()}, {class_df['Y'].max()})")

            # del for memory efficiency
            del orig_points, class_points 

            # -------- Cut to same xy extent as classified file with the pd data frame
            print("Aligning original points to classified points extent...")
            min_x = class_df['X'].min()
            max_x = class_df['X'].max()
            min_y = class_df['Y'].min()
            max_y = class_df['Y'].max()

            orig_df = orig_df[
                (orig_df['X'] >= min_x) & (orig_df['X'] <= max_x) &
                (orig_df['Y'] >= min_y) & (orig_df['Y'] <= max_y)
            ].reset_index(drop=True)
            print(orig_df.head())
            
            # Only keep this if you need orig_points later # align orig_points with filtered orig_df for kd tree approach
            # orig_points = orig_points[orig_df.index] 

            # --------- Dict matching points on xyz and assign classification but memory efficient way
            print("creating keys for classification...")
            # Create a unique coordinate key for both DataFrames
            orig_df['key'] = orig_df[['X', 'Y', 'Z']].round(4).astype(str).agg('_'.join, axis=1)
            class_df['key'] = class_df[['X', 'Y', 'Z']].round(4).astype(str).agg('_'.join, axis=1)
            print(orig_df.head())
            print(class_df.head())
            
            # Map classification values by key
            print(f"Mapping classification values for quadrant {quad}...")
            class_map = dict(zip(class_df['key'], class_df['Classification']))
            
            del class_df  # free up memory
            
            orig_df['Classification'] = orig_df['key'].map(class_map)

            del class_map  # free up memory
            
            
            """ # --------- Fast nearest neighbor search with KDTree
            print("Building KDTree for nearest neighbor search...")
            # build KDTree for fast nearest neighbor search
            tree = cKDTree(class_points[:, :3])  # use only XYZ for matching
            # query nearest neighbors
            distances, indices = tree.query(orig_points[:, :3], k=1)
            # Assign classification from classified file to original file
            print("Assigning classification from classified file to original file...")
            # assign classification from classified file to original file
            orig_df['Classification'] = class_df['Classification'].values[indices]
            print(orig_df.head())
            print(f"Classification values: {orig_df['Classification'].unique()}")
            """
            
            # label unclassified points nan as 9
            print("Filling unclassified points with 9...")
            orig_df['Classification'].fillna(9, inplace=True)


            # --------- Filter points with classification > 3 (keep only ground and vegetation Classes 0, 1, 2, 3)
            print("Filtering points to keep only ground and vegetation classes (0-3)...")
            orig_df = orig_df[orig_df['Classification'] <= 3]
            # Only keep this if you need orig_points later # align orig_points with filtered orig_df 
            # orig_points = orig_points[orig_df.index] 
            print(orig_df.head())
            print(f"Classification values: {orig_df['Classification'].unique()}")

            # --------- Save the new file with classification
            save_laz_file(output_path, orig_df)
            time_after = datetime.datetime.now()
            print(f"Time taken for classification fill back for {quad}: {time_after - time_before}")
            print(f'Processed {quad}, saved to {output_path}')
            
            #del orig_points, orig_df, class_points, class_df  # free up memory
            del orig_df
            
            
    print("Classification fill back process completed for all quadrants.")
    time_stamp_end = datetime.datetime.now()
    print(f"Total time taken: {time_stamp_end - time_stamp_start}")



######################### Below: For testing the function or running it directly 
if __name__ == "__main__":
    square_list = ['nikolai', 'rundfunk', 'elisabeth', 'miesbacher', 'jakobgelb', 'konigsplatz', 'stjakob', 'wittelsbacher']
    for square in square_list:
        print(f"Processing square: {square}")
        main_path = f'tls_data/{square}_reg'
        overwrite = False  # Set to True to overwrite the existing files
        fill_classification_back(square=square, main_path=main_path, overwrite=overwrite)
