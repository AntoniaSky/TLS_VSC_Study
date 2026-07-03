#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Autumn 2025, last updated June 2026

Cut to Height range (e.g. remove points below 0 m and above 40 m) to focus on vegetation and remove noise from below ground and outliers above canopy
After normalization, classification, denoising, but before feature calculation


Author: Antonia Hostlowsky (assisted by ChatGPT, CoPILOT)
"""

#import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import open3d as o3d    

from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar

from StyleMaps import quads

def cut_height(square: str, main_path: str, min_height: float = 0.0, max_height: float = 40.0, overwrite: bool = False):
    """Cut the point cloud to a specified height range, removing points below min_height and above max_height."""
    
    print(f"square: {square}")
    print(f"main_path: {main_path}")
    print(f"min_height: {min_height}")
    print(f"max_height: {max_height}")
    print(f"overwrite: {overwrite}")

    for quad in quads:
        print(f"Cutting quadrant: {quad}")

        # --------Define paths
        path_before = f'{main_path}/{square}_CC/{square}_{quad}_DS5mm_origCC_DN.laz'  # input
        path_after_cut = f'{main_path}/{square}_CC/{square}_{quad}_DS5mm_FINAL.laz'  # cut to height threshold, output

        # Check if input paths exist
        if not os.path.exists(path_before):
            print(f"Original file {path_before} not found. Skipping this quadrant.")
            continue
        
        # ---- Check if file already exists or 
        if not overwrite and os.path.exists(path_after_cut):
            print(f"Cut file {path_after_cut} already exists. Skipping cutting.")
        else:
            # --------- Load and start
            print("Starting CUTTING process...")
            print("Loading all points from the file...")
            points, pd_points = load_laz_file(path_before)

            print("Preparation for CUTTING the point cloud...")
                
            #print max of z
            print(f"Max Z value: {pd_points['Z'].max()}")

            # ------- Threshold settings 
            # Z minimum height
            min_height = min_height  # in meters
            print(f"Cutting points below {min_height} m... (Default -0.0 m)")
            
            # Z maximum height
            max_height = max_height  # in meters
            print(f"Cutting points above {max_height} m... (Default 40 m)") 
            # likely unnecessary for most cases as denoising should have removed outliers

            # Apply height filtering
            height_filter = (pd_points['Z'] >= min_height) & (pd_points['Z'] <= max_height)
            pd_points_cut = pd_points[height_filter]


            #print max of z after cutting height
            print(f"Max Z value after cutting height: {pd_points_cut['Z'].max()}")
            print(f"Number of points before vs after cutting height: {pd_points.shape[0]} vs {pd_points_cut.shape[0]}")

            #save the cut point cloud to a new file
            save_laz_file(path_after_cut, pd_points_cut)

            #print 3 empty rows for better readability
            print("\n\n\n")

    print("Cutting process completed for all quadrants.")



##############
if __name__ == "__main__":
    square = 'nikolai'
    main_path = f'tls_data/{square}_reg'
    overwrite = False  # Set to True to overwrite the existing files
    cut_height(square=square, main_path=main_path, overwrite=overwrite) #use default min and max height

