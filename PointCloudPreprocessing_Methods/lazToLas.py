#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Autum 2025

Script for saving a laz file into a las file format.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
"""

#import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import open3d as o3d    

from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar

#define the main path
square = 'alpenplatz'
main_path = f'tls_data/{square}_reg'



# save the point cloud data in LAS format if not already done
path = f'{main_path}/{square}_merged_noClip/{square}_01m_classrf01VG_SmallAOI_norm_orig.laz'

# load the point cloud data
print("Loading all points from the file...")
points = load_laz_file(path)

print("Converting points to DataFrame...")
#check for shape of points
if points.shape[1] == 6:
    pd_all_points = pd.DataFrame(points, 
                                 columns=['X', 'Y', 'Z', 
                                          'Intensity', 'ReturnNumber', 'NumberOfReturns'])
elif points.shape[1] == 7:
    pd_all_points = pd.DataFrame(points, 
                                 columns=['X', 'Y', 'Z', 
                                          'Intensity', 'ReturnNumber', 'NumberOfReturns', 
                                          'Classification'])

# save as las format
print("Saving points to LAS format...")

file_path = f'{main_path}/{square}_merged_noClip/{square}_01m_classrf01VG_SmallAOI_norm_orig.las'
    
# Create a new laspy file and write the points
header = laspy.LasHeader(point_format=7, version="1.4")
new_las = laspy.LasData(header)

# Assign the points to the new LasData object, ensuring correct data types
new_las.x = points[:, 0].astype(np.float32)
new_las.y = points[:, 1].astype(np.float32)
new_las.z = points[:, 2].astype(np.float32)
new_las.intensity = points[:, 3].astype(np.uint16)  # intensity is usually uint16
new_las.return_number = points[:, 4].astype(np.uint8)  # return_number is usually uint8
new_las.number_of_returns = points[:, 5].astype(np.uint8)  # number_of_returns is usually uint8

# If classification is present, assign it as well
if 'Classification' in pd_all_points.columns:
    new_las.classification = points[:, 6].astype(np.uint8)  # classification is usually uint8

#write
new_las.write(file_path)

print(f"Point cloud data saved to {file_path}")
    
# print empty lines for better readability
print("\n" * 5)


