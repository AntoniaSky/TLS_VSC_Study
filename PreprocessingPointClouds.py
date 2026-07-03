#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a combined script that includes various scripts for processing the TLS Point Cloud Data of Riegl vz 400i without rgb.

Use it after the registration process in RiScanPro. (Automatic Registration 2 and Multi Station Adjustment 2)

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)

Originally created Autum 2025, last updated June 2026
"""

# Import necessary libraries
import os
import open3d as o3d
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import laspy    
import CSF #import CSF library 
from scipy.spatial import cKDTree #import cKDTree for fast nearest neighbor search
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import Delaunay
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator 






    
# Enable line buffering for immediate output of print() statements
sys.stdout.reconfigure(line_buffering=True) 

# Define the list of squares to process. You can modify this list to include the squares you want to process.
square_list = ["nikolai", "wetterstein", "jakobgelb", "elisabeth", "nikolai", "walchensee", "rundfunk"]
square_list = ["alpenplatz"]

#Start the timer to track the duration of the entire script
start_time = datetime.datetime.now()
print(f"Script started at: {start_time}")

####################################################################################
### Loop through each square in the list and run the processing steps for each square
# - need to define: 
#       - "main_path", 
#       - voxelsize for downsampling (default: 0.005 m = 5 mm),
####################################################################################
for square in square_list:
    ############################## Define paths for the current square  
    # path to square folder for raw, registered data #msa = multi station adjustment # folder with tls_data is assumed to be in the same directory as this script
    main_path = f'tls_data/{square}_reg' 
    
    print(f"Processing square: {square}")

    ############################## Define parameters for processing - adjust as needed
    overwrite = False  # Set to True if you want to overwrite existing files, False to skip processing if files already exist
    print(f"Overwrite existing files: {overwrite}")

    # for downsampling
    voxelsize = 0.005  # in m 0.005 means 5 mm
    
    # for open3D library define if GPU or CPU is available - steps here work on a CPU (Macos 2015 Pro)
    device = o3d.core.Device('CPU:0')  # meaning to use the CPU for processing # if gpu is available change settings
    dtype = o3d.core.float32


    ############################## Create Folder Structure
    # {square}_reg --> {square}_singlePos           # all individual scans 
    # {square}_reg --> {square}_mergedSinglePos     # all scan positions
    # {square}_reg --> {square}_merged_noClip       # merged cloud or quadrants (after mergiing, after downsampling, after normalization, .txt files for RandLaNet) # before cleaning in CloudCompare
    # {square}_reg --> {square}_CC                  # cleaned clouds 
    # {square}_reg --> {square}_FINAL               # final cleaned and processed cloud after manual cleaning in CloudCompare (naming convention for following scripts: {square}_square_DS5mm_FINAL.laz)
    
    

    # if not exist then create
    os.makedirs(f'{main_path}/{square}_singlePos', exist_ok=True) 
    os.makedirs(f'{main_path}/{square}_merged_singlePos', exist_ok=True)
    os.makedirs(f'{main_path}/{square}_merged_noClip', exist_ok=True)
    os.makedirs(f'{main_path}/{square}_CC', exist_ok=True)

    ############################## Run the individual scripts in order
    ############ Script 1: Process point cloud data
        # Assumption Vertical scans have uneven ScanPositions and Horizontal even ones (Only true if no scans skipped)
            # --> If not Vertical and Horizontal abwechselnd then manual csv needs to be provided    
        # Assumption tiepoint scans are in seperate folders
    import PointCloudPreprocessing_Methods.processingRegPC_noClip as processingRegPC_noClip
    manual_csv_path = f"{main_path}/{square}_manual.csv"
    if os.path.exists(manual_csv_path):
        print(f"Manual CSV file found at: {manual_csv_path}. Using manual scan position corrections.")
        processingRegPC_noClip.prepare_raw_point_clouds(square=square, 
                                                    main_path_disk=main_path, # path to registered raw data # if not on the same disk as the script, then provide the path to the disk where the data is stored
                                                    main_path=main_path, 
                                                    overwrite=False,
                                                    manual_csv_path=manual_csv_path
                                                    )
    else:
        print(f"No manual CSV file found at: {manual_csv_path}. Using default scan position corrections.")
        processingRegPC_noClip.prepare_raw_point_clouds(square=square, 
                                                    main_path_disk=main_path, 
                                                    main_path=main_path, 
                                                    overwrite=False
                                                    )

    ############ Script 2: Merge point cloud data #large squares need to merged per quadrant (merge.py also provides a method for merging all together but for large clouds it is better to do it in quadrants)
    import PointCloudPreprocessing_Methods.merge as merge
    merge.merge_quadrantwise(square=square, main_path=main_path, overwrite=overwrite)


    ############ Script 3: Downsample data into quadrants of voxel size x (e.g. 5 mm) and save as LAZ file per quadrant (if large cloud, otherwise merge all together and downsample in one step)
    import PointCloudPreprocessing_Methods.downsample as downsample
    downsample.downsample_from_quadrants(square=square, main_path=main_path, voxel_size=voxelsize, 
                                            overwrite=overwrite, device=device, dtype=dtype)

    ###### Script 4: Merge all the quadrants back together in one cloud for small clouds or for large clouds do it manually in CloudCompare 
    import PointCloudPreprocessing_Methods.merge_DS_quadrants as merge_DS_quadrants
    if square in ['nikolai','alpenplatz']: # list only small clouds here 
        merge_DS_quadrants.merge_DS_quadrants(square=square, main_path=main_path, voxelsize=voxelsize, overwrite=overwrite)
 
    ############ Script 5: Classify ground points using CSF and Normalize Cloud
    import PointCloudPreprocessing_Methods.normalizeCSFLowestPointGrid as normalizeCSFLowestPointGrid
    normalizeCSFLowestPointGrid.process_csf(square=square, main_path=main_path, overwrite=overwrite)
    
    ############ Script 6: Prepare the data for Classification with RandLaNet:  Transform the Data into a .txt file 
    import PointCloudPreprocessing_Methods.LazToTxt as LazToTxt
    LazToTxt.convert_laz_to_txt(square=square, main_path=main_path, overwrite=overwrite)

    preprocessing_Time = datetime.datetime.now()
    print(f"Preprocessing completed at: {preprocessing_Time}")

    
    # --- Steps outside of this script:
    ############ Script 7: RandLaNet classification on stronger eg linux machine (Script: PointCloudPreprocessing_Methods/torch_loadersem3d_working.py)
    # CloudCompare: MANUALLY clean the point cloud (removal of urban objects)

    # Before proceeding 
        # Assumption: Result is a classified LAZ file per quadrant with only ground and vegetation points with Attribute X Y Z Intensity Classification
        # Assumption: Resulting classified LAZ files are in folder {square}_CC and end with _CC.laz # donot yet combine the quadrants together
    # ---
    
    #check if in folder a file ending with _CC.laz exists
    path_of_CC_folder = f"{main_path}/{square}_CC"
    classified_laz_files = [f for f in os.listdir(path_of_CC_folder) if f.endswith('_CC.laz')]
    if not classified_laz_files:
        print(f"No classified LAZ files ending with '_CC.laz' found in {path_of_CC_folder}. Please run Classification and save the cleaned clouds to folder before proceeding.")
        #continue with the next square in the list
        continue # to the next iteration of the for loop
    else:
        print(f"Classified LAZ file(s) {classified_laz_files} found. Proceeding to fill back classification.")
        

    ############ Script 8: Combine the attributes Classification and Intensity back to the original merged clouds Attribute Return Number Number of Returns
    import PointCloudPreprocessing_Methods.ClassToPC as ClassToPC
    ClassToPC.fill_classification_back(square=square, main_path=main_path, overwrite=overwrite)

    ############ Script 9: Denoise with Statistical Outlier Removal (SOR) 
    import PointCloudPreprocessing_Methods.denoise as denoise
    denoise.denoise_point_cloud(square=square, main_path=main_path, overwrite=overwrite)

    ############ Script 10: Cut away below ground points
    import PointCloudPreprocessing_Methods.cut_cloud_heights as cut_cloud_heights
    cut_cloud_heights.cut_height(square=square, main_path=main_path, overwrite=overwrite) 
    
    ##### Outside: Manually cut to exact AOI in CloudCompare and save as LAZ file (naming convention for following scripts: {square}_FINAL/{square}_square_DS5mm_FINAL.laz)

    

end_time = datetime.datetime.now()
print(f"Script ended at: {end_time}")
print(f"Total duration: {end_time - start_time}")




# Final is a new folder ending with _FINAL where the final cleaned and processed cloud is saved after all steps in this script and after manual cleaning in CloudCompare.
# The file should be named {square}_square_DS5mm_FINAL.laz (or similar if different voxel size used) and should contain the attributes X Y Z Intensity Classification Return Number Number of Returns.
# This file is then used for the following steps of calculating the metrics per cell and Z-based metrics.