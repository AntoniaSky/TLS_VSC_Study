#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Autum 2025, last updated June 2026

This script processes registered point cloud data of individual scan positions with
- cutting to rough AOI
- filtering based on deviation and reflectance
- cutting based on height above sea level (bottom cut, height theshold will be set after normalization)
- RESULT: single point cloud per scan position with only points in the rough AOI and with good quality (based on deviation and reflectance) - saved in folder {square}_singlePos

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)

"""


# Import necessary libraries and files
import laspy
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os   

from definitions_a import load_laz_file, save_laz_file, transform_to_polar, visualize_xy_z, visualize_transformed_points, density_plot 
from definitions_a import load_sop_backup_csv

def prepare_raw_point_clouds(square: str, main_path_disk: str, main_path: str, devReflFiltering: bool = True, overwrite: bool = False, visualize_yes: bool = False, manual_csv_path: str = None) -> None:
    """
    This function processes point cloud data by loading verifying and cutting it to rough AOI.
    
    Parameters:
        square (str): The name of the square area being processed.
        main_path (str): The main directory path where data is stored.
        devReflFiltering (bool): If True, filtering based on deviation and reflectance will be applied.
        overwrite (bool): If True, existing files will be overwritten.
        visualize_yes (bool): If True, visualizations of the point clouds will be displayed.
        manual_csv_path (str): Optional path to a manual CSV file containing file names and paths. If None, the function will automatically generate this information.

    Returns:
        None (only saves the outputs in between)
    """
    
    
    # ------ Load SOP backup CSV if needed
    # sop_backup_df = load_sop_backup_csv(square, main_path)  # Load SOP backup CSV if needed

    # ------ Define paths 
    # paths for the processed csv file
    csv_filepath = f'{main_path}/{square}.csv'
    updated_filepath = f'{main_path}/{square}_updated.csv'
    processed_filepath = f'{main_path}/{square}_processed.csv'

    #  directory containing the .laz files after MultiStationAdjustment
    laz_directory = f'{main_path_disk}/'

    # Define the output folder path for the processed point clouds
    output_file_path = f"{main_path}/{square}_singlePos"
    os.makedirs(output_file_path, exist_ok=True)  


    # define the merged directory for the cloud of each measuringPos
    merged_directory = f'{main_path}/{square}_merged_singlePos/'
    os.makedirs(merged_directory, exist_ok=True)  

    # Print the processing information
    print(f"Processing square: {square}")
    print(f"Main path: {main_path}")
    print(f"Overwrite existing files: {overwrite}")

    # ------ Dataframe with ScanPositions and Tilt (Vertical/Horizontal)
    # Load Manual Dataframe or create a dataframe with the file names in a specific folder
    # to later load the point clouds from the file names and specify the tilt + Position

    if manual_csv_path is not None:
        print(f"Using manual CSV file at: {manual_csv_path} for scan position corrections.")
    else:
        print(f"No manual CSV file found for {square}. Proceeding with automatic generation.")
        
        # Create data frame if not yet exists
        if not overwrite and os.path.exists(csv_filepath): 
            print(f"Using existing {square}.csv file...")

        else:
            print(f"Creating new {square}.csv file...")
            # Get a list of all .laz files in the raw laz directory
            laz_files = [f for f in os.listdir(laz_directory) 
                        if f.endswith('.laz')
                        if not f.startswith('.')]  # Exclude hidden files

            # Print the list of .laz files
            print("Found .laz files:")
            for f in laz_files:
                print(f)
            # Create a DataFrame with the file names
            file_names_df = pd.DataFrame(laz_files, columns=['file_name'])
            file_names_df['file_path'] = laz_directory + file_names_df['file_name']
            
            # make new Column out of the name: for that seperate by " - " and take the first part as columne "scanPosName"
            file_names_df['scanPosName'] = file_names_df['file_name'].str.split(' - ').str[0]

            # sort the DataFrame by file name
            file_names_df = file_names_df.sort_values(by='file_name').reset_index(drop=True)

            # Print the DataFrame
            print("DataFrame with file names and paths:")
            print(file_names_df)

            # add more columns to the DataFrame for tilt and position
            file_names_df['tilt'] = 'default'  # Default tilt value, can be modified
            file_names_df['Measuring_position'] = 0  # Default Measuring position value, can be modified

            #export the DataFrame to a CSV file
            file_names_df.to_csv(f'{main_path}/{square}.csv', index=False)

            #  Set the tilt and position of the point clouds based on the file name and the assumption that vertical and horizontal scans are alternating and that every two scans belong to one position
            if not overwrite and os.path.exists(updated_filepath):
                # If the CSV file already exists, load it
                print(f"Loading existing {square}_updated.csv file...")
                file_names_df = pd.read_csv(updated_filepath)
                # print the number of rows in the DataFrame
                print(f"Number of rows in {square}_updated.csv: {len(file_names_df)}")
                # print the first few rows of the DataFrame
            else:
                print(f"Creating new {square}_updated.csv file...")
                # add new column as index called ScanPos starting with 1
                file_names_df['ScanPos'] = file_names_df.index + 1

                # every scan position which is uneven has tilt = vertical
                file_names_df.loc[file_names_df['ScanPos'] % 2 == 1, 'tilt'] = 'vertical'
                file_names_df.loc[file_names_df['ScanPos'] % 2 == 0, 'tilt'] = 'horizontal'


                # set the position of the point clouds 
                # always two ScanPos are a pair
                file_names_df['Measuring_position'] = (file_names_df['ScanPos'] + 1) // 2  # // means integer division

                # save the updated DataFrame to a new CSV file
                file_names_df.to_csv(updated_filepath, index=False)

    # ------ Loop through the DataFrame and process the each individual point cloud of a square and save the results to individual files per scanposition in the folder {square}_singlePos
    #decide which csv to use
    if manual_csv_path is not None:
        file_names_df = pd.read_csv(manual_csv_path)
    else:
        file_names_df = pd.read_csv(updated_filepath)

    # check if processed file already exists and not overwrite so skip processing
    if not overwrite and os.path.exists(processed_filepath):
            # load the csv file with the file names and paths
            print(f"Processed file {processed_filepath} already exists. Skipping processing step.")
    else:
        print(f"Processing point clouds and saving to {output_file_path}...")
        # add column for processed points
        file_names_df['statusProcess'] = 0 
        file_names_df['newFilePath'] = None

        # For row thus for each file, load the point cloud data 
        for index, row in file_names_df.iterrows():
            #check if overwrite = false and statusProcess= 1
            if not overwrite and row['statusProcess'] == 1:
                print(f"File {row['file_name']} has already been processed.")
                continue
            else:
                print(f"starting to process file: {index + 1} of {len(file_names_df)}")
                file_path = row['file_path']
                file_name = row['file_name']
                tilt = row['tilt']
                position = row['Measuring_position']

                # Load the point cloud data and name the points after file name
                print(f"Loading point cloud data from {file_path}...")
                points, pd_points = load_laz_file(file_path, devReflFiltering=devReflFiltering)
                numberOfColumns = points.shape[1]  # Get the number of attributes in the point cloud data
                column_names = pd_points.columns.tolist()  # Get the attributes names from the DataFrame
                print(f"Point cloud data shape: {points.shape}, Columns: {column_names}")
                print(f"First few rows of the point cloud data:")
                print(pd_points.head())
                
                if devReflFiltering and 'Deviation' in pd_points.columns:
                    initial_point_count = pd_points.shape[0]
                    deviation_threshold = 15  # set the threshold for deviation
                    pd_points = pd_points[pd_points['Deviation'] <= deviation_threshold] # check setting
                    filtered_point_count = pd_points.shape[0]
                    print(f"{square}: Removed {initial_point_count - filtered_point_count} points with Deviation > {deviation_threshold}")
                    # save txt file with number of points removed
                    with open(f"{output_file_path}/points_removed_{square}_{position}_{tilt}.txt", "w") as f:
                        f.write(f"Removed {initial_point_count - filtered_point_count} points with Deviation > {deviation_threshold}\n")
                        f.write(f"Total points after filtering: {filtered_point_count}\n")
                    # drop the Deviation column after filtering
                    pd_points = pd_points.drop(columns=['Deviation'])
                    # convert back to numpy array
                    points = pd_points.to_numpy()
                
                if devReflFiltering and 'Reflectance' in pd_points.columns:
                    print(type(pd_points['Reflectance']))
                    # Convert the entire Laspy ScaledArrayView column to a numeric NumPy array
                    if isinstance(pd_points['Reflectance'], laspy.point.dims.ScaledArrayView):
                        pd_points['Reflectance'] = np.asarray(pd_points['Reflectance'])
                    # remove the points with Reflectance < -27.5
                    initial_point_count = pd_points.shape[0]
                    reflectance_threshold_lower = -20  # set the threshold for reflectance
                    reflectance_threshold_upper = 5  # set the upper threshold for reflectance
                    pd_points = pd_points[pd_points['Reflectance'] > reflectance_threshold_lower] # check setting
                    pd_points = pd_points[pd_points['Reflectance'] < reflectance_threshold_upper] # check setting
                    filtered_point_count = pd_points.shape[0]
                    print(f"{square}: Removed {initial_point_count - filtered_point_count} points with Reflectance < {reflectance_threshold_lower} or > {reflectance_threshold_upper}")
                    # save txt file with number of points removed
                    with open(f"{output_file_path}/points_removed_{square}_{position}_{tilt}.txt", "a") as f:
                        f.write(f"Removed {initial_point_count - filtered_point_count} points with Reflectance < {reflectance_threshold_lower} or > {reflectance_threshold_upper}\n")
                        f.write(f"Total points after filtering: {filtered_point_count}\n")
                    # drop the Reflectance column after filtering
                    pd_points = pd_points.drop(columns=['Reflectance'])
                    # convert back to numpy array
                    points = pd_points.to_numpy()
            
                """ # Cut distance from origin if SOP backup available
                if sop_backup_df is not None:
                    cut_distance = 15 # cut distance in meters
                    
                    # scanposname should be same as in sop_backup_df
                    scanPosName = row['scanPosName']
                    print(f"Processing SOP of scanPosName: {scanPosName}")
    
                    # get the row from sop_backup_df with the same scanPosName
                    sop_row = sop_backup_df[sop_backup_df['scanPosName'] == scanPosName]
                    
                    if not sop_row.empty:
                        print(f"Cutting point cloud to {cut_distance} m from origin based on SOP backup...")
                        #get the x and y of the scan position from sop_row
                        sop_x = sop_row['x_sop'].values[0]
                        sop_y = sop_row['y_sop'].values[0] # 0 means here take the first value of the array
                        # print the scan position
                        print(f"Scan position for Measuring_position {scanPosName}: x={sop_x}, y={sop_y}")
                        # move the points to origin based on scan position
                        pd_points['X'] = pd_points['X'] - sop_x
                        pd_points['Y'] = pd_points['Y'] - sop_y
                        
                        # calculate distance from origin for each point
                        pd_points['distance_from_origin'] = np.sqrt(pd_points['X']**2 + pd_points['Y']**2)
                        
                        # filter points within the cut distance
                        initial_point_count = pd_points.shape[0]
                        pd_points = pd_points[pd_points['distance_from_origin'] <= cut_distance]
                        filtered_point_count = pd_points.shape[0]
                        print(f"{square}: Removed {initial_point_count - filtered_point_count} points beyond {cut_distance} m from origin")
                        # save to txt file with number of points removed
                        with open(f"{output_file_path}/points_removed_{square}_{position}_{tilt}.txt", "a") as f:
                            f.write(f"Removed {initial_point_count - filtered_point_count} points beyond {cut_distance} m from origin\n")
                            f.write(f"Total points after filtering: {filtered_point_count}\n")
                        
                        # drop the distance column after filtering
                        pd_points = pd_points.drop(columns=['distance_from_origin'])
                        
                        # move the points back to original position
                        pd_points['X'] = pd_points['X'] + sop_x
                        pd_points['Y'] = pd_points['Y'] + sop_y
                        # mirror back to numpy array even not needed anymore
                        points = pd_points.to_numpy()
                    else:
                        print(f"No SOP backup entry found for Measuring_position {scanPosName}. Skipping distance cut.")
                 """
               
                # Cut very low points based already on height above sea level Munich should be safe to use 550 m 
                initial_point_count = pd_points.shape[0]
                height_threshold = 550  # set the threshold for height above sea level in meters
                pd_points = pd_points[pd_points['Z'] >= height_threshold]
                filtered_point_count = pd_points.shape[0]   
                print(f"{square}: Removed {initial_point_count - filtered_point_count} points below {height_threshold} m above sea level")
                # save txt file with number of points removed
                with open(f"{output_file_path}/points_removed_{square}_{position}_{tilt}.txt", "a") as f:
                    f.write(f"Removed {initial_point_count - filtered_point_count} points below {height_threshold} m above sea level\n")
                    f.write(f"Total points after filtering: {filtered_point_count}\n")
               
                # Print the first few rows of the point cloud data
                print(f"First few rows of the point cloud data:")
                print(pd_points.head())
                
                # save the point cloud data to a new file
                new_path = f"{output_file_path}/{position}_{tilt}_{file_name}"
                save_laz_file(new_path, pd_points)

                # update the file_names_df with the new file path
                file_names_df.at[index, 'newFilePath'] = new_path
                
                # Debugging Option: Visualize the point cloud data in 2D
                # print(f"Visualizing point cloud data {file_name} in 2D...")
                #visualize_xy_z(points)
                
                # change the statusProcess column to 1
                file_names_df.at[index, 'statusProcess'] = 1

        # sort the DataFrame by Measuring_position and tilt
        file_names_df = file_names_df.sort_values(by=['Measuring_position', 'tilt']).reset_index(drop=True)

        # save the processed DataFrame with Status and new filepaths to a final processed CSV file
        file_names_df.to_csv(processed_filepath, index=False)

    # -------- Loop trough Dataframe & Merge the point clouds with the Same position and save them into merged_noClip
    file_names_df = pd.read_csv(processed_filepath)

    # check how many positions there are
    positions = file_names_df['Measuring_position'].unique()
    print(f"Unique positions found: {positions}")


    for position in positions:
        if not overwrite and os.path.exists(f"{merged_directory}/{square}_{position}_merged.laz"):
            print("Measuring Position already has merged vertical and horizontal scan")
       
        else:  
            print(f"Processing position: {position} ")
            
            position_df = file_names_df[file_names_df['Measuring_position'] == position]
            print(position_df)
            
            #if only on tilt is available then save the point cloud as is without merging and print a warning that only one tilt is available for the position and check the csv file for the position if the tilt is correctly specified
            if len(position_df) < 2:
                single_file_path = position_df.iloc[0]['newFilePath']
                print(f"Only one tilt available for position {position}. Saving the point cloud as is.")
                
                # load save the single point cloud to merged directory without merging
                np_single, pd_single = load_laz_file(single_file_path)
                merged_file_path = os.path.join(merged_directory, f"{square}_{position}_merged.laz")
                save_laz_file(merged_file_path, pd_single)
                
                #remove memory variables
                del np_single
                del pd_single
                continue
            # if both tilts are available then merge the point clouds and save them in the merged directory
            else:
                # load the two point clouds for the current position
                for index, row in position_df.iterrows():
                    tilt = row['tilt']
                    position = row['Measuring_position']
                    newFilePath = row['newFilePath']
                    if tilt == 'default':
                        # raise an error if tilt is not defined
                        raise ValueError(f"Tilt angle must be specified. Please check the CSV file for position {position}.")
                    elif tilt == 'vertical':
                        cut_vertical, pd_vertical = load_laz_file(newFilePath)
                    elif tilt == 'horizontal':
                        cut_horizontal, pd_horizontal = load_laz_file(newFilePath)
                    else:
                        raise ValueError(f"Unknown tilt angle: {tilt}. Please specify 'vertical' or 'horizontal'.")
                    
                # combine the point clouds stack the two point clouds
                points_combined = np.vstack((cut_vertical, cut_horizontal)) 
        
                # convert to pandas DataFrame and check the column names
                numberOfColumns = points_combined.shape[1]  # Get the number of columns
                if numberOfColumns == 6:
                    merged_pd_points = pd.DataFrame(points_combined, columns=['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns'])
                elif numberOfColumns == 7:
                    merged_pd_points = pd.DataFrame(points_combined, columns=['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns', 'Classification'])
                else:
                    raise ValueError(f"Point cloud data {file_name} issues with column names, please check the data.")
                
                print("First few rows of the merged point cloud data:")
                print(merged_pd_points.head())
                print(f"Merged point clouds shape for position {position}: {merged_pd_points.shape}")
                
                # Save the merged point cloud data to a new .laz file
                merged_file_path = os.path.join(merged_directory, f"{square}_{position}_merged.laz")
                save_laz_file(merged_file_path, merged_pd_points)

                #back to numpy array
                merged_comb_points = merged_pd_points.to_numpy()
            
                if visualize_yes: 
                    # visualize the merged point cloud data in 2D
                    print("Visualizing merged point cloud data in 2D...")
                    #Diagram of the point cloud y-axis is height and x-axis is X, colour is Z

                    #random sample of 80000 points
                    sample_size = 80000
                    sample_indices = np.random.choice(merged_comb_points.shape[0], size=sample_size, replace=False)
                    
                    # Create a random sample of the point cloud
                    points_sample = merged_comb_points[sample_indices]
                    # Create a 2D scatter plot of the point cloud # Create a 2D scatter plot of the point cloud 
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111)
                    # Create a scatter plot of the point cloud
                    ax.scatter(points_sample[:, 1], points_sample[:, 2], 
                            c=points_sample[:, 2], 
                            cmap='viridis', 
                            s=0.5)
                    # Add a color bar
                    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
                    cbar.set_label('Z axis')
                    # Set the labels
                    ax.set_xlabel('Y')
                    ax.set_ylabel('Z')
                    # Set the title
                    ax.set_title(f'Point Cloud of Merged Position {position} of {square}')
                    plt.axis('equal')  # Set the aspect ratio to equal
                    plt.show() 
                

# Use the following to run the script for on its own for testing - use the loop in PreprocessingPointClouds.py for the whole processing pipeline
if __name__ == "__main__":
    # define the square name and main path for testing
    square_list = ["jakobgelb"]
    
    for square in square_list:
        main_path_disk = f"/Volumes/T7_Shield_A/msc/{square}_raw"
        main_path =  f"tls_data/{square}_reg"
        os.makedirs(f'{main_path}/{square}_singlePos', exist_ok=True) 
        os.makedirs(f'{main_path}/{square}_merged_singlePos', exist_ok=True)
        os.makedirs(f'{main_path}/{square}_merged_noClip', exist_ok=True)
        
        overwrite = False

        prepare_raw_point_clouds(square=square, main_path_disk=main_path_disk, main_path=main_path, overwrite=overwrite)

