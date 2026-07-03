#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Autum 2025, last updated June 2026

This script loads the laz file and converts the point cloud data to polar coordinates (azimuth angle, polar angle, range) and visualizes the point cloud in polar coordinates.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)

"""

#### Import necessary libraries and files
import laspy
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_laz_file(file_path: str) -> np.ndarray:
    """
    Load a .laz file and return the point cloud data as a numpy array.
    
    Parameters:
    file_path (str): Path to the .laz file.

    Returns:
    np.ndarray: Point cloud data as a numpy array.
    """
    with laspy.open(file_path) as f:
        print(f"Loading file: {file_path}")
        scan = f.read()
        #print the header information
        print(f"Header info: {scan.header}")
        # Print the number of points
        print(f"Number of points: {scan.header.point_count}")
        
        # get filename and extension
        file_name, file_extension = file_path.split('/')[-1].split('.')
        print(f"File name: {file_name}, File extension: {file_extension}")
    
        
        #check the offset of the point cloud data and print it
        print(f"Offset of X: {scan.x.offset}, Offset of Y: {scan.y.offset}, Offset of Z: {scan.z.offset}")
        print(f"Scale of X: {scan.x.scale}, Scale of Y: {scan.y.scale}, Scale of Z: {scan.z.scale}")
        
        
        #check if there are any NaN values in the point cloud data if they are not NaN, then the point cloud is valid
        # raise an error if there are NaN values
        if np.isnan(scan.x).any() or np.isnan(scan.y).any() or np.isnan(scan.z).any():
            # Print whether there are NaN values in the point cloud data
            print("Checking for NaN values in the point cloud data:")
            print(f"NaN in X: {np.isnan(scan.x).any()}")
            print(f"NaN in Y: {np.isnan(scan.y).any()}")
            print(f"NaN in Z: {np.isnan(scan.z).any()}")
            raise ValueError("Point cloud data contains NaN values. Please check the input file.")
        else:
            print("Point cloud data is valid, no NaN values found.")
        
        
        #numpy array of points
        points = np.vstack((scan.x, scan.y, scan.z,
                            scan.intensity,
                            scan.return_number,
                            scan.number_of_returns)).T

        #print points
        if points is not None:
            print("The numpy array exists")
        else:            
            print("The numpy array does not exist")
        
        
        
        # Return the point cloud data as a numpy array
        print(f"points of {file_name} are created; Returning point cloud data as numpy array with shape: {points.shape}")
        
        #print empty lines for better readability
        print("\n" * 5)

        return points

#define downsampling of point clouds to x amount of points 
def downsample_point_cloud(points: np.ndarray, num_points: int) -> pd.DataFrame:
    """
    Downsample a point cloud to x number of points.
    
    Parameters:
    points (np.ndarray): The point cloud data as a numpy array.
    num_points (int): The number of points to downsample to.

    Returns:
    pd.DataFrame: Downsampled point cloud data as a pandas DataFrame.
    """
    pd_scan = pd.DataFrame(points, columns=['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns'])
    pd_scan_downsampled = pd_scan.sample(n=num_points, random_state=42)
    print(f"Downsampled points shape: {pd_scan_downsampled.shape}")
    
    #print empty lines for better readability
    print("\n" * 5)
    
    return pd_scan_downsampled
    

#####load the point clouds

# Define the file paths for the vertical and horizontal scans
file_path_vertical = 'wetterstein/wetters ScanPos001 - SINGLESCANS - 250625_144932.laz'
file_path_horizontal = 'wetterstein/wetters ScanPos002 - SINGLESCANS - 250625_145323.laz'

"""
file_path_vertical = '/Users/anti/Documents/STUDIUM_MASTER/MSC/python/ScanPos001 - SINGLESCANS - 250613_115651 - Not impacted by exit aperture.laz'
file_path_horizontal = '/Users/anti/Documents/STUDIUM_MASTER/MSC/python/ScanPos002 - SINGLESCANS - 250613_120128.laz'
"""
""" file_path_vertical = '/Users/anti/Documents/STUDIUM_MASTER/MSC/python/untitled folder/fors_o_1_h001.laz'
file_path_horizontal = '/Users/anti/Documents/STUDIUM_MASTER/MSC/python/untitled folder/fors_o_1_h002.laz'
 """

# Load the point clouds
points_scan_vertical = load_laz_file(file_path_vertical)
points_scan_horizontal = load_laz_file(file_path_horizontal)


#####Convert the point clouds to pandas DataFrames
# Convert the point clouds to pandas DataFrames without downsampling
pd_scan_vertical = pd.DataFrame(points_scan_vertical, columns=['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns'])
pd_scan_horizontal = pd.DataFrame(points_scan_horizontal, columns=['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns'])

# Filter points based distance from the origin < 50
#vertical scan
pd_scan_vertical['distance_center'] = np.sqrt(pd_scan_vertical['X']**2 + pd_scan_vertical['Y']**2)
# Filter points based on distance from center <  
pd_scan_vertical = pd_scan_vertical[pd_scan_vertical['distance_center'] < 50]
#horizontal scan
pd_scan_horizontal['distance_center'] = np.sqrt(pd_scan_horizontal['X']**2 + pd_scan_horizontal['Y']**2)
# Filter points based on distance from center < 25
pd_scan_horizontal = pd_scan_horizontal[pd_scan_horizontal['distance_center'] < 50]

#####make to polar coordinates
#make the point clouds to polar coordinates

# add columns for azimut and range and polar angle
pd_scan_vertical['AzimuthAngle'] = np.arctan2(pd_scan_vertical['Y'], pd_scan_vertical['X'])
pd_scan_vertical['Range'] = np.sqrt(pd_scan_vertical['X']**2 + pd_scan_vertical['Y']**2 + pd_scan_vertical['Z']**2)
pd_scan_vertical['PolarAngle'] = np.arccos(pd_scan_vertical['Z'] / pd_scan_vertical['Range'])

pd_scan_horizontal['AzimuthAngle'] = np.arctan2(pd_scan_horizontal['Y'], pd_scan_horizontal['X'])
pd_scan_horizontal['Range'] = np.sqrt(pd_scan_horizontal['X']**2 + pd_scan_horizontal['Y']**2 + pd_scan_horizontal['Z']**2)
pd_scan_horizontal['PolarAngle'] = np.arccos(pd_scan_horizontal['Z'] / pd_scan_horizontal['Range'])

 
 #azimut is arcos z r
 #polar ang le is arctan y x
 # range is sqrt(x^2 + y^2 + z^2)
 
""" #vertical scan
pd_scan_vertical['Range'] = np.sqrt(pd_scan_vertical['X']**2 + pd_scan_vertical['Y']**2 + pd_scan_vertical['Z']**2)
pd_scan_vertical['AzimuthAngle'] = np.arctan2(pd_scan_vertical['Y'], pd_scan_vertical['X'])  # azimuth angle in radians
pd_scan_vertical['PolarAngle'] = np.arctan2(pd_scan_vertical['Y'], pd_scan_vertical['X'])


#horizontal scan
pd_scan_horizontal['Range'] = np.sqrt(pd_scan_horizontal['X']**2 + pd_scan_horizontal['Y']**2 + pd_scan_horizontal['Z']**2) 
pd_scan_horizontal['AzimuthAngle'] = np.arctan2(pd_scan_horizontal['Z'], pd_scan_horizontal['Range'])
pd_scan_horizontal['PolarAngle'] = np.arctan2(pd_scan_horizontal['Y'], pd_scan_horizontal['X'])
 """# Print the first 5 rows of the point clouds in polar coordinates
print("Vertical scan point cloud in polar coordinates:")
print(pd_scan_vertical.head())
print("Horizontal scan point cloud in polar coordinates:")
print(pd_scan_horizontal.head())

#print the histogram of the azimuth angle and polar angle
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)        
plt.hist(pd_scan_vertical['AzimuthAngle'], bins=100, color='blue', alpha=0.7)
plt.title('Azimuth Angle Distribution (Vertical Scan)')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Frequency')     
plt.subplot(1, 2, 2)
plt.hist(pd_scan_horizontal['AzimuthAngle'], bins=100, color='orange', alpha=0.7)
plt.title('Azimuth Angle Distribution (Horizontal Scan)')
plt.xlabel('Azimuth Angle (degrees)')
plt.ylabel('Frequency')
plt.show()  

# Print the histogram of the polar angle
plt.figure(figsize=(12, 6))             
plt.subplot(1, 2, 1)
plt.hist(pd_scan_vertical['PolarAngle'], bins=100, color='blue', alpha=0.7)
plt.title('Polar Angle Distribution (Vertical Scan)')   
plt.xlabel('Polar Angle (degrees)')
plt.ylabel('Frequency')
plt.subplot(1, 2, 2)        
plt.hist(pd_scan_horizontal['PolarAngle'], bins=100, color='orange', alpha=0.7)
plt.title('Polar Angle Distribution (Horizontal Scan)')
plt.xlabel('Polar Angle (degrees)')
plt.ylabel('Frequency')
plt.show()

#####Visualize the point clouds in polar coordinates in figures
# downsample the point clouds to 500.000 points
pd_scan_vertical_ds = pd_scan_vertical.sample(n=100000, random_state=42)
pd_scan_horizontal_ds = pd_scan_horizontal.sample(n=100000, random_state=42)

# Visualize the vertical scan point cloud in polar coordinates
# Create a figure with subplots for vertical cut point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)       
# Plot the points in polar coordinates
ax.scatter(pd_scan_vertical_ds['AzimuthAngle'], pd_scan_vertical_ds['PolarAngle'], 
           c=pd_scan_vertical_ds['Z'], cmap='viridis',
           s=0.5) 
# Set title
ax.set_title('3D Point Cloud in Polar Coordinates', va='bottom')    
# Plot the colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
cbar.set_label('Z Value', rotation=270, labelpad=15)

# Show the plot
plt.show()

# Visualize the horizontal scan point cloud in polar coordinates
# Create a figure with subplots for horizontal cut point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)       
# Plot the points in polar coordinates
ax.scatter(pd_scan_horizontal_ds['AzimuthAngle'], pd_scan_horizontal_ds['PolarAngle'], 
           c=pd_scan_horizontal_ds['Z'], cmap='viridis',
           s=0.5)        
# Set title 
ax.set_title('3D Point Cloud in Polar Coordinates', va='bottom')
# Plot the colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
cbar.set_label('Z Value', rotation=270, labelpad=15)
# Show the plot
plt.show()


#



#####cut the point clouds based on zenith angle
#cut the point clouds based on zenith angle

#selecte the points of vertical scan with zenith angle greater or equal than 35°
pd_scan_vertical_cut = pd_scan_vertical[pd_scan_vertical['PolarAngle'] > 0.01]
#selecte the points of horizontal scan with zenith angle less than 40°
pd_scan_horizontal_cut = pd_scan_horizontal[pd_scan_horizontal['PolarAngle'] < 0.015]

# Print the number of points in the cut point clouds
print(f"Number of points in vertical cut point cloud: {pd_scan_vertical_cut.shape[0]}")
print(f"Number of points in horizontal cut point cloud: {pd_scan_horizontal_cut.shape[0]}")

#####Visualize the vertical cut point cloud
# Create a figure with subplots for vertical

# downsample the point cloud
pd_scan_vertical_cut_ds = pd_scan_vertical_cut.sample(n=100000, random_state=42)
pd_scan_horizontal_cut_ds = pd_scan_horizontal_cut.sample(n=100000, random_state=42)


###### Plot the vertical cut point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
# Plot the points in polar coordinates
ax.scatter(pd_scan_vertical_cut_ds['AzimuthAngle'], pd_scan_vertical_cut_ds['PolarAngle'], 
           c=pd_scan_vertical_cut_ds['Z'], cmap='viridis')
# Set title
ax.set_title('3D Point Cloud in Polar Coordinates', va='bottom')
#plot the colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
cbar.set_label('Z Value', rotation=270, labelpad=15)

# Show the plot
plt.show()

# Plot horizontal cut point cloud
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)
# Plot the points in polar coordinates
ax.scatter(pd_scan_horizontal_cut_ds['AzimuthAngle'], pd_scan_horizontal_cut_ds['PolarAngle'],
           c=pd_scan_horizontal_cut_ds['Z'], cmap='viridis')
# Set title
ax.set_title('3D Point Cloud in Polar Coordinates', va='bottom')    
# Plot the colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
cbar.set_label('Z Value', rotation=270, labelpad=15)

# Show the plot
plt.show()

#####combine the two cut point clouds into one DataFrame
# Combine the two cut point clouds into one DataFrame
pd_scan_combined = pd.concat([pd_scan_vertical_cut, pd_scan_horizontal_cut], ignore_index=True)

#######visualise the combined point cloud in polar coordinates
# Downsample the combined point cloud for visualization
pd_scan_combined_ds = pd_scan_combined.sample(n=800000, random_state=42)

# Create a figure with a polar subplot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, polar=True)

# Plot the points in polar coordinates
ax.scatter(pd_scan_combined_ds['AzimuthAngle'], pd_scan_combined_ds['PolarAngle'], 
           c=pd_scan_combined_ds['Z'], 
           cmap='viridis'
           , s=0.5)

# Set title
ax.set_title('3D Point Cloud in Polar Coordinates', va='bottom')

#plot the colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
cbar.set_label('Z Value', rotation=270, labelpad=15)

# Show the plot
plt.show()