#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Autum 2025, last updated June 2026

This script is a collection of methods used for processing TLS point cloud data.
It contains functions for loading .laz files, transforming point cloud data to polar coordinates, 
visualizing the data, cutting the point cloud based on tilt, saving the processed data, and downsampling the point cloud.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)

"""

import os
import numpy as np
import pandas as pd
import laspy
import matplotlib.pyplot as plt
import seaborn as sns
import open3d as o3d
from typing import Tuple, Dict

from yaml import scan


def load_laz_file(file_path: str, devReflFiltering: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load a .laz file and return the point cloud data as a numpy array and a pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the .laz file.

    Returns:
    np.ndarray: Point cloud data as a numpy array.
    """
    with laspy.open(file_path) as f:
        print(f"Loading file: {file_path}")
        scan = f.read()
        #print(f"Header info: {scan.header}")
        print(f"Number of points: {scan.header.point_count}")
        
        # get filename and extension
        file_name, file_extension = file_path.split('/')[-1].split('.')
        #print(f"File name: {file_name}, File extension: {file_extension}")
    
        #print the dimensions of the point cloud
        dimensions = []
        dimensions = list(scan.point_format.dimension_names)
        print(f"Dimensions of the point cloud: dimension_names: {dimensions}")



        #check the offset of the point cloud data and print it
        #print(f"Offset of X: {scan.x.offset}, Offset of Y: {scan.y.offset}, Offset of Z: {scan.z.offset}")
        print(f"Scale of X: {scan.x.scale}, Scale of Y: {scan.y.scale}, Scale of Z: {scan.z.scale}")
        
        
        
        #check if there are any NaN values in the point cloud data if they are not NaN, then the point cloud is valid
        # raise an error if there are NaN values
        #if np.isnan(scan.x).any() or np.isnan(scan.y).any() or np.isnan(scan.z).any():
            # Print whether there are NaN values in the point cloud data
            #print("Checking for NaN values in the point cloud data:")
            #print(f"NaN in X: {np.isnan(scan.x).any()}")
            #print(f"NaN in Y: {np.isnan(scan.y).any()}")
            #print(f"NaN in Z: {np.isnan(scan.z).any()}")
            #raise ValueError("Point cloud data contains NaN values. Please check the input file.")
        #else:
            #print("Point cloud data is valid, no NaN values found.")
        
        
        # caclulate mean of classification if it exists
        if 'classification' in dimensions:
            mean_class = np.mean(scan.classification)
            #print(f"Mean classification: {mean_class}")
        
        
        #numpy array of points # check for dimensions of the point cloud data and make a scan.dimensionX a numpy array
        # Create a numpy array from the point cloud data
        if 'classification' in dimensions and mean_class > 0:
            points = np.vstack((scan.x, scan.y, scan.z,
                                scan.intensity,
                                scan.return_number,
                                scan.number_of_returns,
                                scan.classification)).T
            
            pd_points = pd.DataFrame(points, columns=['X', 'Y', 'Z', 
                                                      'Intensity', 'ReturnNumber', 'NumberOfReturns', 
                                                      'Classification'])
        else:
            # If classification is not present, create a numpy array without it
            # This is useful for point clouds that do not have classification data
            points = np.vstack((scan.x, scan.y, scan.z,
                                scan.intensity,
                                scan.return_number,
                                scan.number_of_returns)).T
            
            pd_points = pd.DataFrame(points, columns=['X', 'Y', 'Z', 
                                                      'Intensity', 'ReturnNumber', 'NumberOfReturns'])

        if devReflFiltering == True:
            # take the Deviation attribute if it exists
            if 'Deviation' in dimensions and 'Reflectance' in dimensions:
                Deviation = np.asarray(scan.Deviation)
                #print(f"Mean Deviation: {np.mean(Deviation)}")
                #print(f"Max Deviation: {np.max(Deviation)}")
                #print(f"Min Deviation: {np.min(Deviation)}")
                
                Reflectance = np.asarray(scan.Reflectance)
                #print(f"Mean Reflectance: {np.mean(Reflectance)}")
                #print(f"Max Reflectance: {np.max(Reflectance)}")
                #print(f"Min Reflectance: {np.min(Reflectance)}")

                # points np stack
                points = np.vstack((scan.x, scan.y, scan.z,
                                    scan.intensity,
                                    scan.return_number,
                                    scan.number_of_returns,
                                    scan.Deviation,
                                    scan.Reflectance)).T
                # add the Deviation to the pandas dataframe
                pd_points['Deviation'] = Deviation
                pd_points['Reflectance'] = Reflectance
                # filter: to do in main script to be in control of the settings  
            else:
                print("Deviation or Reflectance attribute not found in the point cloud data.")
        else: 
            print("no Deviation filtering applied")
        
        #print points
        if points is None:         
            raise ValueError("The numpy array does not exist. Loading TLS cloud did not work. Please check the input file.")
        
        
    
        # Print the first few points for verification
        print("First few points:")
        print(points[:10])
        
        #print empty lines for better readability
        print("\n" * 5)

        return points, pd_points

def transform_to_polar(points: np.ndarray) -> pd.DataFrame:
    """
    Transform Cartesian coordinates to polar coordinates.
    
    Parameters:
    points (np.ndarray): Point cloud data in Cartesian coordinates.

    Returns:
    pd.DataFrame: Point cloud data in polar coordinates.
    """
    # print
    print("Transforming points to polar coordinates...")
    
    # Convert the point clouds to pandas DataFrame
    # check how many columns are in the points array
    if points.shape[1] < 6:
        print("Warning: The points array has less than 6 columns. It may not contain all necessary attributes.")
        pd_points = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
    elif points.shape[1] == 6:
        pd_points = pd.DataFrame(points, columns=['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns'])
    elif points.shape[1] == 7:
        pd_points = pd.DataFrame(points, columns=['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns', 'Classification'])
    else:
        raise ValueError(f"Unexpected number of columns in points array: {points.shape[1]}. Expected 6 or 7 columns.")

    # Print the first few rows of the DataFrame
    print("First few rows of the point cloud data:")
    print(pd_points.head())
    
    # Filter points based distance from the origin < 50
    #pd_points['distance_center'] = np.sqrt(pd_points['X']**2 + pd_points['Y']**2)
    #pd_points = pd_points[pd_points['distance_center'] < 50]
    
    # Filter points based on the Z coordinate
    #pd_points = pd_points[pd_points['Z'] > -10]
    #pd_points = pd_points[pd_points['Z'] < 100]
    

    #make the point clouds to polar coordinates
    # add columns for azimut and range and polar angle
    pd_points['AzimuthAngle'] = np.arctan2(pd_points['Y'], pd_points['X'])
    pd_points['Range'] = np.sqrt(pd_points['X']**2 + pd_points['Y']**2 + pd_points['Z']**2)
    pd_points['PolarAngle'] = np.arccos(pd_points['Z'] / pd_points['Range'])

    # Convert angles from radians to degrees
    #pd_points['AzimuthAngle'] = np.degrees(pd_points['AzimuthAngle'])
    #pd_points['PolarAngle'] = np.degrees(pd_points['PolarAngle'])

    print("Polar coordinates:")
    print(pd_points.head())
    
    #print empty lines for better readability
    print("\n" * 5)
    

    return pd_points

def visualize_transformed_points(pd_points: pd.DataFrame, sample_size: int = 40000) -> None:
    """
    Visualize the transformed point cloud data.
    
    Parameters:
    pd_points (pd.DataFrame): Point cloud data in polar coordinates.
    """
    # print
    print("Visualizing transformed points...")
    
    # Sample the point cloud data to reduce the number of points for visualization
    pd_points = pd_points.sample(n=sample_size, random_state=42)
    
    # Visualize the vertical scan point cloud in polar coordinates
    # Create a figure with subplots for vertical cut point cloud
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)       
    # Plot the points in polar coordinates
    ax.scatter(pd_points['AzimuthAngle'], pd_points['PolarAngle'], 
            c=pd_points['Z'], cmap='viridis',
            s=0.5) 
    # Set title
    ax.set_title('3D Point Cloud in Polar Coordinates', va='bottom')    
    # Plot the colorbar
    cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
    cbar.set_label('Z Value', rotation=270, labelpad=15)

    # Show the plot
    plt.show()

def cut_point_cloud(pd_points: pd.DataFrame, tilt: str) -> pd.DataFrame:
    """
    Cut the point cloud data based on tilt
    
    Parameters:
    pd_points (pd.DataFrame): Point cloud data in polar coordinates.
    tilt (str): Tilt angle position description.

    Returns:
    pd.DataFrame: Cut point cloud data.
    """
    # print
    print(f"Cutting point cloud data for tilt: {tilt}...")
    
    # Define the cut range based on tilt and position
    if tilt == 'default':
        # error if tilt is not defined
        raise ValueError("Tilt angle must be specified.")
    elif tilt == 'vertical':
        pd_points_cut = pd_points[pd_points['PolarAngle'] > 0.01]
    elif tilt == 'horizontal':
        pd_points_cut = pd_points[pd_points['PolarAngle'] < 0.015]
    else:
        raise ValueError(f"Unknown tilt angle: {tilt}. Please specify 'vertical' or 'horizontal'.")

    print(f"Cut point cloud data shape: {pd_points_cut.shape}")
    
    return pd_points_cut

import numpy as np
import laspy
import pandas as pd
import os

def save_laz_file(file_path: str, pd_points: pd.DataFrame) -> None:
    """
    Save the point cloud data to a .laz file.
    """
    print(f"Saving point cloud data to {file_path}...")
    scale = 0.00025  # use your desired precision

    # Determine which columns exist
    cols = pd_points.columns

    # Build numpy array based on available columns
    base_cols = ['X', 'Y', 'Z']
    if all(c in cols for c in ['Intensity', 'ReturnNumber', 'NumberOfReturns', 'Classification']):
        mode = "full"
        points = pd_points[['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns', 'Classification']].to_numpy()
        point_format = 7
    elif all(c in cols for c in ['Intensity', 'ReturnNumber', 'NumberOfReturns']):
        mode = "standard"
        points = pd_points[['X', 'Y', 'Z', 'Intensity', 'ReturnNumber', 'NumberOfReturns']].to_numpy()
        point_format = 6
    else:
        mode = "xyz_only"
        points = pd_points[['X', 'Y', 'Z']].to_numpy()
        point_format = 6

    # --- Create header *before* creating LasData ---
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([scale, scale, scale])
    header.offsets = np.floor(points[:, :3].min(axis=0))

    # --- Create LasData AFTER setting scales/offsets ---
    las = laspy.LasData(header)

    # --- Assign coordinates ---
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # --- Optional fields ---
    if mode != "xyz_only":
        las.intensity = points[:, 3].astype(np.float32)
        las.return_number = points[:, 4].astype(np.uint8)
        las.number_of_returns = points[:, 5].astype(np.uint8)
    if mode == "full":
        las.classification = points[:, 6].astype(np.uint8)

    # --- Write ---
    las.write(file_path)

    # --- Verification ---
    verify = laspy.read(file_path)
    print(f"✅ Saved {mode} file.")
    print(f"Header scales: {verify.header.scales}")
    print(f"Header offsets: {verify.header.offsets}")
    print(f"Saved point count: {len(verify.points)}\n\n")

def density_plot(points: np.ndarray, save: bool, output_name: str) -> None:
    """
    Create a density plot of the Z values of the point cloud data.
    
    Parameters:
    points (np.ndarray): Point cloud data in polar coordinates.
    save (bool): Whether to save the plot as a file.
    """
    # Create a density plot of the Z values
    z_values = points[:, 2] # Extract Z values from the point cloud data as 3rd column

    #filter points based on Z value greater than 0
    z_values = z_values[z_values > 0]


    # Create a density plot
    sns.kdeplot(y=z_values, 
                bw_adjust=1, bw_method='scott',
                fill=False, color='black',
                alpha=0.8, linewidth=1, label='Z values'
                #cut=0, clip=[0, 50]
                )
    plt.xlabel('Density')
    plt.ylabel('Z Value')
    plt.title('Density plot of Z values')
    #gridlines drawn
    plt.grid(True)
    if save == True:
        # Save the plot to the current directory
        plt.savefig(output_name)
        print(f"Density plot saved as {output_name}")
    # Show the plot
    plt.show()

def visualizing_pcd(pd_points: pd.DataFrame, voxel_size: float = 0.1) -> None:
    """
    Visualize the point cloud data in Open3D.
    
    Parameters:
    pd_points (pd.DataFrame): Point cloud data in polar coordinates.
    voxel_size (float): Voxel size for downsampling the point cloud. higher values result in fewer points. 
    default is 0.1.
    """
    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    
    # Set the points of the point cloud
    pcd.points = o3d.utility.Vector3dVector(pd_points[['X', 'Y', 'Z']].values)
    
    # Downsample the point cloud to reduce the number of points
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Visualize the downsampled point cloud
    o3d.visualization.draw_geometries([downpcd])

def visualize_xy_z(np_points: np.ndarray, sample_size: int = 100_000, save: bool = False, path: str = os.getcwd) -> None:
    """
    Visualize the XY plane of the point cloud data. colours the points based on their Z values.
    """
    sample_indices = np.random.choice(np_points.shape[0], size=sample_size, replace=False)    
    # Create a random sample of the point cloud
    points_sample = np_points[sample_indices]
    # Create a 2D scatter plot of the point cloud # Create a 2D scatter plot of the point cloud 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    # Create a scatter plot of the point cloud
    ax.scatter(points_sample[:, 0], points_sample[:, 1], 
            c=points_sample[:, 2], 
            cmap='viridis', 
            s=0.5)
    # Add a color bar
    cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
    cbar.set_label('Z axis')
    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # set gridlines
    ax.grid(True) 
    # define the distance between the gridlines
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    # set the title
    ax.set_title('Point Cloud in XY Plane') # Set the title
    plt.axis('equal')  # Set the aspect ratio to equal
    if save == True:
        # Save the plot to the current directory
        plt.savefig(f'path')
        print("XY plane plot saved as xy_plane.png")
    plt.show() 



def downsample_pointcloud(pd_points: pd.DataFrame, voxel_size: float, save: bool, output_path: str) -> pd.DataFrame:
    """
    Downsample the point cloud data using voxel downsampling.
    
    Parameters:
    pd_points (pd.DataFrame): Point cloud data in polar coordinates.
    voxel_size (float): Voxel size for downsampling the point cloud. higher values result in fewer points.
    save (bool): Whether to save the downsampled point cloud as a file.
    output_path (str): Name of the output file to save the downsampled point cloud.

    Returns:
    pd.DataFrame: Downsampled point cloud data.
    """
    # Convert to numpy arrays for the tensors
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
    pcd.point.positions = point_set
    pcd.point.intensity = intensity.reshape(-1, 1)  # Reshape to ensure correct dimensions
    pcd.point.return_number = return_number.reshape(-1, 1)  # Reshape to ensure correct dimensions
    pcd.point.number_of_returns = number_of_returns.reshape(-1, 1)  # Reshape to ensure correct dimensions

    if classification is not None:
        pcd.point.classification = classification.reshape(-1, 1)  # Reshape to ensure correct dimensions    

    # Print the point cloud to verify
    print(pcd, "\n")


    # Downsample the point cloud to reduce the number of points
    print(f"Downsampling point cloud with voxel size: {voxel_size} m")
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Visualize the downsampled point cloud
    #print("starting to visualize the downsampled point cloud...")
    #o3d.visualization.draw_geometries([downpcd.to_legacy()])


    #get the attributes of the downsampled point cloud from the object downpcd
    print("Preparing to save the downsampled point cloud to a new file...")
    positions = downpcd.point.positions
    intensity = downpcd.point.intensity
    return_number = downpcd.point.return_number
    number_of_returns = downpcd.point.number_of_returns
    if classification is not None:
        classification = downpcd.point.classification

    #turn tensors into numpy arrays
    positions = positions.numpy()
    intensity = intensity.numpy().flatten()  # Flatten to ensure correct shape
    return_number = return_number.numpy().flatten()  # Flatten to ensure correct shape
    number_of_returns = number_of_returns.numpy().flatten()  # Flatten to ensure correct shape
    if classification is not None:
        classification = classification.numpy().flatten()  # Flatten to ensure correct shape

    # Create a pandas DataFrame from the downsampled point cloud numpy arrays
    downsampled_pd_points = pd.DataFrame({ 
        'X': positions[:, 0],
        'Y': positions[:, 1],
        'Z': positions[:, 2],
        'Intensity': intensity,
        'ReturnNumber': return_number,
        'NumberOfReturns': number_of_returns
    })
    if classification is not None:
        downsampled_pd_points['Classification'] = classification

    # Print the first few rows of the downsampled point cloud
    print("Downsampling done. Here are the first few rows of the downsampled point cloud:")
    print(downsampled_pd_points.head()) 
    
    if save == True:
        # Save the downsampled point cloud to a new file
        save_laz_file(output_path, downsampled_pd_points)
        print(f"Downsampled point cloud saved as {output_path}")

    return downsampled_pd_points


def load_sop_backup_csv(square, main_path) -> pd.DataFrame:
    """
    Load the SOP backup CSV file for the given square.

    Parameters:
    square (str): The name of the square area being processed.
    main_path_disk (str): The main directory path on the disk where data is stored.

    Returns:
    pd.DataFrame: DataFrame containing the SOP backup data.
    """
    sop_backup_path = f"{main_path}/{square}_SOP_Backup.csv"
    if os.path.exists(sop_backup_path):
        # Load the SOP backup CSV file
        sop_backup_df = pd.read_csv(sop_backup_path)
        print(f"SOP backup CSV loaded from {sop_backup_path}")
        
        # only take scanPosName x y z 
        sop_backup_df = sop_backup_df[['scanPosName', 'x', 'y', 'z']]
        
        # rename columns
        sop_backup_df = sop_backup_df.rename(columns={'x': 'x_sop', 'y': 'y_sop', 'z': 'z_sop'})
                
        # print the sop_backup_df
        print("SOP Backup DataFrame:")
        print(sop_backup_df.head())
        
        return sop_backup_df
    else:
        raise FileNotFoundError(f"SOP backup CSV file not found at {sop_backup_path}")


# pack/unpack helpers (same approach as before)
def cells_to_key(cell_x: np.ndarray, cell_y: np.ndarray) -> np.ndarray:
    cx = cell_x.astype(np.int64)
    cy = cell_y.astype(np.int64)
    ux = (cx & 0xFFFFFFFF).astype(np.int64)
    uy = (cy & 0xFFFFFFFF).astype(np.int64)
    return (ux << 32) | uy

def key_to_cells(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ux = (keys >> 32).astype(np.int64) & 0xFFFFFFFF
    uy = (keys & 0xFFFFFFFF).astype(np.int64)
    cx = np.where(ux >= 2**31, ux - 2**32, ux).astype(np.int32)
    cy = np.where(uy >= 2**31, uy - 2**32, uy).astype(np.int32)
    return cx, cy

def create_point_dtype(coord_dtype=np.float32, include_cell_key=True):
    # Build a structured dtype for per-point storage
    fields = [
        ("x", coord_dtype),
        ("y", coord_dtype),
        ("z", coord_dtype),
        ("num_returns", np.uint8),
        ("return_num", np.uint8),
        ("classification", np.uint8),
    ]
    if include_cell_key:
        fields.append(("cell_key", np.int64))  # 64-bit packed (cell_x,cell_y)
    return np.dtype(fields)



"""
Stream a LAZ/LAS and create a per-point structured array with:
  x, y, z, number_of_returns, return_number, classification, cell_key

This simplified version NO LONGER performs any per-cell aggregation.
Per-cell aggregation/grouping should be done later by grouping the per-point
array on "cell_key" (e.g. using numpy, pandas, or dask).

- Uses laspy.chunk_iterator to avoid loading all points into RAM.
- Stores per-point results into a numpy.memmap file (recommended for very large clouds).
- Default coordinate dtype is float32 (saves space). Use float64 if you need full precision.
- Uses laspy points.x/y/z (already scaled). If you prefer raw integer fields, convert using header scales/offsets.
"""
from typing import Optional, Tuple, Dict
import laspy
import numpy as np
import math
import os
from definitions_a import create_point_dtype, cells_to_key, key_to_cells




def assign_pointdata_with_cellkey(
    laz_path: str,
    memmap_path: Optional[str],
    cell_size: float = 1.0,
    chunk_size: int = 5_000_000,
    coord_dtype=np.float32, 
) -> Dict[str, object]:
    """
    Simplified: Stream LAS/LAZ and create a per-point structured array.
    NO per-cell aggregation is done here — only per-point fields and the packed cell_key.

    Parameters
    ----------
    laz_path : str
        Input .laz/.las file.
    memmap_path : Optional[str]
        Path for the memmap file to hold the per-point structured array.
        If None, function will allocate the full array in memory and return it.
        For large clouds (150M points) a memmap is strongly recommended.
    cell_size : float
        Cell edge length in same units as coordinates (meters).
    chunk_size : int
        Number of points to read per laspy chunk.
    coord_dtype : numpy dtype (np.float32 or np.float64)
        Stored dtype for coordinates in the output array.

    Returns
    -------
    dict with keys:
      - "per_point": numpy.memmap or numpy.ndarray (structured) with per-point fields
      - "memmap_path": memmap_path (or None if in-memory)
      - "total_points": int
      - "dtype": the numpy dtype used for the structured array
    """
    
    # pack/unpack helpers (same approach as before)
    def cells_to_key(cell_x: np.ndarray, cell_y: np.ndarray) -> np.ndarray:
        cx = cell_x.astype(np.int64)
        cy = cell_y.astype(np.int64)
        ux = (cx & 0xFFFFFFFF).astype(np.int64)
        uy = (cy & 0xFFFFFFFF).astype(np.int64)
        return (ux << 32) | uy

    def key_to_cells(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ux = (keys >> 32).astype(np.int64) & 0xFFFFFFFF
        uy = (keys & 0xFFFFFFFF).astype(np.int64)
        cx = np.where(ux >= 2**31, ux - 2**32, ux).astype(np.int32)
        cy = np.where(uy >= 2**31, uy - 2**32, uy).astype(np.int32)
        return cx, cy

    def create_point_dtype(coord_dtype=np.float32, include_cell_key=True):
        # Build a structured dtype for per-point storage
        fields = [
            ("x", coord_dtype),
            ("y", coord_dtype),
            ("z", coord_dtype),
            ("intensity", np.float32),
            ("num_of_returns", np.uint8),
            ("return_number", np.uint8),
            ("classification", np.uint8),
        ]
        if include_cell_key:
            fields.append(("cell_key", np.int64))  # 64-bit packed (cell_x,cell_y)
        return np.dtype(fields)



    dtype = create_point_dtype(coord_dtype)

    with laspy.open(laz_path) as fh:
        total_points = int(fh.header.point_count)
        print(f"Total points: {total_points:,}")
        n_chunks = math.ceil(total_points / chunk_size)
        print(f"Processing in {n_chunks} chunks, chunk_size={chunk_size:,}")

        # prepare per-point storage (memmap or in-memory)
        use_memmap = memmap_path is not None
        if use_memmap:
            # remove existing file to ensure clean memmap
            if os.path.exists(memmap_path):
                os.remove(memmap_path)
            per_point = np.memmap(memmap_path, mode="w+", dtype=dtype, shape=(total_points,))
            print(f"Created memmap: {memmap_path}")
        else:
            per_point = np.empty((total_points,), dtype=dtype)
            print("Allocated in-memory per-point array")

        offset = 0

        for chunk_idx, points in enumerate(fh.chunk_iterator(chunk_size)):
            # use laspy float accessors (already scaled)
            x = points.x
            y = points.y
            # keep as numpy arrays (recommended for numeric work / writing back)
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            z = np.asarray(points.z)

            # compute cell indices (we don't map to global ids here)
            cell_x = np.floor(x / cell_size).astype(np.int32)
            cell_y = np.floor(y / cell_size).astype(np.int32)

            # pack cell keys
            keys = cells_to_key(cell_x, cell_y)

            # write per-point structured data for this chunk
            n_here = x.shape[0]
            end = offset + n_here
            per_point_slice = per_point[offset:end]

            # cast coordinates to storage dtype 
            per_point_slice["x"] = x.astype(coord_dtype)
            per_point_slice["y"] = y.astype(coord_dtype)
            per_point_slice["z"] = z.astype(coord_dtype)

            # laspy properties: return_number, number_of_returns, classification
            intensity = np.asarray(points.intensity, dtype=np.float32)
            rn = np.asarray(points.return_number, dtype=np.uint8)
            nr = np.asarray(points.number_of_returns, dtype=np.uint8)
            cl = np.asarray(points.classification, dtype=np.uint8)

            per_point_slice["intensity"] = intensity
            per_point_slice["num_of_returns"] = nr
            per_point_slice["return_number"] = rn
            per_point_slice["classification"] = cl

            per_point_slice["cell_key"] = keys.astype(np.int64)

            offset = end
            print(f"Chunk {chunk_idx+1}/{n_chunks}: pts={n_here:,}, offset={offset:,}")

        # sanity check
        if offset != total_points:
            raise RuntimeError(f"Written points ({offset}) != total points ({total_points})")

        if use_memmap:
            per_point.flush()

        return {
            "per_point": per_point,
            "memmap_path": memmap_path if use_memmap else None,
            "total_points": total_points,
            "dtype": dtype,
        }





def load_all_general_weather_statistics(main_weather_path, squares_list_in_correct_order: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        main_weather_path (_type_): _description_
        squares_list_in_correct_order (list): _description_

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: _description_
        1. general_stats_hourly, 
        2. general_stats_wholeTime
        3. general_stats_hourly_groupedByHour
        4. general_stats_hourly_groupedByDay
        
    """
    squares = squares_list_in_correct_order
    #Load general weather statistics if possible withUTCI
    if not os.path.exists(f"{main_weather_path}/All_Squares_Weather_Hourly_Statistics_WithUTCI.csv"):
        general_stats_hourly = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_Hourly_Statistics.csv")
        general_stats_wholeTime = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_WholeTime_Statistics.csv")
        general_stats_hourly_groupedByHour = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_Hourly_Mean_Statistics.csv") #diurnal
        general_stats_hourly_groupedByDay = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_Daily_Mean_Statistics.csv")
    else: 
        general_stats_hourly = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_Hourly_Statistics_WithUTCI.csv")
        general_stats_wholeTime = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_WholeTime_Statistics_WithUTCI.csv")
        general_stats_hourly_groupedByHour = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_Hourly_Mean_Statistics_WithUTCI.csv") #diurnal
        general_stats_hourly_groupedByDay = pd.read_csv(f"{main_weather_path}/All_Squares_Weather_Daily_Mean_Statistics_WithUTCI.csv")


    #SORT all the dataframes by square name via the order in squares list
    general_stats_hourly['Square'] = pd.Categorical(general_stats_hourly['Square'], categories=squares, ordered=True)
    general_stats_hourly = general_stats_hourly.sort_values(by=['Square','Date'])
    general_stats_wholeTime['Square'] = pd.Categorical(general_stats_wholeTime['Square'], categories=squares, ordered=True)
    general_stats_wholeTime = general_stats_wholeTime.sort_values(by=['Square'])
    general_stats_hourly_groupedByHour['Square'] = pd.Categorical(general_stats_hourly_groupedByHour['Square'], categories=squares, ordered=True)
    general_stats_hourly_groupedByHour = general_stats_hourly_groupedByHour.sort_values(by=['Square','Hour'])
    general_stats_hourly_groupedByDay['Square'] = pd.Categorical(general_stats_hourly_groupedByDay['Square'], categories=squares, ordered=True)
    general_stats_hourly_groupedByDay = general_stats_hourly_groupedByDay.sort_values(by=['Square','Date'])


    print("Loaded general weather statistics:")
    print("General Stats Hourly:")
    print(general_stats_hourly.head())
    print(general_stats_hourly.columns.tolist())
    print("\n")
    print("General Stats Whole Time:")
    print(general_stats_wholeTime.head())
    print(general_stats_wholeTime.columns.tolist())
    print("\n")
    print("General Stats Hourly Grouped By Hour (one row per hour agg all 5 pm slots of a square):")
    print(general_stats_hourly_groupedByHour.head())
    print(general_stats_hourly_groupedByHour.columns.tolist())
    print("\n")
    print("General Stats Hourly Grouped By Day (one row per day agg all slots per day of a square):")
    print(general_stats_hourly_groupedByDay.head())
    print(general_stats_hourly_groupedByDay.columns.tolist())
    
    return general_stats_hourly, general_stats_wholeTime, general_stats_hourly_groupedByHour, general_stats_hourly_groupedByDay