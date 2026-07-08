#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 
This script contains functions for plotting the point cloud data. 
It is used for visualizing the point cloud data in a grid from different views (XY, XZ, YZ) and also in 3D.
The script loads the point cloud data from .laz files, processes it, and generates scatter plots for each square, showing the point cloud data in different planes and in 3D.

Author: Antonia Hostlowsky (assisted by ChatGPT, CoPILOT)
Originally created April 2026, last updated July 2026
"""


import pandas as pd
import seaborn as sns
import os
import laspy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.cluster import DBSCAN
from scipy.optimize import leastsq

import math
import csv
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Dict

from StyleMaps import *
from definitions_a import *


################ HELPER FUNCTIONS ####################
def load_pointcloud_data(laz_path):
    """Load point cloud data from a .laz file and assign cell keys based on the specified cell size. The function also creates a memmap file for efficient access to the point data.

    Args:
        laz_path (str): Path to the .laz file containing the point cloud data.

    Returns:
        tuple: A tuple containing the pandas DataFrame and the numpy array of the point cloud data.
    """

    # eg. laz_path = f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'
    #memmap_path = f'{laz_path}" but with .memmap extension instead of .laz'  # e.g. f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.memmap'
    memmap_path = laz_path.replace(".laz", ".memmap")

    result = assign_pointdata_with_cellkey(
        laz_path=laz_path,
        memmap_path=memmap_path,
        cell_size=1,
        chunk_size=5_000_000,
        coord_dtype=np.float32,
    )

    #dtype = create_point_dtype(np.float32, True)  # from your writer
    #per_point = np.memmap("mycloud_pointdata.memmap", mode="r", dtype=dtype, shape=(total_points,))


    arr = result["per_point"] 
    df = pd.DataFrame({
        "cell_key": arr["cell_key"],
        "X": arr["x"].astype(np.float64),
        "Y": arr["y"].astype(np.float64),
        "Z": arr["z"].astype(np.float64),
        "Intensity": arr["intensity"].astype(np.float32),
        #"Number_of_Returns": arr["num_of_returns"].astype(np.uint8),
        #"Return_Number": arr["return_number"].astype(np.uint8),
        "Classification": arr["classification"].astype(np.int32)
        })
    print(df.head())
    print(f"Total points loaded: {len(df)}")

    # To convert cell_key back to cell_x/cell_y:
    keys = df["cell_key"].to_numpy(dtype=np.int64)
    cx, cy = key_to_cells(keys)

    pd_points = df.copy()
    pd_points["cell_x"] = cx
    pd_points["cell_y"] = cy

    print(pd_points.head())
    
    #make into numpy array for plotting
    np_points = pd_points[["X", "Y", "Z", "Intensity", "Classification"]].to_numpy(dtype=np.float32)
    
    return pd_points , np_points


def plot_pcstuff(pd_points, np_points, main_path, square):
    """ This function creates a horizontal figure with three subplots showing the point cloud data in different planes (XY, XZ, YZ) colored by Z value and classification. 
    2D plots scatters of the planes
    It saves the figure to the specified path.
    
    Args:
       pd_points (pd.DataFrame): DataFrame containing the point cloud data with columns "X", "Y", "Z", "Intensity", "Classification".
       np_points (np.ndarray): Numpy array containing the point cloud data with columns "X", "Y", "Z", "Intensity", "Classification".
       main_path (str): The main path where the figure will be saved.    
       square (str): The name of the square being plotted, used for naming the output file.
    Returns:
       None: The function saves the figure to the specified path and does not return anything.
    
    """
    
    metrics_folder = f"{main_path}/pc_plots"
    os.makedirs(metrics_folder, exist_ok=True)



    sample_indices = np.random.choice(np_points.shape[0], size=300_000, replace=False)
    points_sample = np_points[sample_indices]

    # Create a horizontal figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(24, 6))

    ###### XY plane (left subplot)
    ax = axes[0]
    sc = ax.scatter(points_sample[:, 0], points_sample[:, 1],
                    c=points_sample[:, 2],
                    cmap='turbo',
                    s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title('Point Cloud in XY Plane')
    #ax.set_aspect('equal', adjustable='datalim')
    # Add colorbar to the right
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Z Value')

    ###### XZ plane (middle subplot)
    ax = axes[1]
    sc = ax.scatter(points_sample[:, 0], points_sample[:, 2],
                    c=points_sample[:, 4],
                    cmap=classification_colors_pointcloud_cmap,
                    s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-5, 30)
    ax.set_title('Point Cloud in XZ Plane')
    ax.set_aspect('equal')
    # Add colorbar to the right
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Classification')

    ###### YZ plane (right subplot)
    ax = axes[2]
    sc = ax.scatter(points_sample[:, 1], points_sample[:, 2],
                    c=points_sample[:, 4],
                    cmap=classification_colors_pointcloud_cmap,
                    s=0.5)
    ax.set_xlabel('Y')
    ax.set_ylabel('Z')
    ax.grid(True)
    ax.xaxis.set_major_locator(plt.MultipleLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(10))
    ax.set_xlim(-50, 50)
    ax.set_ylim(-5, 30)
    ax.set_title('Point Cloud in YZ Plane')
    ax.set_aspect('equal')
    # Add colorbar to the right
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Classification')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig(f"{metrics_folder}/{square}_planes_horizontal.png", bbox_inches='tight', dpi=200)
    print(f"Plane plot saved as {square}_planes_horizontal.png")
    #plt.show()
    plt.close()


def plot_pcstuff_3d(np_points_orig: np.ndarray, np_points_manual: np.ndarray, np_point_final: np.ndarray, main_path: str, square: str):
    """
    This function creates a horizontal figure with three subplots showing the point cloud data in different planes (3D view, XZ slice, YZ slice) colored by classification.
    3D view 
    It saves the figure to the specified path.

    Args:
        np_points_orig (np.ndarray): Numpy array containing the original point cloud data with columns "X", "Y", "Z", "Intensity", "Classification".
        np_points_manual (np.ndarray): Numpy array containing the manually cleaned point cloud data with columns "X", "Y", "Z", "Intensity", "Classification".
        np_point_final (np.ndarray): Numpy array containing the final cleaned point cloud data with columns "X", "Y", "Z", "Intensity", "Classification".
        main_path (str): The main path where the figure will be saved.    
        square (str): The name of the square being plotted, used for naming the output file.
    """
    metrics_folder = f"{main_path}/pc_plots"
    os.makedirs(metrics_folder, exist_ok=True)
    
    #use for all three clouds the same random sample of points for better comparison - eventhough the points are not there in the last cloud ... 
    sample_indices = np.random.choice(np_points_orig.shape[0], size=300_000, replace=False)
    points_sample_orig = np_points_orig[sample_indices]

    points_sample_manual = np_points_manual[sample_indices]

    points_sample_final = np_point_final[sample_indices]

    # Create a horizontal figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw=dict(projection='3d'))


    ###### XYZ 3D view (left subplot)
    ax = axes[0]
    sc = ax.scatter(points_sample_orig[:, 0], points_sample_orig[:, 1], points_sample_orig[:, 2],
                    c=points_sample_orig[:, 4],
                    cmap="turbo",
                    s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D View (Colored by Classification)')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_zlim(-5, 30)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Classification')

    ###### XZ 3D slice (middle subplot)
    ax = axes[1]
    sc = ax.scatter(points_sample_orig[:, 0], points_sample_orig[:, 1], points_sample_orig[:, 2],
                    c=points_sample_manual[:, 4],
                    cmap=classification_colors_pointcloud_cmap,
                    s=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('')  # Empty for this slice
    ax.set_zlabel('Z')
    ax.set_title('XZ Slice (Colored by Classification)')
    ax.set_xlim(-50, 50)
    ax.set_zlim(-5, 30)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Classification')

    ###### YZ 3D slice (right subplot)
    ax = axes[2]
    sc = ax.scatter(points_sample_orig[:, 0], points_sample_orig[:, 1], points_sample_orig[:, 2],
                    c=points_sample_final[:, 4],
                    cmap="tab10",
                    s=0.5)
    ax.set_xlabel('')  # Empty for this slice
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('YZ Slice (Colored by Classification)')
    ax.set_ylim(-50, 50)
    ax.set_zlim(-5, 30)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Classification')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{metrics_folder}/{square}_3d_planes_horizontal.png", bbox_inches='tight', dpi=200)
    plt.savefig(f"{metrics_folder}/{square}_3d_planes_horizontal.pdf", bbox_inches='tight')
    print(f"3D plots saved as {square}_3d_planes_horizontal.png and .pdf")
    plt.close()


################ MAIN FUNCTIONS ####################
def main(square_list: list, main_path: str):
    """
    Main function to load the point cloud data and create the figures for each square. 
    It iterates through a list of squares, loads the corresponding point cloud data, and calls the plotting functions to visualize the data. 
    
    Args:
        square_list (list): List of square names to process.
        main_path (str): The main path where the point cloud data and output figures are located
    Returns:
        None: The function saves the figures to the specified path and does not return anything.
    """
    

    for square in square_list:
        laz_path = f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'
        pd_points, np_points = load_pointcloud_data(laz_path)
        plot_pcstuff(pd_points, np_points, main_path, square)
        plot_pcstuff_3d(np_points, np_points, np_points, main_path, square) 
    



if __name__ == "__main__":
    main_path = "/Volumes/T7_Shield_A/msc/"
    square_list = ["rundfunk", "nikolai", "elisabeth", "miesbacher", "jakobgelb", "konigsplatz"]

    main(square_list, main_path)