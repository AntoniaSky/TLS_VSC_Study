#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Dec 2025, last updated July 2026

This script calculates various vegetation structure metrics from the processed TLS Point Cloud Data based on the Height-Attribute (Z) values of the point cloud data. 
Height is normalized height above ground of the vegetation. 

The metrics include 
- diversity metrics (Shannon, Gini-Simpson), --> similar to ENL
- descriptive statistics (mean, median, std, stdstd, var, min, max, range, 99th percentile), 
- proportions of points above certain height thresholds (2m, 5m, 10m).

The script saves the results in a CSV file for each square and also updates a general results CSV file with the metrics for all squares. 
Additionally, it generates a 3D histogram of the point cloud data and saves it as a PNG file.

The script can be used after the preprocessing steps in PreprocessingPointClouds.py.

Sources for metrics: 
- Shokirov, S. et al. (2023). “Habitat highs and lows: Using terrestrial and UAV LiDAR for modelling avian species richness and abundance in a restored woodland”. In: Remote Sensing of Environment 285, p. 113326. DOI: 10.1016/j.rse.2022.113326.


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
"""

#### Import necessary libraries and files
import os
import laspy
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from definitions_a import load_laz_file


##### Main method for all Z-metrics calculation
def z_metrics(main_path: str, square: str):
    """ Calculate various vegetation structure metrics based on the Height-Attribute (Z) values of the point cloud data for a given square.
    The metrics include diversity metrics (Shannon, Gini-Simpson), descriptive statistics (mean, median, std, var, min, max, range, 99th percentile), and proportions of points above certain height thresholds (2m, 5m, 10m).
    
    Args:
        square (str): square identifier (e.g., 'alpenplatz', 'nikolai', etc.)
        main_path (str): path to the directory containing the final preprocessed TLS point cloud data
        
    Returns:
        None: Saves the results to a CSV file and generates plots for vegetation height distribution and diversity metrics.
    """
    
    print(f"START FUNCTION FOR Z METRICS CALCULATION.")
    print(f"Processing square: {square}")
    # Load the LAZ file
    print("Loading LAZ file...")
    laz, pd_laz = load_laz_file(f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz')
    
    # define folder and paths
    metrics_folder = f'{main_path}/{square}_Metrics'
    os.makedirs(metrics_folder, exist_ok=True)  
    results_txt_path = f'{main_path}/{square}_Metrics/{square}_Results.txt'
    
    results_csv_path = f'{main_path}/All_Results.csv'

    # read or create a csv file to store the results
    columns=['Square', 'n_heightlcass', 'Shannon', 'Gini-Simpson',
                                           'Mean_Z', 'Median_Z', 'Std_Z', 'Var_Z', 'Min_Z', 'Max_Z',
                                           'Range_Z', 'Q99_Z', 'Prop_Above_2m', 'Prop_Above_5m', 'Prop_Above_10m']
    if os.path.exists(results_csv_path):
        results_df = pd.read_csv(results_csv_path)
        # check if the columns exist, if not create them
        for col in columns:
            if col not in results_df.columns:
                results_df[col] = np.nan
    else:
        results_df = pd.DataFrame(columns=columns)

    ############################################
    #### histogram of the point cloud
    #visualize the point cloud with a diagram x axis and y axis
    # Create a 2D histogram of the point cloud
    hist, xedges, yedges = np.histogram2d(pd_laz['X'], pd_laz['Y'], bins=50)        
    # Create a meshgrid for the histogram
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    # Flatten the meshgrid  
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    # Flatten the histogram
    zpos = hist.ravel()
    # Construct arrays with the dimensions for the 16 bars.
    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 0.1 * np.ones_like(zpos)
    dz = zpos
    # Create a 3D bar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(xpos, ypos, 0, dx, dy, dz, zsort='average')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Histogram of Point Cloud')
    path = f'{main_path}/{square}_Metrics/{square}_3D_Histogram.png'
    plt.savefig(path)
    plt.close()


    ################################# Gini-Simpson and Shannon diversity metrics
    ########## Calculate diversity metrics 
    print("Calculating diversity metrics and Z statistics for all points...")
    ## # Calculate diversity metrics
    # Calculate diversity metrics
    print("Calculating diversity metrics...")
    # copy only XYZ columns of pd_laz
    scan_gt0m = pd_laz[['X', 'Y', 'Z']].copy()
    
    # Filter points based on Z value greater than 0
    scan_gt0m = scan_gt0m[scan_gt0m['Z'] > 0]

    # Create height bins
    scan_gt0m['height_bins'] = pd.cut(scan_gt0m['Z'], bins=np.arange(0, 41, 1))

    # Group by height bins and calculate the number of points in each bin
    n = scan_gt0m['height_bins'].value_counts()

    # Calculate the proportion of points in each bin
    p = n / n.sum()

    # Calculate diversity metrics
    n_heightlcass = np.power(np.sum(np.power(p, 0)), 1/(1-0))
    #give only the number of classes with non-zero values
    n_heightlcass = np.sum(n > 0)
    # Calculate the Shannon index
    shannon = np.exp(-np.sum(p * np.log(p)))
    # Calculate the Gini-Simpson index
    gini_simpson = np.power(np.sum(np.power(p, 2)), 1/(1-2))

    gini_simpson_dict = {
        'n_heightlcass': n_heightlcass,
        'Shannon': shannon,
        'Gini-Simpson': gini_simpson
    }
    print("Diversity Metrics:")
    print(gini_simpson_dict)

    ########## Calculate statistics for Z values
    # mean
    mean = scan_gt0m['Z'].mean()
    # median
    median = scan_gt0m['Z'].median()
    # standard deviation
    std = scan_gt0m['Z'].std()
    # variance
    var = scan_gt0m['Z'].var()
    # minimum
    min = scan_gt0m['Z'].min()
    # maximum
    max = scan_gt0m['Z'].max()
    # range
    range = max - min
    # 99th percentile
    q99 = scan_gt0m['Z'].quantile(0.99) 
    # proportion of points above 2
    prop_above_2 = np.sum(scan_gt0m['Z'] > 2) / scan_gt0m.shape[0]
    # proportion of points above 5
    prop_above_5 = np.sum(scan_gt0m['Z'] > 5) / scan_gt0m.shape[0]
    # proportion of points above 10
    prop_above_10 = np.sum(scan_gt0m['Z'] > 10) / scan_gt0m.shape[0]

    # print the statistics as in colums 
    stats_dict = {
        'Mean_Z': mean, 
        'Median_Z': median, 
        'Std_Z': std, 
        'Var_Z': var, 
        'Min_Z': min, 
        'Max_Z': max,
        'Range_Z': range, 
        'Q99_Z': q99, 
        'Prop_Above_2m': prop_above_2, 
        'Prop_Above_5m': prop_above_5, 
        'Prop_Above_10m': prop_above_10 
    }   
    print("Z Statistics:")
    print(stats_dict)
    
    #write to a text file if file exists append else create new file
    if not os.path.exists(results_txt_path):
        with open(results_txt_path, 'w') as f:
            f.write('Gini Simpson Metrics \n')
            f.write('============\n\n') # \n creates a new line #\n\n creates a new paragraph
            f.write(f'n_heightlcass: {n_heightlcass}\n')
            f.write(f'Shannon: {shannon}\n')
            f.write(f'Gini-Simpson: {gini_simpson}\n')
            f.write('\n')
            f.write('Z Statistics of whole Square \n')
            f.write('============\n\n') # \n creates a new line #\n\n creates a new paragraph
            f.write(f'Mean: {mean}\n')
            f.write(f'Median: {median}\n')                  
            f.write(f'Standard deviation: {std}\n')
            f.write(f'Variance: {var}\n')
            f.write(f'Minimum: {min}\n')
            f.write(f'Maximum: {max}\n')        
            f.write(f'Range: {range}\n')
            f.write(f'99th percentile: {q99}\n')
            f.write(f'Proportion of points above 2: {prop_above_2}\n')
            f.write(f'Proportion of points above 5: {prop_above_5}\n')
            f.write(f'Proportion of points above 10: {prop_above_10}\n')
            f.write('\n')
    else:      
        with open(results_txt_path, 'a') as f:
            f.write('Gini Simpson Metrics \n')
            f.write('============\n\n') # \n creates a new line #\n\n creates a new paragraph
            f.write(f'n_heightlcass: {n_heightlcass}\n')
            f.write(f'Shannon: {shannon}\n')
            f.write(f'Gini-Simpson: {gini_simpson}\n')
            f.write('\n')
            f.write('Z Statistics of whole Square \n')
            f.write('============\n\n') # \n creates a new line #\n\n creates a new paragraph
            f.write(f'Mean: {mean}\n')
            f.write(f'Median: {median}\n')                  
            f.write(f'Standard deviation: {std}\n')
            f.write(f'Variance: {var}\n')
            f.write(f'Minimum: {min}\n')
            f.write(f'Maximum: {max}\n')        
            f.write(f'Range: {range}\n')
            f.write(f'99th percentile: {q99}\n')
            f.write(f'Proportion of points above 2: {prop_above_2}\n')
            f.write(f'Proportion of points above 5: {prop_above_5}\n')
            f.write(f'Proportion of points above 10: {prop_above_10}\n')
            f.write('\n')
            
    # append the results to the dataframe
    new_row = {'Square': square,
               **gini_simpson_dict, 
               **stats_dict}
    results_df.loc[len(results_df)] = new_row 
    #save the dataframe to csv
    results_df.to_csv(results_csv_path, index=False)
    
    
    
    ########################### Calculate same metrics for vegetation points only
    print("Calculating metrics for vegetation points only...")
    # Filter out ground points (Classification 2 and 3 are vegetation)
    veg_points = pd_laz[(pd_laz['Classification'] == 2) | (pd_laz['Classification'] == 3)] # | means OR
    print(f"Total vegetation points: {len(veg_points)}")    
    
    #### Diversity metrics for vegetation points
    veg_gt0m = veg_points[['X', 'Y', 'Z']].copy()
    veg_gt0m = veg_gt0m[veg_gt0m['Z'] > 0] # Filter points based on Z value greater than 0
    veg_gt0m['height_bins'] = pd.cut(veg_gt0m['Z'], bins=np.arange(0, 41, 1)) # height bins
    n = veg_gt0m['height_bins'].value_counts() # Group by height bins and calculate the number of points in each bin
    p = n / n.sum() # Calculate the proportion of points in each bin

    # Calculate diversity metrics
    n_heightlcass_veg = np.power(np.sum(np.power(p, 0)), 1/(1-0))
    n_heightlcass_veg = np.sum(n > 0) #give only the number of classes with non-zero values
    # Calculate the Shannon index
    shannon_veg = np.exp(-np.sum(p * np.log(p)))
    # Calculate the Gini-Simpson index
    gini_simpson_veg = np.power(np.sum(np.power(p, 2)), 1/(1-2))

    gini_simpson_dict_veg = {
        'n_heightlcass_veg': n_heightlcass_veg,
        'Shannon_veg': shannon_veg,
        'Gini-Simpson_veg': gini_simpson_veg
    }
    print("Diversity Metrics:")
    print(gini_simpson_dict)

    
    #### Z metrics
    mean_veg = veg_points['Z'].mean()
    median_veg = veg_points['Z'].median()
    std_veg = veg_points['Z'].std()
    var_veg = veg_points['Z'].var()
    min_veg = veg_points['Z'].min()
    max_veg = veg_points['Z'].max()
    range_veg = max_veg - min_veg
    q99_veg = veg_points['Z'].quantile(0.99) 
    prop_above_2_veg = np.sum(veg_points['Z'] > 2) / veg_points.shape[0]
    prop_above_5_veg = np.sum(veg_points['Z'] > 5) / veg_points.shape[0]
    prop_above_10_veg = np.sum(veg_points['Z'] > 10) / veg_points.shape[0]
    
    
    # addd all vegetation metrics to the text file
    with open(results_txt_path, 'a') as f:
        f.write('Z Statistics of Vegetation Points Only \n')
        f.write('============\n\n') # \n creates a new line #\n\n creates a new paragraph
        f.write(f'Mean_Veg: {mean_veg}\n')
        f.write(f'Median_Veg: {median_veg}\n')                  
        f.write(f'Standard deviation_Veg: {std_veg}\n')
        f.write(f'Variance_Veg: {var_veg}\n')
        f.write(f'Minimum_Veg: {min_veg}\n')
        f.write(f'Maximum_Veg: {max_veg}\n')        
        f.write(f'Range_Veg: {range_veg}\n')
        f.write(f'99th percentile_Veg: {q99_veg}\n')
        f.write(f'Proportion of points above 2_Veg: {prop_above_2_veg}\n')
        f.write(f'Proportion of points above 5_Veg: {prop_above_5_veg}\n')
        f.write(f'Proportion of points above 10_Veg: {prop_above_10_veg}\n')
        f.write('\n')
        f.write('Diversity Metrics of Vegetation Points Only \n')
        f.write('============\n\n') # \n creates a new line #\n\n creates a new paragraph
        f.write(f'n_heightlcass_Veg: {n_heightlcass_veg}\n')
        f.write(f'Shannon_Veg: {shannon_veg}\n')
        f.write(f'Gini-Simpson_Veg: {gini_simpson_veg}\n')
        f.write('\n')
    
    
    
    # append the vegetation results to the dataframe
    new_row_veg = {'Square': square,
               'Mean_Z_Veg': mean_veg, 
               'Median_Z_Veg': median_veg, 
               'Std_Z_Veg': std_veg, 
               'Var_Z_Veg': var_veg, 
               'Min_Z_Veg': min_veg, 
               'Max_Z_Veg': max_veg, 
               'Range_Z_Veg': range_veg, 
               'Q99_Z_Veg': q99_veg, 
               'Prop_Above_2m_Veg': prop_above_2_veg, 
               'Prop_Above_5m_Veg': prop_above_5_veg, 
               'Prop_Above_10m_Veg': prop_above_10_veg,
                **gini_simpson_dict_veg
               }
    #add columns if they do not exist
    for col in new_row_veg.keys():
        if col not in results_df.columns:
            results_df[col] = np.nan
    #append the new row
    results_df.loc[len(results_df)] = new_row_veg
    #save the dataframe to csv
    results_df.to_csv(results_csv_path, index=False)
    
    
    
    print(f"FINISHED SQUARE {square} Z-metrics CALCULATION")
    
    
############################################ (for testing purposes or indicual/manual script execution)
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square = 'konigsplatz'
    square_list = [ "alpenplatz", "nikolai", "rundfunk", "konigsplatz", "miesbacher", "jakobgelb"]  # Replace with your square identifier
    square_list = ["elisabeth"]
    for square in square_list:
            z_metrics(main_path, square, overwrite=False)
    print("ALL SQUARES Z-METRICS CALCULATION FINISHED.")