"""This script calculates the Kernal Density Estimation (KDE) of the Z values of the point cloud data for each square,
and creates density plots to visualize the distribution of heights in the point cloud data. It also saves rough the KDE data to CSV files for further analysis.
Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT), please exuse the inefficiency of the code, it is a first version and can be optimized in the future.
Originally created Winter 2025, last updated June 2026
"""

# Import necessary libraries and files
import os
import laspy
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from definitions_a import load_laz_file



def kde_plot_laz_square(main_path, square):
    """This function creates KDE plots of the Z values of the point cloud data for a given square, and saves the plots and the KDE data to CSV files.
    
    
    Args:
        main_path (str): The path to the main directory containing the data and the folders to the final processed point cloud data.
        square (str): The identifier for the square area.
    Returns:
        None: The function saves the calculated VAI values to the general results CSV file and does not return any value.
    """
    
    print(f"START FUNCTION FOR KDE PLOT.")
    print(f"Processing square: {square}")
    # Load the LAZ file
    print("Loading LAZ file...")
    laz, pd_laz = load_laz_file(f'{main_path}/{square}_FINAL/{square}_square_DS5mm_FINAL.laz')
    
    # define folder and paths
    metrics_folder = f'{main_path}/0_Metrics/{square}_Metrics'
    os.makedirs(metrics_folder, exist_ok=True)  

    ############################################
    # Create a density plot of the Z values all points
    z_values = pd_laz['Z'].values

    #filter points based on Z value greater than 0
    z_values = z_values[z_values > 0]

    print("Creating density plot of Z values...")

    # Create a density plot
    sns.kdeplot(y=z_values, 
                bw_adjust=1, bw_method='scott',
                fill=False, color='black',
                alpha=0.8, linewidth=1, label='Z values'
                #cut=0, clip=[0, 50]
                )
    plt.xlabel('Density')
    plt.ylabel('Z Value [m]')
    #plt.title(f'Density plot of Z values ({square})')
    #y axis limits
    plt.ylim(0, 25) # set y axis limits change if a square has higher vegetation
    #gridlines drawn
    plt.grid(True)
    path = f'{metrics_folder}/{square}_Z_Density_Plot.png'
    plt.savefig(path)
    plt.close()
    
    
    ############################################
    # Create a density plot of the Z values only vegetation points
    print("Creating density plot of Z values Vegetation...")

    # Create a density plot without ground points (Classification only 2 3 taken)
    z_values = pd_laz[pd_laz['Classification'].isin([2, 3])]['Z'].values
    z_values = z_values[z_values > 0]
    
    sns.kdeplot(y=z_values, 
                bw_adjust=1, bw_method='scott',
                fill=False, color='black',
                alpha=0.8, linewidth=1, label='Z values'
                #cut=0, clip=[0, 50]
                )
    plt.xlabel('Density')
    plt.ylabel('Z Value [m]')
    #plt.title(f'Density plot of Z values Vegetation Only ({square})')
    #y axis limits
    plt.ylim(0, 25) # set y axis limits change if a square has higher vegetation
    #gridlines drawn
    plt.grid(True)
    path = f'{metrics_folder}/{square}_Z_Density_Plot_VegetationOnly.png'
    plt.savefig(path)
    plt.close()

 
    ############################################
    # save the lines of sns kdeplot to a csv file for vegetation only
    print("Saving KDE plot data of Z values Vegetation Only to CSV...")
    kde = sns.kdeplot(y=z_values, 
                      bw_adjust=1, bw_method='scott',
                      fill=False, color='black',
                      alpha=0.8, linewidth=1, label='Z values'
                      #cut=0, clip=[0, 50]
                      )
    kde_data = kde.get_lines()[0].get_data()
    kde_df = pd.DataFrame({'Z_Value': kde_data[1], 'Density': kde_data[0]})
    kde_df.to_csv(f'{metrics_folder}/{square}_Z_Density_Data_VegetationOnly.csv', index=False)
    plt.close()


    
    print(f"FINISHED SQUARE {square} KDE-Plot CALCULATION")
    

############################################
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square = 'konigsplatz'
    square_list = [# "alpenplatz",
                   "nikolai", "rundfunk", 
                   "konigsplatz", "miesbacher", "jakobgelb",
                   "elisabeth"]
    for square in square_list:
        kde_plot_laz_square(main_path, square, overwrite=False)

