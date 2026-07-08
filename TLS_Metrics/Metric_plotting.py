#Metric data statistics
# -*- coding: utf-8 -*-

"""
Script for plotting the results of the TLS metrics calculated for the different squares.
The script loads the metrics data from a CSV file, processes it, and generates a scatter plot for each metric, showing the variation of the metric across different squares.

The plots are saved in the specified directory within the main path.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
Originally created April 2026, last updated July 2026
"""


#### Import necessary libraries
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import matplotlib.cm as cm
from windrose import WindroseAxes


#### Define the function to calculate and plot metric data statistics
def metric_data_statistics(main_path, square_list: list):
    """
    Function to load the TLS metrics data from a CSV file, process it, and generate scatter plots for each metric across different squares.
    Parameters:
    - main_path: str, the main directory path where the metrics CSV file is located and where the plots will be saved.
    - square_list: list, the list of squares to process. 
    Returns:    
    - None: The function saves the plots in the specified directory and does not return any value.
    """
    
    squares = square_list
    print(f"Squares to process: {squares}")
    
    #Load general weather statistics
    metrics = pd.read_csv(f"{main_path}/All_Results.csv")

    #SORT all the dataframes by square name via the order in squares list
    metrics['Square'] = pd.Categorical(metrics['Square'], categories=squares, ordered=True)
    metrics = metrics.sort_values(by=['Square'])
    
    print("Loaded data:")
    print(metrics.head())
    print(metrics.columns.tolist())
    print("\n")
    
    #define plots directory
    plots_dir = f"{main_path}/Statistical_Plots/PlotsMetrics"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory at {plots_dir}")
    


    
    ##############################Begin Plotting##################################
    from StyleMaps import color_map, style_map, style_map_markers_6
    print("Color Map:", color_map)
    
    ################################# Plot of TLS Metric per Square ######################################
    #columns of dataframe metrics are our varialbes all but "Square" 
    #Visualize the value(s) of the metrics per square in a scatter plot
    #With each square represented acccordint to their complexity level (defined in StyleMaps.py)
    
    #list of variables to plot (all tls metrics in all_results.csv except for the square name)
    variables= [col for col in metrics.columns if col != 'Square']
    
    for var in variables:
        plt.figure(figsize=(15,10))
        sns.scatterplot(data=metrics, x='Square', y=var, hue='Square', palette=color_map, markers=True)#, style=style_map_markers_6)#, hue='Square', palette=color_map, style='Square', markers=True, dashes=False)
        plt.title(f"Variation of {var} across Squares", fontsize=16)
        plt.xlabel("Square", fontsize=14)
        plt.ylabel(f"{var}", fontsize=14)
        plt.legend(title='Square', fontsize=12)
        plt.grid(True)
        plt.savefig(f"{plots_dir}/Variation_{var}.png")
        plt.close()
        print(f"Saved Variation plot for {var}")
  
  
        
##############################################################################################################################
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square_list = ["rundfunk", "nikolai", "elisabeth", "miesbacher", "jakobgelb", "konigsplatz"]
    metric_data_statistics(main_path, square_list)