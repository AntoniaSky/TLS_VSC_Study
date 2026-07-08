"""This script creates KDE-Plots (Kernel Density Estimation)

Plots to visualize the KDE of the Z values of the point cloud data for each square, both individually and combined, and also as subplots in A4 landscape format.

Use after the script TLS_Metrics/kd_plot.py which calculates the KDE values and saves them to CSV files, please exuse the inefficiency of the code, it is a first version and can be optimized in the future.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
Originally created Winter 2025, last updated July 2026
"""

# imports 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from StyleMaps import *
from scipy.interpolate import make_interp_spline


##############
# Helper
##############
# load the files with stored kde values for each square

def load_kde_values(main_path, square) -> pd.DataFrame:
    """This function loads the KDE values from the CSV files for a given square, and adds a column to identify the square."""
    
    path = f'{main_path}/0_Metrics/{square}_Metrics/{square}_Z_Density_Data_VegetationOnly.csv'
    df = pd.read_csv(path)
    df['Square'] = square  # add a column to identify the square
    variable_name = f'{square}_df'
    globals()[variable_name] = df
    
    return df


def combine_kde_dataframes(main_path, square_list) -> pd.DataFrame: 
    """This function combines the KDE dataframes for all squares into a single dataframe, 
    and adds columns for measurement month, complexity category, and public square based on the square name using the corresponding dictionaries."""
    
    combined_df = pd.DataFrame()  # create an empty dataframe to store the combined data
    for square in square_list:
        df = load_kde_values(main_path, square)  # load the dataframe for each square
        combined_df = pd.concat([combined_df, df], ignore_index=True)  # concatenate the dataframes
    combined_df["Measurement_Month"] = combined_df["Square"].map(month_KeySquares)  # add a column for measurement month based on the square name using the month_KeySquares dictionary
    combined_df["Complexity_Category"] = combined_df["Square"].map(complexity_level_KeySquares)  # add a column for complexity category based on the square name using the complexity_level_KeySquares dictionary
    combined_df["Public Square"] = combined_df["Square"].map(squares_labels)  # add a column for public square based on the square name using the public_square_KeySquares dictionary
    
    return combined_df


##############
# Different Main Plotting Functions
##############
# plot individual density plots for each square individually with the same x and y limits for better comparison
def plot_kde_individual(main_path, square_list):
    combined_df = combine_kde_dataframes(main_path, square_list)  # combine the dataframes for all squares
    print(combined_df.head())  # print the first few rows of the combined dataframe to check if it looks correct
    # create a density plot for each square individually with the same x and y limits for better comparison
    for square in square_list:
        df = combined_df[combined_df['Square'] == square]  # filter the dataframe for the current square

        # sort values
        df = df.sort_values("Z_Value")

        x = df["Density"].values
        y = df["Z_Value"].values

        # remove duplicates if necessary
        unique_idx = np.unique(y, return_index=True)[1]
        y = y[unique_idx]
        x = x[unique_idx]

        # smooth interpolation
        y_smooth = np.linspace(y.min(), y.max(), 500)

        spline = make_interp_spline(y, x, k=3)
        x_smooth = spline(y_smooth)

        plt.plot(
            x_smooth,
            y_smooth,
            color=color_map[square],
            #linestyle=style_map[square],
            linewidth=4,
            alpha=0.9,
            label=squares_labels[square]
        )

        plt.xlabel('Density', fontsize=40, fontweight='bold')
        plt.ylabel('Z Value [m]', fontsize=40, fontweight='bold')
        plt.ylim(0, 25)
        plt.xticks(fontsize=40)
        plt.yticks(fontsize=40)

        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

        plt.legend(title='Public Square', fontsize=40, title_fontsize=40, labelcolor="black")

        path = f'{main_path}/0_Metrics/Z_Density_Plot_Individual_{square}.png'
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
            
# plot all squares in one plot with different colors for each square and measurement month as line style
def plot_kde_combined(main_path, square_list, color_map, style_map):

    combined_df = combine_kde_dataframes(main_path, square_list)

    plt.figure(figsize=(10, 8))

    for square in square_list:

        df = combined_df[combined_df['Square'] == square]

        # sort values
        df = df.sort_values("Z_Value")

        x = df["Density"].values
        y = df["Z_Value"].values

        # remove duplicates if necessary
        unique_idx = np.unique(y, return_index=True)[1]
        y = y[unique_idx]
        x = x[unique_idx]

        # smooth interpolation
        y_smooth = np.linspace(y.min(), y.max(), 500)

        spline = make_interp_spline(y, x, k=3)
        x_smooth = spline(y_smooth)

        plt.plot(
            x_smooth,
            y_smooth,
            color=color_map[square],
            linestyle=style_map[square],
            linewidth=2,
            alpha=0.9,
            label=squares_labels[square]
        )

    plt.xlabel('Density')
    plt.ylabel('Z Value [m]')
    plt.ylim(0, 25)

    plt.grid(True)

    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.3)

    plt.legend(title='Public Square')

    path = f'{main_path}/0_Metrics/Combined_Z_Density_Plot.png'

    plt.savefig(path, dpi=300, bbox_inches='tight')

    plt.close()


# Plot 6 KDE subplots in A4 landscape format Fits inside DIN A4 with 2.5 cm margins
def plot_kde_subfigures(main_path, square_list):

    combined_df = combine_kde_dataframes(main_path, square_list)

    # ------------------------------------------------------
    # DIN A4 landscape size
    # A4 = 29.7 x 21 cm
    # minus 2.5 cm margins on all sides
    # usable width  = 24.7 cm
    # usable height = 16 cm
    # ------------------------------------------------------

    cm = 1 / 2.54

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(29.7 * cm, 21 * cm),  # A4 landscape
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    # ------------------------------------------------------
    # GLOBAL STYLE
    # ------------------------------------------------------

    plt.rcParams.update({
        "font.size": 10,
        "axes.linewidth": 1.2
    })

    # ------------------------------------------------------
    # LOOP THROUGH SUBPLOTS
    # ------------------------------------------------------

    for ax, square in zip(axes, square_list):

        df = combined_df[combined_df['Square'] == square]

        # sort values
        df = df.sort_values("Z_Value")

        x = df["Density"].values
        y = df["Z_Value"].values

        # remove duplicate y values
        unique_idx = np.unique(y, return_index=True)[1]

        y = y[unique_idx]
        x = x[unique_idx]

        # smooth interpolation
        y_smooth = np.linspace(y.min(), y.max(), 500)

        spline = make_interp_spline(y, x, k=3)
        x_smooth = spline(y_smooth)

        # --------------------------------------------------
        # PLOT
        # --------------------------------------------------

        ax.plot(
            x_smooth,
            y_smooth,
            color=color_map[square],
            linewidth=2.5,
            alpha=0.95
        )

        # --------------------------------------------------
        # TITLES
        # --------------------------------------------------

        ax.set_title(
            squares_labels[square],
            fontsize=14,
            fontweight='bold'
        )

        # --------------------------------------------------
        # AXES
        # --------------------------------------------------

        ax.set_ylim(0, 25)

        ax.tick_params(
            axis='both',
            labelsize=10
        )

        # --------------------------------------------------
        # GRID
        # --------------------------------------------------

        ax.grid(True)

        ax.minorticks_on()

        ax.grid(
            which='minor',
            linestyle=':',
            linewidth=0.4,
            alpha=0.3
        )

    # ------------------------------------------------------
    # COMMON LABELS
    # ------------------------------------------------------

    fig.supxlabel(
        'Density',
        fontsize=16,
        fontweight='bold'
    )

    fig.supylabel(
        'Z Value [m]',
        fontsize=16,
        fontweight='bold'
    )

    # ------------------------------------------------------
    # SPACING
    # tuned for A4 + 2.5 cm margins
    # ------------------------------------------------------

    plt.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=0.12,
        top=0.90,
        wspace=0.20,
        hspace=0.25
    )

    # ------------------------------------------------------
    # SAVE
    # ------------------------------------------------------

    path = f'{main_path}/0_Metrics/KDE_Subplots_A4.pdf'

    plt.savefig(
        path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Saved subplot figure to:\n{path}")
    
##########################################################

  
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    square_list = [ "rundfunk", "elisabeth", "konigsplatz", 
                   "nikolai", "miesbacher", "jakobgelb"] #only vegetated squares not reference squares
    plot_kde_individual(main_path, square_list)
    plot_kde_combined(main_path, square_list, color_map, style_map)
    plot_kde_subfigures(main_path, square_list)