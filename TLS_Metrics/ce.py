#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script calculates the Canopy Entropy Metric (CE) from the processed TLS point cloud data.
The CE metric is based on the distribution of points in the vertical layers of the canopy and is calculated using kernel density estimation.

Code is adapted from the original code by Liu et al. (2022) (calculation was not changed, but the code was adapted to fit into the workflow of this project and to be applied to the TLS point cloud data of this project):
Code uses file ce_utils.py (in same folder)


Source:
- Liu, X., et al. (2022). “A novel entropy-based method to quantify forest canopy structural complexity from multiplatform lidar point clouds”. In: Remote Sensing of Environment 282, p. 113280. DOI: 10.1016/j.rse.2022.113280.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT) adapted from original code by Liu et al. (2022) uploaded in github repository of the project https://github.com/LidarSu/Canopy_entropy (June 2026)

Originally created Spring 2026, last updated June 2026
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from ce_utils import read_las, filter_pts, adaptive_sample_pts


def cal_entropy(pts, is_resample=True, bandwidth=0.2, grid_size=0.1, is_fig=False, fig_path='./tmp', save_data=True, return_bw=False, square="alpenplatz"):
    """ Calculate the continuous entropy by using kernel density estimation and project 3D point cloud
        into plane include (x-y, x-z, y-z)

    Args:
        pts:            np.array(n, 3),  point cloud
        is_resample:    bool,            wheter using MK-test resampling strategy
        band_width:     float,           band width of kernel density, if band_width is None, it would be calculated using grid search
        grid_size:      float,           grid size for calcualting the integral of continous entropy
        is_fig:         bool,            whether output figure
        fig_path:       str,             path to output figure
        save_data:      bool,            whether save the density data
        return_bw:      bool,            whether return band width

    Return:
        c_entropys:                 pd.DataFrame    entropy include three planes
        band_width(optional):       list, band width for generating density
    """
    
    if len(pts.shape) == 1:
        pts = pts.reshape(-1, 1)
    if is_fig:
        plt.figure()
        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))

    if is_resample:
        pts = adaptive_sample_pts(pts,  is_figure=True, fig_path=fig_path, square=square)

    print("Starting the calculation of canopy entropy after the resampling of point cloud using MK-test strategy" if is_resample else "Starting the calculation of canopy entropy without resampling")

    pts_min = np.min(pts, axis=0)
    pts_max = np.max(pts, axis=0)
    c_entropys = []
    bw = bandwidth
    for axis_0, axis_1 in zip([0, 0, 1], [1, 2, 2]):
        print("Starting with axis")
        # print('{}{}'.format('xyz'[axis_0], 'xyz'[axis_1]))
        kde = KernelDensity(kernel='gaussian', bandwidth=bw, rtol=0.001).fit(pts[:, [axis_0, axis_1]])
        if grid_size is None:
            grid_size = bw
        col_grids = int((pts_max[axis_0] - pts_min[axis_0] + 8*bw) / grid_size) + 2     # Plusing 8bw for generating appropriate border of kernal density estimation
        col_locs = pts_min[axis_0] - 4*bw + (np.arange(col_grids)-0.5) * grid_size

        row_grids = int((pts_max[axis_1] - pts_min[axis_1] + 8*bw) / grid_size) + 2
        row_locs = pts_max[axis_1] + 4*bw - (np.arange(row_grids)-0.5) * grid_size
        X, Y = np.meshgrid(col_locs, row_locs)
        xy = np.vstack([X.ravel(), Y.ravel()]).T

        print("Calculating density for plane {}{}".format('xyz'[axis_0], 'xyz'[axis_1]))
        density = np.exp(kde.score_samples(xy))     # The density of KernelDensity in sklearn is logirithm
        density_nozero = density[density > 0]       # Removing the density whose value is zero
        c_entropy = -1 * np.sum(density_nozero*np.log(density_nozero)*grid_size*grid_size)

        if is_fig:
            density_matrix = density.reshape(row_grids, col_grids)
            plt.imshow(density_matrix)
            plt.tight_layout()
            plt.savefig(fig_path + '_{}{}{}.jpg'.format('xyz'[axis_0], 'xyz'[axis_1], square))
            plt.clf()
            if save_data:
                np.save(fig_path + '_{}{}{}.npy'.format('xyz'[axis_0], 'xyz'[axis_1], square), density_matrix)

        c_entropys.append(c_entropy)
        print("Continuous entropy for plane {}{}: {}".format('xyz'[axis_0], 'xyz'[axis_1], c_entropy))
    c_entropys = np.array(c_entropys).reshape(1, -1)
    c_entropys = pd.DataFrame(c_entropys, columns=['ce_xy', 'ce_xz', 'ce_yz'])
    c_entropys['ce'] = np.sqrt(np.sum(c_entropys[['ce_xy', 'ce_xz', 'ce_yz']].values ** 2))
    
    print("Canopy entropy calculation finished. Final canopy entropy is {}.".format(c_entropys['ce'].values[0]))

    if is_fig:
        plt.close()
    return c_entropys

###############################
if __name__ == '__main__':
    square_list = [ 'miesbacher', "jakobgelb", "elisabeth", "konigsplatz", 'alpenplatz','rundfunk','nikolai']
    for square in square_list:
        print("Starting processing square {}.".format(square))
        file_path = f'/Volumes/T7_Shield_A/msc/{square}_FINAL/{square}_square_DS5mm_FINAL.laz'         # Tha path to point cloud file
        pts, classification = read_las(file_path)

        # Removing the ground points. Note that the point cloud must be normalized before this step
        pts = filter_pts(pts, classification)

        # Calculating canopy entropy
        print("Calculating canopy entropy")
        c_entropys = cal_entropy(pts, is_fig=True, fig_path=os.path.join('../data', 'figure/test'), square=square)
        # Saving calculated canopy entropy to file
        c_entropys.to_csv(f'/Volumes/T7_Shield_A/msc/{square}_FINAL/{square}_square_DS5mm_FINAL_ce.csv', index=False)
        print(f"Canopy entropy calculation finished, the result is saved in the same folder as point cloud file with name {os.path.basename(file_path).split('.')[0]}.csv")
        
        
        # load the general results csv
        general_results_path = f'/Volumes/T7_Shield_A/msc/All_Results.csv'        
        
        new_row = {
            "Square": square,
            "ce_xy": c_entropys['ce_xy'].values[0],
            "ce_xz": c_entropys['ce_xz'].values[0],
            "ce_yz": c_entropys['ce_yz'].values[0],
            "ce": c_entropys['ce'].values[0],
        }
        
        if os.path.exists(general_results_path):
            general_results_df = pd.read_csv(general_results_path)
            # check if the columns exist, if not create them
            for col in new_row.keys():
                if col not in general_results_df.columns:
                    general_results_df[col] = np.nan
        else:
            general_results_df = pd.DataFrame(columns=new_row.keys())
        
        general_results_df.loc[len(general_results_df)] = new_row
        general_results_df.to_csv(general_results_path, index=False)
        print(f"General results updated at {general_results_path}")
        
        
        print("Finished processing square {}.".format(square))
        