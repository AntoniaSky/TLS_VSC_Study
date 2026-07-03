#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in Autum 2025

Script for loading a single laz file with laspy and printing the first few rows of the point cloud data.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
"""


import laspy
import numpy as np
import os
from definitions_a import load_laz_file

#path to the input LAZ file e.g.:
input_laz = "/Users/anti/Documents/STUDIUM_MASTER/MSC/python/data/jakobgelb/jakobgelb_msa/jakobgelb_singlePos/1_horizontal_JGP_ScanPos002 - SINGLESCANS - 250807_110929.laz"

# Read the LAZ file
las = laspy.read(input_laz)

x, y, z = las.x, las.y, las.z

#make data array
points = np.vstack((x, y, z)).transpose()

#print first few rows of the point cloud data
print("First few rows of the point cloud data:")
print(points[:5, :])


las_np, las_pd = load_laz_file(input_laz)
print("first few rows of the point cloud data using load_laz_file:")
print(las_pd.head())
print(las[:5, :])  # print first 5 points from las object
