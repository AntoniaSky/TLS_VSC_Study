#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Autum 2025, last updated June 2026

Processing SOP Backup per square (each square has its own backup csv file)

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)

"""

import os
import open3d as o3d
import numpy as np
import pandas as pd
import datetime
import laspy

from definitions_a import load_laz_file, save_laz_file, load_sop_backup_csv
from pyproj import Transformer

def enu_to_geodetic(lat0, lon0, h0, xEast, yNorth, zUp):
    """
    Convert local ENU offsets (m) to global WGS84 (lat, lon, alt).
    """
    # Define transformers
    transformer_llh_to_ecef = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
    transformer_ecef_to_llh = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

    # Reference point in ECEF
    x0, y0, z0 = transformer_llh_to_ecef.transform(lon0, lat0, h0)

    # Convert ENU offsets to ECEF offsets using rotation matrix
    lat0_rad = np.radians(lat0)
    lon0_rad = np.radians(lon0)

    t = np.array([
        [-np.sin(lon0_rad), np.cos(lon0_rad), 0],
        [-np.sin(lat0_rad) * np.cos(lon0_rad), -np.sin(lat0_rad) * np.sin(lon0_rad), np.cos(lat0_rad)],
        [np.cos(lat0_rad) * np.cos(lon0_rad), np.cos(lat0_rad) * np.sin(lon0_rad), np.sin(lat0_rad)]
    ])

    d_ecef = t.T @ np.array([xEast, yNorth, zUp])
    x, y, z = np.array([x0, y0, z0]) + d_ecef

    # Back to geodetic
    lon, lat, h = transformer_ecef_to_llh.transform(x, y, z)
    return lat, lon, h


def process_sop_backup(square: str, main_path: str, overwrite: bool = False) -> None:
    """
    Process SOP backup data for a given square.

    Parameters:
    - square (str): Name of the square to process.
    - main_path (str): Main path where data is stored.
    - overwrite (bool): Whether to overwrite existing files.
    """
    print(f"Processing SOP backup for square: {square}")
    # Load SOP backup CSV
    sop_backup_df = load_sop_backup_csv(square=square, main_path=main_path)
    print(sop_backup_df.head())
    # load csv file for FirstPosition
    first_position_df = pd.read_csv(f"data/DataStorage(FirstPosition).csv")
    first_position_df = first_position_df[first_position_df['Square'] == square]
    print(first_position_df.head())
    latitude = first_position_df['Latitude [deg]'].values[0]
    longitude = first_position_df['Longitude [deg]'].values[0]
    altitude = first_position_df['Altitude [m]'].values[0]

    # new column for sop_backup_df for x,y,z
    sop_backup_df['latitude'] = latitude
    sop_backup_df['longitude'] = longitude
    sop_backup_df['altitude'] = altitude
    
    # sort sop_backup_df by ScanPos
    sop_backup_df = sop_backup_df.sort_values(by=['scanPosName']).reset_index(drop=True)

    

    # Process each entry in the DataFrame
    for index, row in sop_backup_df.iterrows():
        print(f"Processing ScanPos {index} for square {square}")
            
        # get offsets in meters
        offset_x = row['x_sop']
        offset_y = row['y_sop']
        offset_z = row['z_sop']

        new_lat, new_lon, new_alt = enu_to_geodetic(latitude, longitude, altitude, offset_x, offset_y, offset_z)
        sop_backup_df.at[index, 'latitude'] = new_lat
        sop_backup_df.at[index, 'longitude'] = new_lon
        sop_backup_df.at[index, 'altitude'] = new_alt
            
    # Save updated SOP backup CSV
    sop_backup_path = f"{main_path}/{square}_SOP_WorldCoord.csv"
    sop_backup_df.to_csv(sop_backup_path, index=False)
    print(f"Completed processing SOP backup for square: {square}")
    
if __name__ == "__main__":
    # square = "konigsplatz"
    square_list = ["konigsplatz", "alpenplatz", "miesbacher", "wetterstein", "jakobgelb", "elisabeth", "nikolai",  "rundfunk"] #"walchensee",
    for square in square_list:
        main_path = f"tls_data/{square}_reg"  # manually enter the main path used for the following scripts
        process_sop_backup(square=square, main_path=main_path, overwrite=False)
        #read the saved file and 
        sop_worldcoord_path = f"{main_path}/{square}_SOP_WorldCoord.csv"
        sop_worldcoord_df = pd.read_csv(sop_worldcoord_path)
        #append to a master csv file
        master_csv_path = f"data/All_Squares_SOP_WorldCoord.csv"
        if os.path.exists(master_csv_path):
            master_df = pd.read_csv(master_csv_path)
            master_df = pd.concat([master_df, sop_worldcoord_df], ignore_index=True)
            master_df["Square"] = square
            master_df.to_csv(master_csv_path, index=False)
        else:
            sop_worldcoord_df.to_csv(master_csv_path, index=False)
            #add column for Square
            sop_worldcoord_df['Square'] = square
    print(f"All squares processed.")