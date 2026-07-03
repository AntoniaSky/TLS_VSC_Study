#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Dec 2025, last updated June 2026

Use after height normalization with CSF (Script: PointCloudPreprocessing_Methods/normalizeCSFLowestPointGrid.py)
Preparing for Model pretrained on Semantic3D (Hackel et al., 2017) which uses the attributes X Y Z Intensity R G B for each point.
The output of this script is a TXT file with the attributes X Y Z Intensity R G B for each point.
RGB values are replaced with the intensity values, when rgb not available. 
If intensity is used it is normalized to the range 0-255 and used for all three RGB channels mimicking a black and white image.

Sources: 
- Hackel, T., Savinov, N., Ladicky, L., Wegner, J. D., Schindler, K., and Pollefeys, M. (2017). 
“SEMANTIC3D.NET: A new large-scale point cloud classification benchmark”. In: ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences. 
Vol. IV-1-W1, pp. 91 - 98.


Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
"""


import laspy
import numpy as np
import os

from definitions_a import load_laz_file, save_laz_file

from StyleMaps import quads

def convert_laz_to_txt(square: str, main_path: str, overwrite: bool = False):
    """
    Convert LAZ files to TXT format for Classification. Preparing for Model pretrained on Semantic3D (Hackel et al., 2017) which uses the attributes X Y Z Intensity R G B for each point.

    Output TXT file will have the attributes X Y Z Intensity R G B for each point.
    RGB values are replaced with the intensity values, when rgb not available. 
    If intensity is used it is normalized to the range 0-255 and used for all three RGB channels mimicking a black and white image.

    Args:
        square (str): The square name
        main_path (str): The main path where the files are located
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    Returns:
        None - Writes the TXT file to the specified location
    """
    if os.path.exists(f"{main_path}/{square}_merged_noClip/{square}_all_points_DS5mm_relH.laz"):
        quad_list = ["all_points"]
        #quad_list = ["all_points_lower", "all_points_upper"]
    else:
        quad_list = quads #from StyleMaps.py
        
        
    for quad in quad_list:
        input_laz = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH.laz"
        output_txt = f"{main_path}/{square}_merged_noClip/{square}_{quad}_DS5mm_relH_I.txt"
        print(f"Converting {input_laz} to {output_txt}")
        
        if not overwrite and os.path.exists(output_txt):
            print(f"Output file {output_txt} already exists. Skipping conversion.")
            continue
        elif not os.path.exists(input_laz):
            print(f"This quadrant has no laz file: quad = {quad}, path = {input_laz}")
            continue
        else:
            print(f"Processing conversion for {input_laz}...")  
            # Read the LAZ file
            las_points, pd_las = load_laz_file(input_laz)
            
            # Extract x, y, z coordinates
            x = pd_las['X'].values
            y = pd_las['Y'].values
            z = pd_las['Z'].values

            intensity = pd_las['Intensity'].values

            # check if RGB values are available, if not use intensity values for RGB
            if 'Red' in pd_las.columns and 'Green' in pd_las.columns and 'Blue' in pd_las.columns:
                r = pd_las['Red'].values
                g = pd_las['Green'].values
                b = pd_las['Blue'].values
            else:
                # add rgb values based on intensity and return number
                r = intensity
                g = r.copy()
                b = r.copy()

            # Normalize the rgb values to the range 0-255
            r = np.clip((r / np.max(r)) * 255, 0, 255).astype(np.uint8) 
            g = np.clip((g / np.max(g)) * 255, 0, 255).astype(np.uint8)
            b = np.clip((b / np.max(b)) * 255, 0, 255).astype(np.uint8)

            print(f"Writing to TXT file: {output_txt}...")
            with open(output_txt, "w") as f:
                for xi, yi, zi, inten, ri, gi, bi in zip(x, y, z, intensity, r, g, b):
                    f.write(f"{xi} {yi} {zi} {inten} {ri} {gi} {bi}\n")

            print(f"CONVERSION COMPLETE: {output_txt}")
    print("CONVERSION OF ALL QUADRANTS COMPLETE.")
    print("\n" * 3)

if __name__ == "__main__":
    square = "elisabeth" # manually enter the square name used for the following scripts
    overwrite = False  # Set to True to overwrite the existing CSV file
    main_path = f'tls_data/{square}_reg'

    convert_laz_to_txt(square, main_path, overwrite=overwrite)