# TLS_VSC_Study 
A repository for TLS Point Cloud Data for vegetation structural complexity analysis (VSC) on public squres. 

Data used was aquired with Riegl-VZ400i without RGB


### Getting Started
- For Preprocessing: Check the Script PreprocessingPointClouds.py and run it
- For Calculating the VSC: Check out the Script VSC_Metrics_forTLS.py and run it

### Directory Structure
*Folder tls_data* 
Used to store the registered data in a subfolder named squarename_reg 

*Folder PointCloudPreprocessing_Methods* 
Used to store the scripts of individual functions/methods for the preprocessing steps. 
The combined script (PreprocessingPointClouds) is refering to these methods.

*Folder TLS_Metrics* 
Used to store the scripts of individual functions/methods for calculating the VSC-Metrics from TLS Data
The combined script (VSC_Metrics_forTLS) is refering to these methods.


*Script definitions_a.py* 
Some methods for handling TLS data stored as .laz files used for preprocessing the files


*Script PreprocessingPointClouds.py* 
Combined Script refering to indiviudal functions in the correct order for preprocessing TLS Data

*Script VSC_Metrics_forTLS.py* 
Combined Script refering to indiviudal functions and calculates all VSC-Metrics for a final (preprocessed) point cloud

*Script StyleMaps.py* 
Collection of Stylemaps and lists for labelling the squares and quadrants of the pointcloud for preprossing and visualization

This rep was used for the Abstract submitted to EGU 2026: https://doi.org/10.5194/egusphere-egu26-19722