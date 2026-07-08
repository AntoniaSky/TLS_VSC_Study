# TODO add relative paths

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a combined script that includes various scripts 
for calculating the vegetation structure metrics from the processed TLS Point Cloud Data.
It also provides functions for visualizing the point cloud data and the derived metrics.

Further information about the individual scripts can be found in the respective script files.

Use it after the preprocessing steps in PreprocessingPointClouds.py 

Author: Antonia Hostlowsky  (Assisted by ChatGPT, CoPILOT)
Originally created June 2026, last updated July 2026
"""


# Import necessary libraries
import os


##### Z-based Metrics
# Z-based metrics are calculated based on the height (Z) values of the point cloud data
# Scripts: TLS_Metrics/Z_metrics.py and TLS_Metrics/skew_kurt.py 
from TLS_Metrics.skew_kurt import skew_kurt
from TLS_Metrics.Z_metrics import z_metrics

##### Metrics per cell: Canopy Cover, Rumple, Heterogeneity, etc.
# Calculated per cell (e.g. 1m x 1m) for the whole square, and aggregated to the square level (mean, median, std, etc.)
# Script: TLS_Metrics/calculateMetricsPerCell_efficientCollums.py
from TLS_Metrics.calculateMetricsPerCell_efficientCollums import metrics_per_cell

#### Canopy Roughness
# This script calculates the canopy roughness metric (CR) based on the method described in Huerta et al. (2020)
# Script: TLS_Metrics/canopy_roughness.py
from TLS_Metrics.canopy_roughness import canopy_roughness_metric

#### Vegetation Volume
from TLS_Metrics.volumeVegetation import volumeVegetation

#### Vegetation Area Index (VAI) and Density (VAD)
# This script calculates the Vegetation Area Index (VAI) and Vegetation Area Density (VAD) based on the method described in Zhao et al. (2021)
# Script: TLS_Metrics/vai.py
from TLS_Metrics.vai import vai_calc

#### Entropy-based metrics 
# Metrics use the distribution of points in the vertical layers to calculate metrics such as entropy, fhd, vci, etc.
# Script 1: TLS_Metrics/ce.py for canopy entropy (CE)
from TLS_Metrics.ce import cal_entropy
# Script 2: TLS_Metrics/ENL_SSCI_veg.py (ENL) 
from TLS_Metrics.ENL_SSCI_veg import enl_ssci
# Script 3: TLS_Metrics/VCI.py (VCI, FHD, Entropy)
from TLS_Metrics.VCI import vci_calc

#### Fractal-based metrics
# Metrics based on the fractal dimension of the point cloud data, such as SSCI and UCI
# Script 1: TLS_Metrics/ENL_SSCI_veg.py (SSCI)
from TLS_Metrics.ENL_SSCI_veg import enl_ssci
# Script 2: TLS_Metrics/UCI.py (UCI)
from TLS_Metrics.UCI import uci_calc


#### KDE Visualization of the point cloud data heights
# Script: TLS_Metrics/kd_plot.py and TLS_Metrics/kde_plotting.py
from TLS_Metrics.kd_plot import kde_plot_laz_square
from TLS_Metrics.kde_plotting import plot_kde_individual, plot_kde_combined, plot_kde_subfigures


#### Polishing the results after calculating the metrics
# It removes duplicated columns, keeps the latest added calculations, and saves the polished results back to CSV
# Script: TLS_Metrics/polishAllResults.py
from TLS_Metrics.polishAllResults import polish_all_results



#### Basic plotting functions for visualizing the point clouds and the derived metrics
# Script 1: TLS_Metrics/figures_plot_pc.py
from TLS_Metrics.figures_plot_pc import main
# Script 2: TLS_Metrics/Metric_plotting.py
from TLS_Metrics.Metric_plotting import metric_data_statistics




"""List of Metrics: 
- Canopy-Cover-Proportion               Atkins et al. (2018)
- Canopy Cover                          Atkins et al. (2023)
- Understorey Cover Percentage          Atkins et al. (2023)

- Canopy Relief Ratio (CRR)             Atkins et al. (2023)
- Rugosity                              Parker et al. (2004)
- Height Heterogeneity                  LaRue, et al. (2020)
- Rumple                                Parker et al. (2004)

- Canopy Roughness (CR)                 Herrero-Huerta et al. (2020)

- Vegetation Area Index (VAI)           Jia et al. (2025)

- Shannon Entropy (H)                   Jia et al. (2025)
- Vegetation Complexity Index (VCI)     Jia et al. (2025)
- Foliage Height Diversity (FHD)        Jia et al. (2025)
- Effective Number of Layers (ENL1D)    Ehbrecht et al. (2016)
- Effective Number of Layers (ENL2D)    Ehbrecht et al. (2016)
- Canopy Entropy (CE)                   X. Liu et al. (2022)

- Stand Structural Complexity Index (SSCI)      Ehbrecht et al. (2017)
- Understorey Complexity Index (UCI)    Willim et al. (2019)

- Volume Index (VI)                     Shokirov et al. (2023)

- descriptive Height Statistics         Shokirov et al. (2023)
- Mean Outer Canopy Height (MOCH)       Shokirov et al. (2023)
- Average Understorey Height            Shokirov et al. (2023)
"""

