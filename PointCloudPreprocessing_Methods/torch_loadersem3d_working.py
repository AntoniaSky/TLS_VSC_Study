#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created in Dec 2025, last updated June 2026

This script classifies a point cloud with the RandLA-Net model (Hu et al., 2020) trained on the Semantic3D dataset (Hackel et al. 2017). 
This script is meant to be run on a stronger machine (eg linux) due to the computational demands of the RandLA-Net model.
This script used the implementation of the RandLA-Net model and the Semantic3D dataset loader from the Open3D-ML library.

Input requirements:
- A text file containing the point cloud data in the format expected by the Semantic3D dataset (< 100 Mio Points)
- A configuration file for the RandLA-Net model (also trained on the Semantic3D dataset, downloaded from the Model Zoo, GitHub repository of Open3D-ML)
- A checkpoint file containing the trained weights of the RandLA-Net model (trained on the Semantic3D dataset, downloaded from the Model Zoo; GitHub repository of Open3D-ML)

Output:
- A text file containing the predicted Classification labels for each point in the input point cloud.

The script uses all laz files in the specified input directory, put only one laz file in the input directory if you want to classify only one specific point cloud.
Sources: 
- Hackel, T., Savinov, N., Ladicky, L., Wegner, J. D., Schindler, K., and Pollefeys, M. (2017). 
“SEMANTIC3D.NET: A new large-scale point cloud classification benchmark”. In: ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences. 
Vol. IV-1-W1, pp. 91 - 98.
- Hu, Q., Yang, B., Xie, L., Rosa, S., Guo, Y., Wang, Z., Trigoni, N., and Markham, A. (2020). “RandLA-Net: Efficient Semantic Segmentation of Large-Scale Point Clouds”. 
In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11105 - 11114. DOI: 10.1109/CVPR42600.2020.01112.




Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT, based on: open3d_experiments/semantic_torch.py by carlos-argueta (Dec 2025))
"""

import open3d as o3d
import open3d.ml as _ml3d
import open3d.ml.torch as ml3d
import open3d.ml.torch.datasets as datasets
import yaml
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: manually change the filename variable and pathes to the input and output files as needed.
filename = "nikolai_quad5_DS5mm_relH_I"

# Use Semantic3D loader
dataset = datasets.Semantic3D(dataset_path="/media/philip/T7_Shield_A/msc/text_Sem3d", test_files=[f"{filename}.txt"]) #test_files=["jakobgelb_DS100mm_quad2.txt"])
output_path = f"/media/philip/T7_Shield_A/msc/output_Randlanet/{filename}_class.txt"
checkpoint_path = "/home/philip/Desktop/msc/python/lidar/config_weights/randlanet_semantic3d_202201071330utc.pth"
config_path = "/home/philip/Desktop/msc/python/lidar/config_weights/randlanet_semantic3d.yml"
print(f"Loading config from: {config_path}")
print(filename)
try:
    cfg = _ml3d.utils.Config.load_from_file(config_path)
except ImportError:
    print("using yaml import")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

model = ml3d.models.RandLANet(**cfg['model'])

print("Building model...")
model = ml3d.models.RandLANet(**cfg.model)

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
pipeline = ml3d.pipelines.SemanticSegmentation(model)

# Run inference (chunking is handled automatically)
indx = 0
test_split = dataset.get_split("test")
data = test_split.get_data(indx)
print(data)
    
result = pipeline.run_inference(data)
labels = result["predict_labels"]
# Save or visualize labels here

point = data["point"]
intensity = data["intensity"]

output = np.hstack((point,intensity[:, None],labels[:, None]))

np.savetxt(output_path, output)

