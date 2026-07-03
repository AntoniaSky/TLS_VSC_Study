# for moving point cloud
#%%import necessary libraries
#import necessary libraries
import laspy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from definitions_a import load_laz_file, save_laz_file, visualize_xy_z, visualize_transformed_points, transform_to_polar


#define square
square = 'alpenplatz'
# define the main path
main_path = f'tls_data/{square}_reg'
print(f"Main path: {main_path}")

#####load the point cloud data
file_path = f'{main_path}/{square}_merged_noClip/{square}_01m_classrf01VG_SmallAOI_norm.laz'
laz = load_laz_file(file_path)

#####transform to pd dataframe
pd_downpcd = pd.DataFrame(laz, 
                          columns=['X', 'Y', 'Z', 
                                   'Intensity', 'ReturnNumber', 
                                   'NumberOfReturns', 'Classification'])

#histo gram of classification
classification_counts = pd_downpcd['Classification'].value_counts()
print("Classification counts:")
print(classification_counts)        

##### visualize the point cloud before moving to origin for manual checking the origin
print("Visualizing the point cloud before moving to origin...")
visualize_xy_z(laz
               )

###### Move the point cloud to the origin
x_dist = 0 #manually set the distance to move in X direction
y_dist = -5 #manually set the distance to move in Y direction

pd_downpcd['X'] += x_dist
pd_downpcd['Y'] += y_dist


#Diagram of the point cloud y-axis is height and x-axis is X, colour is Z
# Create a random sample of 10000 points
sample_size = 300000
sample_indices = np.random.choice(pd_downpcd.shape[0], size=sample_size, replace=False)    
# Create a random sample of the point cloud
points_sample = pd_downpcd.iloc[sample_indices]
# Create a 2D scatter plot of the point cloud # Create a 2D scatter plot of the point cloud 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
# Create a scatter plot of the point cloud
ax.scatter(points_sample['X'], points_sample['Y'], 
           c=points_sample['Z'], cmap='viridis', 
           s=0.5 #size 20.0 makes it like in R
           )
# Add a color bar
cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
cbar.set_label('Z axis')
# Set the labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')  # Set the aspect ratio to equal
# Set the title
ax.set_title('Point Cloud')
plt.show() 

# ####REPEAT THE steps if needed
# If you want to repeat the steps, you can uncomment the following lines

# %%# lets move the point cloud a little to origin in middle of the plot
""" # Move the point cloud to the origin
x_dist = 0 #manually set the distance to move in X direction
y_dist = 0 #manually set the distance to move in Y direction

pd_downpcd['X'] += x_dist
pd_downpcd['Y'] += y_dist

 """

#####save the DataFrame with normalized Z values to a new laz file
#save the DataFrame with normalized Z values to a new laz file
output_file_path = f'{main_path}/{square}_merged_noClip/{square}_01m_classrf01VG_SmallAOI_norm_orig.laz'
print(f"Saving DataFrame with normalized Z moved to origin to {output_file_path}...")
save_laz_file(output_file_path, pd_downpcd)


#Diagram of the point cloud y-axis is height and x-axis is X, colour is Z
# Create a random sample of 10000 points
sample_size = 300000
sample_indices = np.random.choice(pd_downpcd.shape[0], size=sample_size, replace=False)    
# Create a random sample of the point cloud
points_sample = pd_downpcd.iloc[sample_indices]
# Create a 2D scatter plot of the point cloud # Create a 2D scatter plot of the point cloud 
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
# Create a scatter plot of the point cloud
ax.scatter(points_sample['X'], points_sample['Y'], 
           c=points_sample['Z'], cmap='viridis', 
           s=0.5 #size 20.0 makes it like in R
           )
# Add a color bar
cbar = plt.colorbar(ax.collections[0], ax=ax, pad=0.1)
cbar.set_label('Z axis')
# Set the labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.axis('equal')  # Set the aspect ratio to equal
# Set the title
ax.set_title('Point Cloud')
plt.show() 

