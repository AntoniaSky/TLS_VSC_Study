##### Style Maps
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


#Canopy Metric:
canopy = "Canopy_Coverage_Percentage_Above2m_1"
#canopy = "Mean_CC_corr_1" 
#canopy = "Median_CC_corr_1"

#Confoundng variables:
confounding_vars_all = ["BuildingVol_m3_ClimStat",
                    "area_m2", 
                    "DistanceToMeasurementStation_km", 
                    "DistanceCityCenter_km", 
                     "Green_Percentage_20m", 
                     #"TLS_Date",
                     "Hertz_khz", "Resolution_Phi_Teta_mdeg", 
                     "Number_Total_Points", "Storage_Size_MB"
                    ]

confounding_vars_final = ["area_m2", "DistanceCityCenter_km", "BuildingVol_m3_ClimStat"]


#VSC Metrics: all and final with old names
vsc_metrics_all = [
    "Understorey_Coverage_Percentage_1",
                    "Canopy_Coverage_Percentage_Above2m_1",
                    "Mean_CC_corr_1","Median_CC_corr_1",
                    
                   
                    
                    "Mean_CRR_1","Max_CRR_1","Min_CRR_1",
                    
                    "Rumple_Index_1",
                    "Rugosity_VegMaxZ_1",
                    "Rugosity_VegStdZ_1",
                    
                    "canopy_roughness_m2",
                    
                    #"vai_1m",
                    "vai_1m_corr",
                    
                    
                    "entropy_1m",
                    "fhd_1m",
                    "vci_1m",
                    
                    #"fractal_dim",
                    "ENL_1D", "ENL_2D",
                    
                    "ce",
                    
                    "UCI_Frac", #renamed ones
                    "ssci",
                    
                    "veg_volume",
                   
                    "Mean_Z_Veg","Median_Z_Veg","Std_Z_Veg","Var_Z_Veg",
                    "Max_Z_Veg",#"Range_Z_Veg",
                    "Q99_Z_Veg",
                    "Prop_Below_2m_Veg",#"Prop_Above_5m_Veg","Prop_Above_10m_Veg",
                    
                    
                    
                    "Mean_Z_1","Std_Z_1",
                    "Mean_Understorey_Height_1",
                    "MOCH_1", 
                    
                    "skewness","kurtosis",
                    
                    
                    #calculated ones
                    "ce_per_area", "ENL_to_HeightLayers_Ratio",'SSCI_to_HeightLayers_Ratio',
                    'Veg_Volume_to_Area_Ratio',
                    
            ]
                    
   
vsc_final = [
    "Rugosity_VegStdZ_1",
    "ce", 
    "Prop_Below_2m_Veg",
    "fhd_1m",
    "Max_Z_Veg",
    "ENL_1D",   
    "ssci",
    "UCI_Frac",
    "Understorey_Coverage_Percentage_1"
    ]





#### Square Ordering
#squure order complexity categories and measurement month for coloring and styling plots
squares_order_complexity = ["nikolai","rundfunk", "elisabeth", "miesbacher","jakobgelb","konigsplatz", "stjakob", "wittelsbacher"]
squares_order_measurement_month = ["rundfunk", "elisabeth", "konigsplatz", "stjakob", "nikolai", "miesbacher","jakobgelb","wittelsbacher"]
squares_order_complexity_NoC = ["nikolai","rundfunk", "elisabeth", "miesbacher","jakobgelb","konigsplatz"]
squares_order_measurement_month_NoC = ["rundfunk", "elisabeth", "konigsplatz", "nikolai", "miesbacher","jakobgelb"]


squares_labels = {
    "rundfunk": "Rundfunkplatz",
    "elisabeth": "Elisabethplatz",
    "konigsplatz": "Königsplatz",
    "stjakob": "St.-Jakobs-Platz",
    "nikolai": "Nikolaiplatz",
    "miesbacher": "Miesbacher Platz",
    "jakobgelb": "Jakob-Gelb-Platz",
    "wittelsbacher": "Wittelsbacherplatz"
}


# Define complexity categories and measurement months for coloring and styling plots
complexity_level_KeySquares = {
    'nikolai': 'Low Complexity',
    'rundfunk': 'Low Complexity',
    'elisabeth': 'Medium Complexity',
    "miesbacher": 'Medium Complexity',
    'jakobgelb': 'High Complexity',
    'konigsplatz': 'High Complexity',
    'stjakob': 'Control',
    'wittelsbacher': 'Control'
}
color_map_forComplexities = {
            'Low Complexity': '#A2AD00',
            'Medium Complexity': '#85A202',
            'High Complexity': '#3E4D03FF'
        }


color_map_forComplexities_plusCS = {
            'Low Complexity': '#A2AD00',
            'Medium Complexity': '#85A202',
            'High Complexity': '#3E4D03FF',
            "Control": '#808080'
        }


classification_colors_pointcloud_cmap = ListedColormap(["#F4C20E", "#85A202", "#3E4D03FF"])

month_KeySquares = {
    'nikolai': 'August',
    'rundfunk': 'July',
    'elisabeth': 'July',
    "miesbacher": 'August',
    'jakobgelb': 'August',
    'konigsplatz': 'July',
    'stjakob': 'July',
    'wittelsbacher': 'August'
}


complexity_categories = {
    "Low Complexity":       [ "nikolai","rundfunk"],
    "Medium Complexity":    [ "elisabeth", "miesbacher"],
    "High Complexity":      ["jakobgelb","konigsplatz"],
    "Control":              ["stjakob", "wittelsbacher"]
    }
complexity_categories_niceNames = {
    "Low Complexity":       [ "Nikolaiplatz","Rundfunkplatz"],
    "Medium Complexity":    [ "Elisabethplatz", "Miesbacher Platz"],
    "High Complexity":      ["Jakob-Gelb-Platz","Königsplatz"],
    "Control":              ["St.-Jakobs-Platz", "Wittelsbacherplatz"]
    }

measurement_month = {
    "July": ["rundfunk", "elisabeth", "konigsplatz", "stjakob"],
    "August": ["nikolai", "miesbacher", "jakobgelb", "wittelsbacher"],
        }



dict_hours = {"early_morning": [3,4,5],# 456
                  "peak": [13,14,15], #121314
                  "evening": [18,19,20], #181920
                  "after_sunset": [22,22,0], #212223
                  "other": [1,2,6,7,8,9,10,11,12,16,17,21] # all other hours
                  }


# Create a color mapping from complexity categories
color_map = {}
colors = ["#A2AD00", "#85A202", "#3E4D03FF", "#808080"]
for idx, (complexity_label, squares_in_category) in enumerate(complexity_categories.items()):
    for square in squares_in_category:
        color_map[square] = colors[idx]
# Create a color mapping from complexity categories
color_map_niceNames = {}
colors = ["#A2AD00", "#85A202", "#3E4D03FF", "#808080"]
for idx, (complexity_label, squares_in_category) in enumerate(complexity_categories_niceNames.items()):
    for square in squares_in_category:
        color_map_niceNames[square] = colors[idx]



# Create a style mapping from measurement month
style_map = {}
for month, squares_in_month in measurement_month.items():
    line_style = "-" if month == "July" else "--" # solid for July, dashed for August
    for square in squares_in_month:
        style_map[square] = line_style

    
 
 
complexity_categories_nonContr = {
    "Low Complexity":       [ "nikolai","rundfunk"],
    "Medium Complexity":    [ "elisabeth", "miesbacher"],
    "High Complexity":      ["jakobgelb","konigsplatz"],
   # "Control":              ["stjakob", "wittelsbacher"]
    }

color_map_nonContr = {}
colors_nonContr = ["#A2AD00", "#85A202", "#3E4D03FF"] # same colors as before but without the control color
for idx, (complexity_label, squares_in_category) in enumerate(complexity_categories_nonContr.items()):
    for square in squares_in_category:
        color_map_nonContr[square] = colors_nonContr[idx]


measurement_month_nonContr = {
    "July": ["rundfunk", "elisabeth", "konigsplatz"],
    "August": ["nikolai", "miesbacher", "jakobgelb"],
        }


squares_no_control = ["nikolai","rundfunk", "elisabeth", "miesbacher","jakobgelb","konigsplatz"]    
        
style_map_markers_6 = {}
markers = ["o", "x"]
for idx, (month, squares_in_month) in enumerate(measurement_month_nonContr.items()):
    marker_style = markers[idx]
    for square in squares_in_month:
        style_map_markers_6[square] = marker_style
        
        
        



# colour map for these 4 complexity levels with new colours #
complexity_colors = {
    'LVC': "#A2AD00", "MVC": "#85A202", "HVC": "#3E4D03FF", "Control": "#808080"
}
color_map_coplexity = {}
for complexity, color in complexity_colors.items():
    color_map_coplexity[complexity] = color
    
    
    

tum_colors = {
    "blue": "#0065BD",# TUM 
    "white": "#FFFFFF",# TUM 
    "black": "#000000",# TUM 
    "med_blue": "#005293",# TUM 
    "dark_blue": "#003359",# TUM 
    "dark_grey": "#333333",# TUM 
    "darker_grey": "#262626",# darker shade of TUM dark grey
    "grey": "#808080",# TUM 
    "light_grey": "#CCCCCC",# TUM
    "beige": "#DAD7CB",# TUM 
    "orange": "#E37222",    # TUM 
    "green": "#8DAD00", #additional colors
    "light_blue": "#98C6EA", # TUM 
    "med_light_blue": "#64A0C8", # TUM 
    "red": "#C4101C",#additional colors
    "yellow": "#F4C20E", #additional colors
    "tum_green": "#A2AD00", # TUM green
    "green": "#85A202", #additional colors
     "dark_green": "#3E4D03FF", #additional colors
    "light_green": "#c4ff04ff" #additional colors
}

#Definition of quad_list so it will be same for everyfile basing on quads 
quads = ["quad1", "quad2", "quad3", "quad4", 
         "quad1_lower", "quad2_lower", "quad3_lower", "quad4_lower",
         "quad1_upper", "quad2_upper", "quad3_upper", "quad4_upper", 
         "all_points",
         "all_points_lower", "all_points_upper", 
         "quad1_part1", "quad1_part2", "quad1_part3", "quad1_part4", "quad1_part5",
         "quad2_part1", "quad2_part2", "quad2_part3", "quad2_part4", "quad2_part5",
         "quad3_part1", "quad3_part2", "quad3_part3", "quad3_part4", "quad3_part5",
         "quad4_part1", "quad4_part2", "quad4_part3", "quad4_part4", "quad4_part5"
         ]