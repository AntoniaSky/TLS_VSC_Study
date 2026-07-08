"""
Script for polishing dataframe "all results" after metric calculations are complete
This script removes duplicated columns, keeps the latest added calculations, and saves the polished results back to CSV.

Author: Antonia Hostlowsky (Assisted by ChatGPT, CoPILOT)
Originally created Autum 2025, last updated June 2026

"""


#opening the general results csv and merging  via square name the results from volumeVegetation.py and VCI.py
import pandas as pd
import os

def polish_all_results(main_path):
    """This function polishes the general results CSV file by removing duplicated columns, keeping the latest added calculations, and saving the polished results back to CSV.
    Args:
        main_path (str): The path to the main directory containing the data and the folders to the final processed point cloud data.
    Returns:
        None: The function saves the calculated VAI values to the general results CSV file and does not return any value.
    """
    # Load the general results CSV file
    general_results_path = f'{main_path}/All_Results.csv'
    if not os.path.exists(general_results_path):
        print(f"General results file not found at {general_results_path}")
        return

    general_results_df = pd.read_csv(general_results_path)

    # remove the following columns: 
    cols_to_remove = ['veg_volume_per_height',
                      'Canopy_Cover_Percentage_2','Understorey_Cover_Percentage_2',
                      'Mean_CRR_2','Mean_CC_2','Mean_Z_2','Mean_VDM_2']
    general_results_df = general_results_df.drop(columns=cols_to_remove, errors='ignore')
    
    #remove cols ending "_5"
    cols_to_remove_5 = [col for col in general_results_df.columns if col.endswith('_5')]
    print(f"Removing columns: {cols_to_remove_5}")
    general_results_df = general_results_df.drop(columns=cols_to_remove_5, errors='ignore')
    
    # filter out duplicated rows keep=second to keep the latest added calculations 
    general_results_df = general_results_df.drop_duplicates(keep='last')
    
    # group the rows by 'Square' and if double variables exist take the last entry
    general_results_df = general_results_df.groupby('Square', as_index=False).last()

    # Save the polished results back to CSV
    polished_results_path = f'{main_path}/All_Results.csv'
    general_results_df.to_csv(polished_results_path, index=False)
    print(f"Polished results saved to {polished_results_path}")
    
if __name__ == "__main__":
    main_path = '/Volumes/T7_Shield_A/msc'
    polish_all_results(main_path)