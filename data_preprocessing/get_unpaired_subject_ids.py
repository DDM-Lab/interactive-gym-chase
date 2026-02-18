# Script will receive a series of csv files with the following format:
# column names: subject_id,category,compensation,team_id,partner_num,quitter_ID
# example of values: ee509710-774e-402c-b6f6-da35707f5b9c,unpaired,0.5,,,

# The script should 
# (1) iterate across a specified list of folders in folder_list within the directory file_path, and find the subject_compensations.csv files.
# (2) For each csv file, extract the subject_id values where category is "unpaired"
# (2a) If there are repeated participant IDs across folders, only include them once in the final output, but note all folders they were found in.
# (3) Generate a new CSV file that contains the subject_id values for all unpaired subjects across all folders, along with the folder name they were found in.

import pandas as pd
import os
from pathlib import Path

folder_list = ["pilot_2_aggregated_data", "post_pilot", "run_1_aws", "run_1_Janus", "run_2", "run_2_9", "run_2_9_t2", "run_3_and_4", "run_5", "run_6", "run_7"]

file_path = "C:\\Users\\groessli\\Documents\\GitHub\\interactive-gym-chase\\data_preprocessing\\human_only\\aggregated_data"

csv_filename = "subject_compensations.csv"

# Dictionary to store subject_id and list of folders where they were found
unpaired_subjects = {}

# Dictionary to store team_id and list of folders where they were found
completed_teams = {}

# Iterate through each folder in folder_list
for folder in folder_list:
    folder_path = os.path.join(file_path, folder)
    csv_path = os.path.join(folder_path, csv_filename)
    
    # Check if the CSV file exists
    if os.path.exists(csv_path):
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Filter for unpaired subjects
            unpaired_df = df[df['category'] == 'unpaired']
            
            # Extract subject_ids and add folder info
            for subject_id in unpaired_df['subject_id']:
                if subject_id not in unpaired_subjects:
                    unpaired_subjects[subject_id] = []
                unpaired_subjects[subject_id].append(folder)
            
            # Collect completed team_ids
            completed_df = df[df['category'] == 'completed']
            # Get unique team_ids to avoid duplicates (each team appears twice in the CSV)
            unique_team_ids = completed_df['team_id'].unique()
            for team_id in unique_team_ids:
                if pd.notna(team_id):  # Handle NaN values
                    if team_id not in completed_teams:
                        completed_teams[team_id] = []
                    completed_teams[team_id].append(folder)
        
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
    else:
        print(f"File not found: {csv_path}")

# Create output dataframe
output_data = []
for subject_id, folders in unpaired_subjects.items():
    output_data.append({
        'subject_id': subject_id,
        'folders_found': '; '.join(folders)
    })

output_df = pd.DataFrame(output_data)

# Sort by subject_id for consistency
output_df = output_df.sort_values('subject_id').reset_index(drop=True)

# Save to CSV file
output_path = os.path.join(file_path, "unpaired_subjects.csv")
output_df.to_csv(output_path, index=False)

# Create output dataframe for completed teams
completed_data = []
for team_id, folders in completed_teams.items():
    completed_data.append({
        'team_id': team_id,
        'folders_found': '; '.join(folders)
    })

completed_df_output = pd.DataFrame(completed_data)

# Sort by team_id for consistency
completed_df_output = completed_df_output.sort_values('team_id').reset_index(drop=True)

# Save completed teams to CSV file
completed_teams_path = os.path.join(file_path, "completed_teams.csv")
completed_df_output.to_csv(completed_teams_path, index=False)

print(f"Script completed successfully!")
print(f"Found {len(output_df)} unique unpaired subjects")
print(f"Output saved to: {output_path}")
print(f"\nCompleted teams summary:")
print(f"Total number of completed teams: {len(completed_teams)}")
print(f"Completed teams output saved to: {completed_teams_path}")