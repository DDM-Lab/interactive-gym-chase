import os
import re
import pandas as pd
from pathlib import Path

# dir_1 = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\Pilot1-2\human-only-data-pilot-2-internal\overcooked_hh_start_scene"
# dir_2 = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\Pilot1-2\human-only-data-pilot-3\overcooked_hh_start_scene"



def extract_subject_ids(directory):
    """Extract unique subject IDs from filenames in a directory."""
    subject_ids = set()
    
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return subject_ids
    
    # Pattern to match subject IDs (UUID or MTurk worker ID format)
    # Assuming filenames are like: <subject_id>_metadata.json, <subject_id>_1.csv, etc.
    pattern = r'^(.+)_metadata'
    
    files = os.listdir(directory)
    print(f"Found {len(files)} files in {directory}")
    print(f"Sample files: {files[:5]}")
    
    for filename in files:
        match = re.match(pattern, filename)
        if match:
            subject_id = match.group(1)
            subject_ids.add(subject_id)
    
    return subject_ids

# # Get subject IDs from both directories
# ids_1 = extract_subject_ids(dir_1)
# ids_2 = extract_subject_ids(dir_2)

# print(f"Subject IDs in dir_1: {len(ids_1)}")
# print(f"Subject IDs in dir_2: {len(ids_2)}")
# print()

# # Find IDs in dir_2 that are not in dir_1
# unique_to_dir_2 = ids_2 - ids_1

# if unique_to_dir_2:
#     print(f"Subject IDs in dir_2 but NOT in dir_1 ({len(unique_to_dir_2)} total):")
#     for subject_id in sorted(unique_to_dir_2):
#         print(f"  - {subject_id}")
# else:
#     print("All subject IDs in dir_2 are also present in dir_1.")

# # Filter to keep only MTurk Worker IDs (uppercase alphanumeric, 13+ characters, no special chars)
# mturk_id_pattern = r'^[A-Z0-9]{13,}$'
# mturk_ids = {subject_id for subject_id in unique_to_dir_2 if re.match(mturk_id_pattern, subject_id)}

# print()
# print(f"MTurk Worker IDs in dir_2 but NOT in dir_1 ({len(mturk_ids)} total):")

# # Output to text file
# output_file = "mturk_ids_unique_to_dir_2.txt"
# with open(output_file, 'w') as f:
#     for subject_id in sorted(mturk_ids):
#         f.write(f"  - {subject_id}\n")
#         print(f"  - {subject_id}")

# print()
# print(f"Results written to: {output_file}")


def compare_csv_files(csv_file_1, csv_file_2, output_file, subject_id_column='subject_id', comparison_type='symmetric_diff'):
    """
    Compare two CSV files based on a subject ID column and create a new CSV with non-shared IDs.
    
    Args:
        csv_file_1: Path to first CSV file
        csv_file_2: Path to second CSV file
        output_file: Path to output CSV file
        subject_id_column: Name of the column containing subject IDs (default: 'subject_id')
        comparison_type: Type of comparison to perform:
            - 'symmetric_diff': IDs that appear in one file but not the other (default)
            - 'unique_to_file1': IDs only in file 1
            - 'unique_to_file2': IDs only in file 2
            - 'common': IDs that appear in both files
    
    Returns:
        Tuple of (result_dataframe, count_summary_dict)
    """
    
    try:
        # Read both CSV files
        df1 = pd.read_csv(csv_file_1)
        df2 = pd.read_csv(csv_file_2)
        
        print(f"Loaded {csv_file_1}: {len(df1)} rows")
        print(f"Loaded {csv_file_2}: {len(df2)} rows")
        
        # Extract subject IDs
        ids_1 = set(df1[subject_id_column].unique())
        ids_2 = set(df2[subject_id_column].unique())
        
        print(f"Unique subject IDs in file 1: {len(ids_1)}")
        print(f"Unique subject IDs in file 2: {len(ids_2)}")
        
        # Perform comparison based on type
        if comparison_type == 'symmetric_diff':
            # IDs in one file but not the other
            unique_ids = ids_1.symmetric_difference(ids_2)
            result_df1 = df1[df1[subject_id_column].isin(unique_ids)]
            result_df2 = df2[df2[subject_id_column].isin(unique_ids)]
            result_df = pd.concat([result_df1, result_df2], ignore_index=True)
            print(f"Found {len(unique_ids)} IDs that appear in only one file")
            
        elif comparison_type == 'unique_to_file1':
            # IDs only in file 1
            unique_ids = ids_1 - ids_2
            result_df = df1[df1[subject_id_column].isin(unique_ids)]
            print(f"Found {len(unique_ids)} IDs unique to file 1")
            
        elif comparison_type == 'unique_to_file2':
            # IDs only in file 2
            unique_ids = ids_2 - ids_1
            result_df = df2[df2[subject_id_column].isin(unique_ids)]
            print(f"Found {len(unique_ids)} IDs unique to file 2")
            
        elif comparison_type == 'common':
            # IDs in both files
            common_ids = ids_1.intersection(ids_2)
            result_df1 = df1[df1[subject_id_column].isin(common_ids)]
            result_df2 = df2[df2[subject_id_column].isin(common_ids)]
            result_df = pd.concat([result_df1, result_df2], ignore_index=True)
            print(f"Found {len(common_ids)} IDs that appear in both files")
        
        # Save to output file
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")
        
        summary = {
            'file_1_total_ids': len(ids_1),
            'file_2_total_ids': len(ids_2),
            'result_rows': len(result_df),
            'comparison_type': comparison_type
        }
        
        return result_df, summary
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None, None
    except KeyError as e:
        print(f"Error: Column '{subject_id_column}' not found in CSV file - {e}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


#Example usage (uncomment to use):
result_df, summary = compare_csv_files(
    'C:\\Users\\groessli\\Documents\\GitHub\\interactive-gym-chase\\data_preprocessing\\human_only\\aggregated_data\\run_1_Janus\\subject_compensations.csv',
    'C:\\Users\\groessli\\Documents\\GitHub\\interactive-gym-chase\\data_preprocessing\\human_only\\aggregated_data\\run_2\\subject_compensations.csv',
    'C:\\Users\\groessli\\Documents\\GitHub\\interactive-gym-chase\\data_preprocessing\\human_only\\aggregated_data\\comparison_result_between_run_1_and_run_2.csv',
    subject_id_column='subject_id',
    comparison_type='symmetric_diff'  # or 'unique_to_file1', 'unique_to_file2', 'common'
   )

