import pandas as pd
from pathlib import Path
from collections import defaultdict

#data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed\Data\FullRun1+2+3\data"
data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-AI-Speed\Data\Human-AI-Condition\AI-Speed-5FPS"
#data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed\Data\Pilot2\data"
agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\AI_speed\aggregated_data\AI_speed_5"

def aggregate_subject_data():
    """
    Iterate through CSV files in cramped_room_sp_0, group by subject ID,
    and create a dataframe for each subject containing all 20 episodes.
    
    Only includes subjects who have a completion code in end_completion_code_scene.
    
    Process:
    (1) Iterate through all CSV files in cramped_room_sp_0
    (2) Group files by subject ID (string before first "_" or the full filename if no "_")
    (3) For CSV files without episode numbers, populate "episode_num" column with 0
    (4) Generate a dataframe for each subject combining all 20 episodes in order
    (5) Exclude subjects without completion codes
    (6) Save each aggregated dataframe as a CSV file in agg_data_dir
    
    Returns:
        dict: Dictionary with subject IDs as keys and DataFrames as values,
              where each DataFrame contains all episodes for that subject
    """
    
    sp_0_dir = Path(data_dir) / "cramped_room_sp_0"
    end_scene_dir = Path(data_dir) / "end_completion_code_scene"
    agg_dir = Path(agg_data_dir)
    
    if not sp_0_dir.exists():
        raise FileNotFoundError(f"Directory not found: {sp_0_dir}")
    
    # Create aggregated data directory if it doesn't exist
    agg_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store CSV files grouped by subject ID
    subject_files = defaultdict(list)
    
    # Iterate through all CSV files and group by subject ID
    for csv_file in sorted(sp_0_dir.glob("*.csv")):
        filename = csv_file.stem  # Remove .csv extension
        
        # Extract subject ID and episode number
        parts = filename.split("_")
        subject_id = parts[0]
        
        # Determine episode number
        if len(parts) == 1:
            # No underscore: this is the base episode (episode 0)
            episode_num = 0
        else:
            # Has underscore(s): extract the episode number from the last part
            try:
                episode_num = int(parts[-1])
            except ValueError:
                # If last part is not a number, this might be a metadata file
                continue
        
        subject_files[subject_id].append((episode_num, csv_file))
    
    # Filter to only keep subjects with episode 19 (indicating completion of all 20 episodes)
    subject_files = {subject_id: files for subject_id, files in subject_files.items() 
                     if any(episode_num == 19 for episode_num, _ in files)}
    
    # Create a dataframe for each subject
    subject_dataframes = {}
    
    for subject_id, file_list in subject_files.items():
        # Sort files by episode number to maintain order
        file_list.sort(key=lambda x: x[0])
        
        # Read and concatenate all CSV files for this subject
        dfs = []
        for episode_num, csv_file in file_list:
            df = pd.read_csv(csv_file)
            
            # Check if 'episode_num' column exists
            if 'episode_num' not in df.columns:
                # Add episode_num column if it doesn't exist
                df['episode_num'] = episode_num
            else:
                # If column exists but is empty/NaN, fill it with the episode number
                if df['episode_num'].isna().all() or (df['episode_num'] == '').all():
                    df['episode_num'] = episode_num
                # If it already has values, we keep them but verify the first row
                elif episode_num == 0:
                    # For base episode, fill any empty values
                    df['episode_num'] = df['episode_num'].fillna(episode_num)
            
            dfs.append(df)
        
        # Concatenate all episodes for this subject, preserving row order
        subject_df = pd.concat(dfs, ignore_index=True)
        subject_dataframes[subject_id] = subject_df
        
        # Save aggregated dataframe to CSV file
        output_file = agg_dir / f"{subject_id}_aggregated.csv"
        subject_df.to_csv(output_file, index=False)
    
    return subject_dataframes


if __name__ == "__main__":
    print("=" * 70)
    print("DATA AGGREGATION")
    print("=" * 70)
    
    # Example usage
    subject_dfs = aggregate_subject_data()
    
    print("=" * 70)
    print("SUBJECT DATA AGGREGATION (Only Completed Subjects)")
    print("=" * 70)
    
    print(f"\nNumber of subjects included: {len(subject_dfs)}")
    print(f"Output directory: {agg_data_dir}")
    
    for subject_id, df in sorted(subject_dfs.items()):
        print(f"\nSubject ID: {subject_id}")
        print(f"  Total rows: {len(df)}")
        print(f"  Episodes: {sorted(df['episode_num'].unique())}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Saved to: {subject_id}_aggregated.csv")
        
        # Show episode distribution
        episode_counts = df['episode_num'].value_counts().sort_index()
        print(f"  Episodes distribution:")
        for ep, count in episode_counts.items():
            print(f"    Episode {int(ep)}: {count} rows")
    
    print("\n" + "=" * 70)