import pandas as pd
from pathlib import Path
from collections import defaultdict

data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\Pilot1-2\human-only-data-pilot-1"
agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data"

# Parameter: "human-only", "AI-only", or "Human-AI"
experiment_type = "human-only"
suffix = "hh" if experiment_type.lower() == "human-only" else "sp"


def get_max_episode():
    """
    Dynamically determine the max episode number from the data.
    """
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    max_ep = 0
    
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*_ep*.csv"):
            filename = file_path.stem
            parts = filename.split("_ep")
            if len(parts) == 2:
                try:
                    ep_num = int(parts[1])
                    max_ep = max(max_ep, ep_num)
                except ValueError:
                    pass
    
    return max_ep


def get_unpaired_subjects():
    """
    Get IDs that are in start_scene but not in cramped_room.
    """
    start_scene_dir = Path(data_dir) / f"overcooked_{suffix}_start_scene"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    start_scene_ids = set()
    cramped_room_ids = set()
    
    if start_scene_dir.exists():
        for file_path in start_scene_dir.glob("*"):
            id_part = file_path.name.split("_")[0]
            start_scene_ids.add(id_part)
    
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            id_part = file_path.name.split("_")[0]
            cramped_room_ids.add(id_part)
    
    unpaired_ids = sorted(list(start_scene_ids - cramped_room_ids))
    return unpaired_ids


def aggregate_subject_data():
    """
    Aggregate data for two groups of subjects:
    (1) Subjects who fully completed the experiment (reached max episode)
    (2) Subjects who were unpaired (in start_scene but not in cramped_room)
    
    Each aggregated file includes a "status" column indicating:
    - "completed" for subjects who reached the max episode
    - "unpaired" for subjects who didn't get paired
    
    Returns:
        dict: Dictionary with subject IDs as keys and DataFrames as values
    """
    
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    agg_dir = Path(agg_data_dir)
    
    if not cramped_room_dir.exists():
        raise FileNotFoundError(f"Directory not found: {cramped_room_dir}")
    
    # Create aggregated data directory if it doesn't exist
    agg_dir.mkdir(parents=True, exist_ok=True)
    
    # Get max episode and unpaired subjects
    max_episode = get_max_episode()
    unpaired_ids = get_unpaired_subjects()
    
    # Dictionary to store CSV files grouped by subject ID
    subject_files = defaultdict(list)
    
    # Iterate through all episode CSV files and group by subject ID
    for csv_file in sorted(cramped_room_dir.glob("*_ep*.csv")):
        filename = csv_file.stem  # Remove .csv extension
        
        # Extract subject ID and episode number from pattern like "ID_ep5"
        parts = filename.split("_ep")
        if len(parts) != 2:
            continue
        
        subject_id = parts[0]
        try:
            episode_num = int(parts[1])
        except ValueError:
            continue
        
        subject_files[subject_id].append((episode_num, csv_file))
    
    # Filter to only keep subjects who completed the max episode
    subject_files = {subject_id: files for subject_id, files in subject_files.items() 
                     if any(episode_num == max_episode for episode_num, _ in files)}
    
    # Create aggregated dataframes
    subject_dataframes = {}
    
    # (1) Process completed subjects
    for subject_id, file_list in subject_files.items():
        # Sort files by episode number
        file_list.sort(key=lambda x: x[0])
        
        # Read and concatenate all CSV files for this subject
        dfs = []
        for episode_num, csv_file in file_list:
            df = pd.read_csv(csv_file)
            
            # Ensure episode_num column exists and is populated
            if 'episode_num' not in df.columns:
                df['episode_num'] = episode_num
            else:
                if df['episode_num'].isna().all() or (df['episode_num'] == '').all():
                    df['episode_num'] = episode_num
            
            dfs.append(df)
        
        # Concatenate all episodes for this subject
        subject_df = pd.concat(dfs, ignore_index=True)
        
        # Add status column
        subject_df['status'] = 'completed'
        
        subject_dataframes[subject_id] = subject_df
        
        # Save aggregated dataframe
        output_file = agg_dir / f"{subject_id}_aggregated.csv"
        subject_df.to_csv(output_file, index=False)
    
    # (2) Create entries for unpaired subjects
    for subject_id in unpaired_ids:
        # Create a single-row dataframe with status="unpaired" and NaN for other columns
        unpaired_df = pd.DataFrame({
            'status': ['unpaired']
        })
        
        subject_dataframes[subject_id] = unpaired_df
        
        # Save unpaired subject dataframe
        output_file = agg_dir / f"{subject_id}_aggregated.csv"
        unpaired_df.to_csv(output_file, index=False)
    
    return subject_dataframes, max_episode, unpaired_ids


if __name__ == "__main__":
    print("=" * 70)
    print("DATA AGGREGATION")
    print("=" * 70)
    
    subject_dfs, max_episode, unpaired_ids = aggregate_subject_data()
    
    print("=" * 70)
    print("SUBJECT DATA AGGREGATION")
    print("=" * 70)
    
    # Separate completed and unpaired subjects
    completed_subjects = [id for id in subject_dfs.keys() if id not in unpaired_ids]
    
    print(f"\nMax episode number: {max_episode}")
    print(f"\nCompleted subjects (reached episode {max_episode}): {len(completed_subjects)}")
    print(f"Unpaired subjects (not in cramped_room): {len(unpaired_ids)}")
    print(f"Total subjects included: {len(subject_dfs)}")
    print(f"\nOutput directory: {agg_data_dir}")
    
    # Show completed subjects
    if completed_subjects:
        print("\n" + "-" * 70)
        print("COMPLETED SUBJECTS")
        print("-" * 70)
        for subject_id in sorted(completed_subjects):
            df = subject_dfs[subject_id]
            print(f"\nSubject ID: {subject_id}")
            print(f"  Status: completed")
            print(f"  Total rows: {len(df)}")
            if 'episode_num' in df.columns:
                print(f"  Episodes: {sorted(df['episode_num'].unique())}")
            print(f"  Columns: {len(df.columns)}")
    
    # Show unpaired subjects
    if unpaired_ids:
        print("\n" + "-" * 70)
        print("UNPAIRED SUBJECTS")
        print("-" * 70)
        for subject_id in unpaired_ids:
            print(f"  - {subject_id}")
    
    print("\n" + "=" * 70)

