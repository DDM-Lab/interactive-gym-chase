import pandas as pd
from pathlib import Path
from collections import defaultdict

data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\Pilot1-2\human-only-data-pilot-3"
agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\pilot_2_aggregated_data"

# Parameter: "human-only", "AI-only", or "Human-AI"
experiment_type = "human-only"
suffix = "hh" if experiment_type.lower() == "human-only" else "sp"

# Category folder names
CATEGORY_FOLDERS = {
    1: "category_1_failed_to_reach_main_session",
    2: "category_2_failed_to_play_main_session",
    3: "category_3_team_ended_during_main_session",
    4: "category_4_completed_experiment",
    5: "category_5_uncategorized"
}

# Load valid MTurk IDs from file
valid_mturk_ids = set()
mturk_file = Path(__file__).parent.parent / "mturk_ids_unique_to_dir_2.txt"
if mturk_file.exists():
    with open(mturk_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('- '):
                valid_mturk_ids.add(line[2:])
            elif line and not line.startswith('-'):
                valid_mturk_ids.add(line)


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


def categorize_subjects():
    """
    Categorize subjects based on the same logic as completeness_check.py.
    Returns a dictionary with category numbers as keys and lists of subject IDs as values.
    """
    start_scene_dir = Path(data_dir) / f"overcooked_{suffix}_start_scene"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    end_scene_dir = Path(data_dir) / "end_completion_code_scene"
    
    # Get IDs from each scene
    start_scene_ids = set()
    if start_scene_dir.exists():
        for file_path in start_scene_dir.glob("*"):
            id_part = file_path.name.split("_")[0]
            if not valid_mturk_ids or id_part in valid_mturk_ids:
                start_scene_ids.add(id_part)
    
    cramped_room_ids = set()
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            if file_path.is_file():
                id_part = file_path.name.split("_")[0]
                if not valid_mturk_ids or id_part in valid_mturk_ids:
                    cramped_room_ids.add(id_part)
    
    end_scene_ids = set()
    if end_scene_dir.exists():
        for file_path in end_scene_dir.glob("*"):
            id_part = file_path.name.split("_")[0]
            if not valid_mturk_ids or id_part in valid_mturk_ids:
                end_scene_ids.add(id_part)
    
    # Get episode data
    max_episode_per_subject = {}
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*_ep*.csv"):
            filename = file_path.stem
            parts = filename.split("_ep")
            if len(parts) == 2:
                id_part = parts[0]
                if not valid_mturk_ids or id_part in valid_mturk_ids:
                    try:
                        ep_num = int(parts[1])
                        if id_part not in max_episode_per_subject:
                            max_episode_per_subject[id_part] = ep_num
                        else:
                            max_episode_per_subject[id_part] = max(max_episode_per_subject[id_part], ep_num)
                    except ValueError:
                        pass
    
    overall_max_episode = max(max_episode_per_subject.values()) if max_episode_per_subject else 0
    
    # Category 1: Failed to reach main session
    category_1 = sorted([id_str for id_str in start_scene_ids 
                         if id_str not in cramped_room_ids and id_str not in end_scene_ids])
    
    # Category 2: Failed to play main session
    category_2 = sorted([id_str for id_str in start_scene_ids 
                        if id_str in cramped_room_ids and id_str not in end_scene_ids
                        and id_str not in max_episode_per_subject])
    
    # Category 3: Team ended during main session
    category_3 = sorted([id_str for id_str in start_scene_ids 
                        if id_str in cramped_room_ids and id_str not in end_scene_ids
                        and id_str in max_episode_per_subject])
    
    # Category 4: Completed experiment
    category_4 = sorted([id_str for id_str in start_scene_ids 
                        if id_str in max_episode_per_subject 
                        and max_episode_per_subject[id_str] == overall_max_episode])
    
    # Category 5: Uncategorized
    all_categorized = set(category_1 + category_2 + category_3 + category_4)
    category_5 = sorted([id_str for id_str in valid_mturk_ids if id_str not in all_categorized])
    
    return {
        1: category_1,
        2: category_2,
        3: category_3,
        4: category_4,
        5: category_5,
        'max_episode_per_subject': max_episode_per_subject,
        'overall_max_episode': overall_max_episode
    }


def aggregate_subject_data():
    """
    Aggregate data for subjects based on 5 categories:
    Category 1: Failed to reach main session (placeholder)
    Category 2: Failed to play main session (placeholder)
    Category 3: Team ended during main session (aggregate episode data)
    Category 4: Completed experiment (aggregate episode data)
    Category 5: Uncategorized (placeholder)
    
    Returns:
        dict: Dictionary with category info and subject DataFrames
    """
    
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    base_agg_dir = Path(agg_data_dir)
    
    if not cramped_room_dir.exists():
        raise FileNotFoundError(f"Directory not found: {cramped_room_dir}")
    
    # Create base aggregated data directory and category subdirectories
    base_agg_dir.mkdir(parents=True, exist_ok=True)
    for category_num, folder_name in CATEGORY_FOLDERS.items():
        (base_agg_dir / folder_name).mkdir(parents=True, exist_ok=True)
    
    # Get categorized subjects
    categories = categorize_subjects()
    max_episode_per_subject = categories['max_episode_per_subject']
    overall_max_episode = categories['overall_max_episode']
    
    # Dictionary to store all CSV files grouped by subject ID
    subject_files = defaultdict(list)
    
    # Collect all episode CSV files
    for csv_file in sorted(cramped_room_dir.glob("*_ep*.csv")):
        filename = csv_file.stem
        parts = filename.split("_ep")
        if len(parts) != 2:
            continue
        
        subject_id = parts[0]
        if valid_mturk_ids and subject_id not in valid_mturk_ids:
            continue
        
        try:
            episode_num = int(parts[1])
        except ValueError:
            continue
        
        subject_files[subject_id].append((episode_num, csv_file))
    
    subject_dataframes = {}
    
    # Process Category 4: Completed subjects (aggregate episode data)
    for subject_id in categories[4]:
        if subject_id not in subject_files:
            continue
            
        file_list = subject_files[subject_id]
        file_list.sort(key=lambda x: x[0])
        
        dfs = []
        for episode_num, csv_file in file_list:
            df = pd.read_csv(csv_file)
            
            if 'episode_num' not in df.columns:
                df['episode_num'] = episode_num
            else:
                if df['episode_num'].isna().all() or (df['episode_num'] == '').all():
                    df['episode_num'] = episode_num
            
            dfs.append(df)
        
        subject_df = pd.concat(dfs, ignore_index=True)
        subject_df['status'] = 'completed'
        subject_df['category'] = 4
        
        subject_dataframes[subject_id] = subject_df
        
        output_file = base_agg_dir / CATEGORY_FOLDERS[4] / f"{subject_id}_aggregated.csv"
        subject_df.to_csv(output_file, index=False)
    
    # Process Category 3: Team ended during main session (aggregate episode data)
    for subject_id in categories[3]:
        if subject_id not in subject_files:
            continue
            
        file_list = subject_files[subject_id]
        file_list.sort(key=lambda x: x[0])
        
        dfs = []
        for episode_num, csv_file in file_list:
            df = pd.read_csv(csv_file)
            
            if 'episode_num' not in df.columns:
                df['episode_num'] = episode_num
            else:
                if df['episode_num'].isna().all() or (df['episode_num'] == '').all():
                    df['episode_num'] = episode_num
            
            dfs.append(df)
        
        subject_df = pd.concat(dfs, ignore_index=True)
        subject_df['status'] = 'team_ended'
        subject_df['category'] = 3
        max_ep = max_episode_per_subject.get(subject_id, 0)
        subject_df['max_episode_reached'] = max_ep
        
        subject_dataframes[subject_id] = subject_df
        
        output_file = base_agg_dir / CATEGORY_FOLDERS[3] / f"{subject_id}_aggregated.csv"
        subject_df.to_csv(output_file, index=False)
    
    # Process Category 1: Failed to reach main session (placeholder)
    for subject_id in categories[1]:
        placeholder_df = pd.DataFrame({
            'status': ['failed_to_reach_main_session'],
            'category': [1]
        })
        
        subject_dataframes[subject_id] = placeholder_df
        
        output_file = base_agg_dir / CATEGORY_FOLDERS[1] / f"{subject_id}_aggregated.csv"
        placeholder_df.to_csv(output_file, index=False)
    
    # Process Category 2: Failed to play main session (placeholder)
    for subject_id in categories[2]:
        placeholder_df = pd.DataFrame({
            'status': ['failed_to_play_main_session'],
            'category': [2]
        })
        
        subject_dataframes[subject_id] = placeholder_df
        
        output_file = base_agg_dir / CATEGORY_FOLDERS[2] / f"{subject_id}_aggregated.csv"
        placeholder_df.to_csv(output_file, index=False)
    
    # Process Category 5: Uncategorized (placeholder)
    for subject_id in categories[5]:
        placeholder_df = pd.DataFrame({
            'status': ['uncategorized'],
            'category': [5]
        })
        
        subject_dataframes[subject_id] = placeholder_df
        
        output_file = base_agg_dir / CATEGORY_FOLDERS[5] / f"{subject_id}_aggregated.csv"
        placeholder_df.to_csv(output_file, index=False)
    
    return {
        'subject_dataframes': subject_dataframes,
        'categories': categories,
        'overall_max_episode': overall_max_episode
    }


if __name__ == "__main__":
    print("=" * 70)
    print("DATA AGGREGATION")
    print("=" * 70)
    
    result = aggregate_subject_data()
    subject_dfs = result['subject_dataframes']
    categories = result['categories']
    overall_max_episode = result['overall_max_episode']
    
    print("\n" + "=" * 70)
    print("AGGREGATION SUMMARY")
    print("=" * 70)
    print(f"\nMax episode number: {overall_max_episode}")
    print(f"\nOutput directory: {agg_data_dir}")
    
    # Print category summaries
    category_names = {
        1: "FAILED TO REACH MAIN SESSION",
        2: "FAILED TO PLAY MAIN SESSION",
        3: "TEAM ENDED DURING MAIN SESSION",
        4: "COMPLETED EXPERIMENT",
        5: "UNCATEGORIZED"
    }
    
    for cat_num in range(1, 6):
        cat_ids = categories[cat_num]
        print(f"\n{'=' * 70}")
        print(f"CATEGORY {cat_num}: {category_names[cat_num]}")
        print(f"{'=' * 70}")
        print(f"Count: {len(cat_ids)}")
        print(f"Folder: {CATEGORY_FOLDERS[cat_num]}")
        
        if cat_num in [3, 4]:  # Categories with actual episode data
            print(f"Data type: Aggregated episode data")
            if cat_ids:
                print("\nSubject IDs:")
                for subject_id in cat_ids:
                    if subject_id in subject_dfs:
                        df = subject_dfs[subject_id]
                        print(f"  - {subject_id}")
                        if 'episode_num' in df.columns:
                            episodes = sorted(df['episode_num'].unique())
                            print(f"      Episodes: {episodes}")
                            print(f"      Total rows: {len(df)}")
        else:  # Categories with placeholders
            print(f"Data type: Placeholder (single-row)")
            if cat_ids:
                print(f"\nSubject IDs: {', '.join(cat_ids)}")
    
    print("\n" + "=" * 70)
    print(f"TOTAL SUBJECTS PROCESSED: {len(subject_dfs)}")
    print("=" * 70)

