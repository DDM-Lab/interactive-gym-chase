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


def identify_team_pairs(base_agg_dir, categories):
    """
    Identify team pairs by reading player_subjects.0 and player_subjects.1 from CSV files.
    Returns a dictionary mapping each subject_id to their partner's subject_id.
    """
    team_pairs = {}  # Maps subject_id -> partner_subject_id
    
    # Only process categories 3 and 4
    for category_num in [3, 4]:
        category_dir = base_agg_dir / CATEGORY_FOLDERS[category_num]
        
        for subject_id in categories[category_num]:
            csv_file = category_dir / f"{subject_id}_aggregated.csv"
            
            if not csv_file.exists():
                continue
            
            # Read first row to get player_subjects
            df = pd.read_csv(csv_file, nrows=1)
            
            if 'player_subjects.0' in df.columns and 'player_subjects.1' in df.columns:
                player_0 = df['player_subjects.0'].iloc[0]
                player_1 = df['player_subjects.1'].iloc[0]
                
                # Determine which one is the current subject and which is the partner
                if player_0 == subject_id:
                    partner_id = player_1
                elif player_1 == subject_id:
                    partner_id = player_0
                else:
                    continue
                
                team_pairs[subject_id] = partner_id
    
    return team_pairs


def align_team_timesteps(base_agg_dir, categories):
    """
    Align timesteps between team pairs for categories 3 and 4.
    Creates combined CSV files with consensus values from both agents.
    """
    team_pairs = identify_team_pairs(base_agg_dir, categories)
    
    if not team_pairs:
        return
    
    # Create output directory
    aligned_dir = base_agg_dir / "aligned_team_data"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    processed_pairs = set()
    alignment_stats = []
    
    for subject_id, partner_id in team_pairs.items():
        pair_key = tuple(sorted([subject_id, partner_id]))
        if pair_key in processed_pairs:
            continue
        processed_pairs.add(pair_key)
        
        # Determine categories
        subject_category = None
        partner_category = None
        
        for cat_num in [3, 4]:
            if subject_id in categories[cat_num]:
                subject_category = cat_num
            if partner_id in categories[cat_num]:
                partner_category = cat_num
        
        if subject_category is None or partner_category is None:
            continue
        
        # Load both CSV files
        subject_file = base_agg_dir / CATEGORY_FOLDERS[subject_category] / f"{subject_id}_aggregated.csv"
        partner_file = base_agg_dir / CATEGORY_FOLDERS[partner_category] / f"{partner_id}_aggregated.csv"
        
        if not subject_file.exists() or not partner_file.exists():
            continue
        
        df_subject = pd.read_csv(subject_file)
        df_partner = pd.read_csv(partner_file)
        
        # Get common episodes
        episodes_subject = sorted(df_subject['episode_num'].unique())
        episodes_partner = sorted(df_partner['episode_num'].unique())
        common_episodes = set(episodes_subject).intersection(set(episodes_partner))
        
        if not common_episodes:
            continue
        
        # Find all reward columns
        reward_cols = [col for col in df_subject.columns if 'reward' in col.lower()]
        
        combined_episodes = []
        
        for episode in sorted(common_episodes):
            ep_subject = df_subject[df_subject['episode_num'] == episode].copy()
            ep_partner = df_partner[df_partner['episode_num'] == episode].copy()
            
            # Find shared timesteps
            t_subject = set(ep_subject['t'].values)
            t_partner = set(ep_partner['t'].values)
            shared_t = t_subject.intersection(t_partner)
            
            # Filter and deduplicate (keep last occurrence)
            ep_subject_aligned = ep_subject[ep_subject['t'].isin(shared_t)].sort_values('t')
            ep_subject_aligned = ep_subject_aligned.drop_duplicates(subset=['t'], keep='last').reset_index(drop=True)
            
            ep_partner_aligned = ep_partner[ep_partner['t'].isin(shared_t)].sort_values('t')
            ep_partner_aligned = ep_partner_aligned.drop_duplicates(subset=['t'], keep='last').reset_index(drop=True)
            
            # Sort by 't' for alignment
            ep_subject_aligned = ep_subject_aligned.sort_values('t').reset_index(drop=True)
            ep_partner_aligned = ep_partner_aligned.sort_values('t').reset_index(drop=True)
            
            # Determine which agent is more accurate based on reward sums
            subject_reward_sum = sum(ep_subject_aligned[col].sum() for col in reward_cols if col in ep_subject_aligned.columns)
            partner_reward_sum = sum(ep_partner_aligned[col].sum() for col in reward_cols if col in ep_partner_aligned.columns)
            
            more_accurate_is_subject = subject_reward_sum >= partner_reward_sum
            
            # Create consensus dataframe
            consensus_df = ep_subject_aligned.copy()
            
            # Align all reward columns (take maximum)
            for col in reward_cols:
                if col in ep_subject_aligned.columns and col in ep_partner_aligned.columns:
                    consensus_df[col] = pd.concat([ep_subject_aligned[col], ep_partner_aligned[col]], axis=1).max(axis=1)
            
            # For non-reward columns, use more accurate agent's values when they differ
            non_reward_cols = [col for col in consensus_df.columns if col not in reward_cols and col != 't']
            
            for col in non_reward_cols:
                if col not in ep_partner_aligned.columns:
                    continue
                
                # Check if values differ
                if not ep_subject_aligned[col].equals(ep_partner_aligned[col]):
                    if more_accurate_is_subject:
                        consensus_df[col] = ep_subject_aligned[col]
                    else:
                        consensus_df[col] = ep_partner_aligned[col]
            
            combined_episodes.append(consensus_df)
        
        # Combine all episodes
        if combined_episodes:
            combined_df = pd.concat(combined_episodes, ignore_index=True)
            
            # Save combined file
            output_file = aligned_dir / f"team_{subject_id}_{partner_id}_aligned.csv"
            combined_df.to_csv(output_file, index=False)
            
            alignment_stats.append({
                'team': f"{subject_id}_{partner_id}",
                'episodes': len(common_episodes),
                'total_rows': len(combined_df)
            })
    
    return alignment_stats


def report_team_rewards():
    """
    Examine aligned_team_data and report, per episode, per team,
    what each team scored across each of the reward columns.
    """
    aligned_dir = Path(agg_data_dir) / "aligned_team_data"
    
    if not aligned_dir.exists():
        return
    
    print("\n" + "=" * 100)
    print("TEAM REWARD BREAKDOWN PER EPISODE (Categories 3 & 4)")
    print("=" * 100)
    
    # Process all aligned team files
    for team_file in sorted(aligned_dir.glob("team_*_aligned.csv")):
        df = pd.read_csv(team_file)
        
        # Extract team info from filename
        team_id = team_file.stem.replace("_aligned", "")
        
        # Get category from first row
        if 'category' in df.columns:
            category = df['category'].iloc[0]
            category_name = "quit" if category == 3 else "completed" if category == 4 else "unknown"
        else:
            category_name = "unknown"
        
        print(f"\n{team_id} ({category_name}):")
        
        # Find all reward columns
        reward_cols = [col for col in df.columns if 'reward' in col.lower()]
        
        if 'episode_num' not in df.columns:
            continue
        
        # Group by episode and sum rewards
        episodes = sorted(df['episode_num'].unique())
        
        for episode in episodes:
            episode_data = df[df['episode_num'] == episode]
            print(f"  Episode {int(episode)}:")
            
            for reward_col in reward_cols:
                if reward_col in episode_data.columns:
                    reward_sum = episode_data[reward_col].sum()
                    if reward_sum != 0:  # Only print non-zero rewards
                        print(f"    {reward_col}: {reward_sum}")
        
        # Print total across all episodes
        print(f"  Total Across All Episodes:")
        for reward_col in reward_cols:
            if reward_col in df.columns:
                reward_sum = df[reward_col].sum()
                if reward_sum != 0:  # Only print non-zero rewards
                    print(f"    {reward_col}: {reward_sum}")
    
    print("\n" + "=" * 100)


if __name__ == "__main__":
    result = aggregate_subject_data()
    subject_dfs = result['subject_dataframes']
    categories = result['categories']
    overall_max_episode = result['overall_max_episode']
    
    # Perform time alignment analysis for categories 3 and 4
    alignment_stats = align_team_timesteps(Path(agg_data_dir), categories)
    
    # Report team rewards from aligned data
    report_team_rewards()

