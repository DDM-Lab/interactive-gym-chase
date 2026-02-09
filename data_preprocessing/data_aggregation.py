import pandas as pd
from pathlib import Path
from collections import defaultdict
import csv
from quit_initiator_detector import detect_quit_initiator

#data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-data_run_1"
#data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-run-1-aws"
agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\post_pilot"
data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-post-pilot"

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

def extract_subject_id(filename):
    """
    Extract subject ID from filename, handling multi-underscore IDs like 'grace_t_6'.
    For episode files (e.g., 'grace_t_6_ep0.csv'), splits on '_ep' to get the ID.
    For other files (e.g., 'grace_t_6_metadata.json'), removes known suffixes.
    """
    # For episode files, split on '_ep'
    if '_ep' in filename:
        id_part = filename.split('_ep')[0]
    else:
        # Remove known suffixes for non-episode files
        known_suffixes = ['_metadata', '_globals', '_multiplayer_metrics']
        id_part = filename
        for suffix in known_suffixes:
            if id_part.endswith(suffix):
                id_part = id_part[:-len(suffix)]
                break
    return id_part

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


def build_early_team_mapping():
    """
    Identify team pairs early by reading player_subjects from CSV files.
    This needs to happen before category assignment so we can use team info for categorization.
    Returns: dict mapping subject_id -> partner_id
    """
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    id_to_teammate = {}
    
    if cramped_room_dir.exists():
        processed_ids = set()
        
        # Read one CSV per subject to extract team pair
        for csv_file in sorted(cramped_room_dir.glob("*_ep0.csv")):
            filename = csv_file.stem
            id_part = filename.split("_ep")[0]
            
            if id_part in processed_ids:
                continue
            processed_ids.add(id_part)
            
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    first_row = next(reader, None)
                    if first_row:
                        player_0 = first_row.get("player_subjects.0", "").strip()
                        player_1 = first_row.get("player_subjects.1", "").strip()
                        if player_0 and player_1:
                            id_to_teammate[player_0] = player_1
                            id_to_teammate[player_1] = player_0
            except Exception:
                pass
    
    return id_to_teammate


def categorize_subjects(id_to_teammate=None):
    """
    Categorize subjects based on the same logic as completeness_check.py.
    Enhanced: If a subject's partner completed all episodes, the subject is moved to 
    Category 4 (completed) even if their own episode data is incomplete due to technical issues.
    
    Args:
        id_to_teammate: Optional dict mapping subject_id -> partner_id for team-aware categorization
    
    Returns:
        dict with category numbers as keys and lists of subject IDs as values.
    """
    if id_to_teammate is None:
        id_to_teammate = {}
    
    start_scene_dir = Path(data_dir) / f"overcooked_{suffix}_start_scene"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    end_scene_dir = Path(data_dir) / "end_completion_code_scene"
    
    # Get IDs from each scene
    start_scene_ids = set()
    if start_scene_dir.exists():
        for file_path in start_scene_dir.glob("*"):
            id_part = extract_subject_id(file_path.stem)
            start_scene_ids.add(id_part)
    
    cramped_room_ids = set()
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            if file_path.is_file():
                id_part = extract_subject_id(file_path.stem)
                cramped_room_ids.add(id_part)
    
    end_scene_ids = set()
    if end_scene_dir.exists():
        for file_path in end_scene_dir.glob("*"):
            id_part = extract_subject_id(file_path.stem)
            end_scene_ids.add(id_part)
    
    # Get episode data
    max_episode_per_subject = {}
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*_ep*.csv"):
            filename = file_path.stem
            parts = filename.split("_ep")
            if len(parts) == 2:
                id_part = parts[0]
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
    
    # Category 4: Completed experiment
    # Enhanced logic: Include subjects whose partner completed all episodes
    category_4 = []
    for id_str in start_scene_ids:
        # Own data complete
        if id_str in max_episode_per_subject and max_episode_per_subject[id_str] == overall_max_episode:
            category_4.append(id_str)
        # Own data incomplete, but partner completed all episodes
        elif id_str in id_to_teammate:
            partner_id = id_to_teammate[id_str]
            if partner_id in max_episode_per_subject and max_episode_per_subject[partner_id] == overall_max_episode:
                category_4.append(id_str)
    
    category_4 = sorted(category_4)
    
    # Category 3: Team ended during main session
    # Enhanced logic: Only if BOTH teammates have incomplete data
    category_3 = []
    for id_str in start_scene_ids:
        # Not in category 1 or 2, and not in category 4
        if id_str not in category_1 and id_str not in category_2 and id_str not in category_4:
            # Must have some episode data
            if id_str in cramped_room_ids and id_str in max_episode_per_subject:
                # If subject is in a team, check partner status
                if id_str in id_to_teammate:
                    partner_id = id_to_teammate[id_str]
                    # Only add to category 3 if partner also has incomplete data
                    if partner_id in max_episode_per_subject and max_episode_per_subject[partner_id] < overall_max_episode:
                        category_3.append(id_str)
                    # If partner not in data at all, still add to category 3
                    elif partner_id not in max_episode_per_subject:
                        category_3.append(id_str)
                else:
                    # Not in a team pair, but has incomplete data
                    category_3.append(id_str)
    
    category_3 = sorted(category_3)
    
    # Category 5: Uncategorized
    all_categorized = set(category_1 + category_2 + category_3 + category_4)
    # Build universe of IDs from discovered sources
    all_ids_universe = start_scene_ids.union(cramped_room_ids).union(end_scene_ids).union(set(max_episode_per_subject.keys()))
    category_5 = sorted([id_str for id_str in all_ids_universe if id_str not in all_categorized])
    
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
    
    # Build team mapping FIRST, before categorization
    id_to_teammate = build_early_team_mapping()
    
    # Get categorized subjects with team-aware logic
    categories = categorize_subjects(id_to_teammate=id_to_teammate)
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


def get_quitter_for_team(player_0, player_1, console_logs_dir):
    """
    Attempt to determine which player quit first by analyzing console logs.
    
    Args:
        player_0: First player's subject ID
        player_1: Second player's subject ID
        console_logs_dir: Path to directory containing console.jsonl files
    
    Returns:
        str: The subject ID of the quitter, or "unknown" if not found
    """
    console_logs_dir = Path(console_logs_dir)
    
    # Build paths to console logs
    log_file_1 = console_logs_dir / f"{player_0}_console.jsonl"
    log_file_2 = console_logs_dir / f"{player_1}_console.jsonl"
    
    # Check if both files exist
    if not log_file_1.exists() or not log_file_2.exists():
        return "unknown"
    
    try:
        quitter, details = detect_quit_initiator(str(log_file_1), str(log_file_2))
        return quitter if quitter else "unknown"
    except Exception:
        return "unknown"


def align_team_timesteps(base_agg_dir, categories):
    """
    Align timesteps between team pairs for categories 3 and 4.
    For shared episodes: uses alignment logic to combine data from both agents.
    For episodes only one agent has: uses that agent's data wholesale (no alignment).
    Adds quitter_id column for Category 3 teams (quit teams).
    """
    team_pairs = identify_team_pairs(base_agg_dir, categories)
    
    if not team_pairs:
        return
    
    # Create output directory
    aligned_dir = base_agg_dir / "aligned_team_data"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine console logs directory
    console_logs_dir = Path(data_dir) / "console_logs"
    
    processed_pairs = set()
    alignment_stats = []
    episode_coverage_report = []
    
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
        
        # Get episodes for each agent
        episodes_subject = sorted(df_subject['episode_num'].unique())
        episodes_partner = sorted(df_partner['episode_num'].unique())
        common_episodes = set(episodes_subject).intersection(set(episodes_partner))
        subject_only_episodes = set(episodes_subject) - set(episodes_partner)
        partner_only_episodes = set(episodes_partner) - set(episodes_subject)
        
        if not common_episodes and not subject_only_episodes and not partner_only_episodes:
            continue
        
        # Find all reward columns
        reward_cols = [col for col in df_subject.columns if 'reward' in col.lower()]
        
        combined_episodes = []
        
        # Process common episodes with alignment logic
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
        
        # Process subject-only episodes (use subject's data wholesale)
        for episode in sorted(subject_only_episodes):
            ep_subject = df_subject[df_subject['episode_num'] == episode].copy()
            combined_episodes.append(ep_subject)
        
        # Process partner-only episodes (use partner's data wholesale)
        for episode in sorted(partner_only_episodes):
            ep_partner = df_partner[df_partner['episode_num'] == episode].copy()
            combined_episodes.append(ep_partner)
        
        # Combine all episodes
        if combined_episodes:
            combined_df = pd.concat(combined_episodes, ignore_index=True)
            combined_df = combined_df.sort_values('episode_num').reset_index(drop=True)
            
            # Add quitter_id column for Category 3 teams (quit teams)
            if subject_category == 3:
                quitter_id = get_quitter_for_team(subject_id, partner_id, console_logs_dir)
                combined_df['quitter_id'] = quitter_id
            
            # Save combined file
            output_file = aligned_dir / f"team_{subject_id}_{partner_id}_aligned.csv"
            combined_df.to_csv(output_file, index=False)
            
            # Log episode coverage
            if subject_only_episodes or partner_only_episodes:
                coverage_info = {
                    'team': f"{subject_id}_{partner_id}",
                    'common_episodes': len(common_episodes),
                    'subject_only_episodes': sorted(list(subject_only_episodes)),
                    'partner_only_episodes': sorted(list(partner_only_episodes)),
                    'subject_id': subject_id,
                    'partner_id': partner_id
                }
                episode_coverage_report.append(coverage_info)
            
            alignment_stats.append({
                'team': f"{subject_id}_{partner_id}",
                'common_episodes': len(common_episodes),
                'subject_only_episodes': len(subject_only_episodes),
                'partner_only_episodes': len(partner_only_episodes),
                'total_episodes': len(set(episodes_subject).union(set(episodes_partner))),
                'total_rows': len(combined_df)
            })
    
    return alignment_stats, episode_coverage_report


def report_episode_coverage(episode_coverage_report, categories):
    """
    Report on teams that had episode coverage disparities between teammates.
    Includes category (quit/completed) and episode counts for both agents.
    """
    if not episode_coverage_report:
        return
    
    print("\n" + "=" * 100)
    print("EPISODE COVERAGE DISPARITIES (Team Pairs with Unequal Episode Data)")
    print("=" * 100)
    
    for coverage in episode_coverage_report:
        team = coverage['team']
        subject_id = coverage['subject_id']
        partner_id = coverage['partner_id']
        common = coverage['common_episodes']
        subject_only = coverage['subject_only_episodes']
        partner_only = coverage['partner_only_episodes']
        
        # Determine category
        team_category = None
        category_name = "unknown"
        if subject_id in categories[3]:
            team_category = 3
            category_name = "quit"
        elif subject_id in categories[4]:
            team_category = 4
            category_name = "completed"
        elif partner_id in categories[3]:
            team_category = 3
            category_name = "quit"
        elif partner_id in categories[4]:
            team_category = 4
            category_name = "completed"
        
        # Calculate total episodes for each agent
        subject_total_episodes = common + len(subject_only)
        partner_total_episodes = common + len(partner_only)
        
        print(f"\n{team} [{category_name} (Category {team_category})]:")
        print(f"  {subject_id}: Episodes 0-{subject_total_episodes - 1} (total: {subject_total_episodes})")
        print(f"  {partner_id}: Episodes 0-{partner_total_episodes - 1} (total: {partner_total_episodes})")
        print(f"  Common episodes (aligned): {common}")
        
        if subject_only:
            print(f"  Episodes only in {subject_id}: {subject_only}")
        
        if partner_only:
            print(f"  Episodes only in {partner_id}: {partner_only}")
    
    print("\n" + "=" * 100)


def validate_aligned_data(episode_coverage_report, categories):
    """
    Validate aligned team data CSV files for data integrity.
    Checks:
    1. Episode continuity (sequential, no gaps/duplicates)
    2. Timestamp ordering within episodes
    3. No duplicate timesteps within episodes
    4. Wholesale data matches source files
    """
    aligned_dir = Path(agg_data_dir) / "aligned_team_data"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    if not aligned_dir.exists():
        print("\nNo aligned_team_data directory found. Skipping validation.")
        return
    
    # Create a map of coverage for easy lookup
    coverage_map = {}
    if episode_coverage_report:
        for coverage in episode_coverage_report:
            coverage_map[coverage['team']] = coverage
    
    # Collect agent episode data for wholesale verification
    agent_episodes = {}  # Maps (agent_id, episode_num) -> source_df
    for csv_file in sorted(cramped_room_dir.glob("*_ep*.csv")):
        filename = csv_file.stem
        parts = filename.split("_ep")
        if len(parts) == 2:
            agent_id = parts[0]
            try:
                episode_num = int(parts[1])
                agent_episodes[(agent_id, episode_num)] = pd.read_csv(csv_file)
            except ValueError:
                pass
    
    print("\n" + "=" * 120)
    print("ALIGNED DATA VALIDATION REPORT")
    print("=" * 120)
    
    validation_summary = {
        'total_teams': 0,
        'teams_passed': 0,
        'teams_failed': 0,
        'issues': []
    }
    
    for team_file in sorted(aligned_dir.glob("team_*_aligned.csv")):
        team_id = team_file.stem.replace("_aligned", "")
        validation_summary['total_teams'] += 1
        
        df = pd.read_csv(team_file)
        team_issues = []
        
        # Parse team ID to get agent IDs
        parts = team_id.replace("team_", "").split("_")
        if len(parts) >= 2:
            agent_0 = "_".join(parts[:-1])  # Everything except last part
            agent_1 = parts[-1]
        else:
            agent_0 = agent_1 = "unknown"
        
        # Get coverage info if available
        coverage = coverage_map.get(team_id, None)
        
        # ===== CHECK 1: Episode Continuity =====
        if 'episode_num' in df.columns:
            episodes = sorted(df['episode_num'].unique())
            expected_episodes = list(range(int(episodes[0]), int(episodes[-1]) + 1))
            actual_episodes = [int(e) for e in episodes]
            
            if actual_episodes != expected_episodes:
                team_issues.append(f"Episode continuity FAILED: Expected {expected_episodes}, got {actual_episodes}")
            else:
                team_issues.append("✓ Episode continuity: PASS")
        else:
            team_issues.append("⚠ 'episode_num' column not found")
        
        # ===== CHECK 2: Timestamp Ordering =====
        if 'episode_num' in df.columns and 't' in df.columns:
            ordering_pass = True
            for episode in sorted(df['episode_num'].unique()):
                episode_data = df[df['episode_num'] == episode].sort_values('t')
                t_values = episode_data['t'].values
                # Check if t is ordered
                if not all(t_values[i] <= t_values[i+1] for i in range(len(t_values)-1)):
                    team_issues.append(f"Timestamp ordering FAILED in episode {int(episode)}: t values not ordered")
                    ordering_pass = False
                    break
            
            if ordering_pass:
                team_issues.append("✓ Timestamp ordering: PASS")
        else:
            team_issues.append("⚠ 'episode_num' or 't' column not found")
        
        # ===== CHECK 3: No Duplicate Timesteps =====
        if 'episode_num' in df.columns and 't' in df.columns:
            duplicates_pass = True
            for episode in sorted(df['episode_num'].unique()):
                episode_data = df[df['episode_num'] == episode]
                if episode_data.duplicated(subset=['t']).any():
                    dup_t_values = episode_data[episode_data.duplicated(subset=['t'], keep=False)]['t'].values
                    team_issues.append(f"Duplicate timesteps FAILED in episode {int(episode)}: t={dup_t_values}")
                    duplicates_pass = False
                    break
            
            if duplicates_pass:
                team_issues.append("✓ No duplicate timesteps: PASS")
        else:
            team_issues.append("⚠ 'episode_num' or 't' column not found")
        
        # ===== CHECK 4: Wholesale Data Correctness =====
        wholesale_pass = True
        if coverage:
            subject_id = coverage['subject_id']
            partner_id = coverage['partner_id']
            subject_only = coverage['subject_only_episodes']
            partner_only = coverage['partner_only_episodes']
            
            # Check subject-only episodes
            for ep in subject_only:
                if (subject_id, ep) in agent_episodes:
                    source_df = agent_episodes[(subject_id, ep)]
                    aligned_ep_df = df[df['episode_num'] == ep]
                    
                    if len(aligned_ep_df) != len(source_df):
                        team_issues.append(f"Wholesale data FAILED for {subject_id} episode {int(ep)}: Row count mismatch (aligned={len(aligned_ep_df)}, source={len(source_df)})")
                        wholesale_pass = False
                    # Could add more detailed column checks here if needed
            
            # Check partner-only episodes
            for ep in partner_only:
                if (partner_id, ep) in agent_episodes:
                    source_df = agent_episodes[(partner_id, ep)]
                    aligned_ep_df = df[df['episode_num'] == ep]
                    
                    if len(aligned_ep_df) != len(source_df):
                        team_issues.append(f"Wholesale data FAILED for {partner_id} episode {int(ep)}: Row count mismatch (aligned={len(aligned_ep_df)}, source={len(source_df)})")
                        wholesale_pass = False
            
            if wholesale_pass:
                team_issues.append("✓ Wholesale data correctness: PASS")
        else:
            team_issues.append("✓ Wholesale data correctness: N/A (team has no disparities)")
        
        # ===== Print Results =====
        print(f"\n{team_id}:")
        for issue in team_issues:
            print(f"  {issue}")
        
        # Update summary
        if any("FAILED" in issue for issue in team_issues):
            validation_summary['teams_failed'] += 1
            validation_summary['issues'].extend([(team_id, issue) for issue in team_issues if "FAILED" in issue])
        else:
            validation_summary['teams_passed'] += 1
    
    # ===== Summary =====
    print("\n" + "=" * 120)
    print("VALIDATION SUMMARY")
    print("=" * 120)
    print(f"Total teams: {validation_summary['total_teams']}")
    print(f"Teams passed: {validation_summary['teams_passed']}")
    print(f"Teams failed: {validation_summary['teams_failed']}")
    
    if validation_summary['issues']:
        print(f"\n⚠ {len(validation_summary['issues'])} issues found:")
        for team_id, issue in validation_summary['issues']:
            print(f"  [{team_id}] {issue}")
    else:
        print("\n✓ All validations passed!")
    
    print("=" * 120 + "\n")


def report_team_rewards(episode_coverage_report):
    """
    Report team rewards in a condensed table format.
    Columns: team_id, category, rewards.0, rewards.1, pair_matched
    """
    aligned_dir = Path(agg_data_dir) / "aligned_team_data"
    
    if not aligned_dir.exists():
        return
    
    # Create set of teams with coverage disparities
    mismatched_teams = set()
    if episode_coverage_report:
        for coverage in episode_coverage_report:
            mismatched_teams.add(coverage['team'])
    
    # Collect data for all teams
    team_data = []
    
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
        
        # Find reward columns for agents
        rewards_0_cols = [col for col in df.columns if 'reward' in col.lower() and '.0' in col]
        rewards_1_cols = [col for col in df.columns if 'reward' in col.lower() and '.1' in col]
        
        # Sum rewards across all episodes
        rewards_0_total = sum(df[col].sum() for col in rewards_0_cols if col in df.columns)
        rewards_1_total = sum(df[col].sum() for col in rewards_1_cols if col in df.columns)
        
        # Check if pair was matched
        pair_matched = "T" if team_id not in mismatched_teams else "F"
        
        team_data.append({
            'team_id': team_id,
            'category': category_name,
            'rewards.0': rewards_0_total,
            'rewards.1': rewards_1_total,
            'pair_matched': pair_matched
        })
    
    # Create and print table
    if team_data:
        team_df = pd.DataFrame(team_data)
        
        print("\n" + "=" * 130)
        print("TEAM REWARD SUMMARY (Categories 3 & 4)")
        print("=" * 130)
        print(team_df.to_string(index=False))
        print("=" * 130 + "\n")
    else:
        print("\nNo aligned team data found.\n")


if __name__ == "__main__":
    result = aggregate_subject_data()
    subject_dfs = result['subject_dataframes']
    categories = result['categories']
    overall_max_episode = result['overall_max_episode']
    
    # Perform time alignment analysis for categories 3 and 4
    alignment_stats, episode_coverage_report = align_team_timesteps(Path(agg_data_dir), categories)
    
    # Report episode coverage disparities with category information
    report_episode_coverage(episode_coverage_report, categories)
    
    # Report team rewards from aligned data
    report_team_rewards(episode_coverage_report)
    
    # Validate aligned data integrity
    validate_aligned_data(episode_coverage_report, categories)

