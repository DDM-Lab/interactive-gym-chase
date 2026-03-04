import pandas as pd
from pathlib import Path
from collections import defaultdict

# Configuration
agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\pilot_2_aggregated_data\category_4_completed_experiment"

def identify_team_pairs():
    """
    Identify which subjects are paired in teams by examining the first row of each aggregated file.
    Returns a dictionary mapping team identifiers to pairs of subject IDs.
    """
    category_4_dir = Path(agg_data_dir)
    
    if not category_4_dir.exists():
        raise FileNotFoundError(f"Directory not found: {category_4_dir}")
    
    # Dictionary to store team information
    # Key: (agent_0_id, agent_1_id) from first row, Value: subject_id (file owner)
    team_mapping = defaultdict(list)
    subject_data = {}
    
    # Read all aggregated files
    for file_path in sorted(category_4_dir.glob("*_aggregated.csv")):
        subject_id = file_path.stem.replace("_aggregated", "")
        
        try:
            df = pd.read_csv(file_path)
            
            if len(df) == 0:
                print(f"Warning: Empty file for subject {subject_id}")
                continue
            
            # Get the first row to identify agents
            first_row = df.iloc[0]
            
            # Look for columns that might identify the agents
            # Common patterns: 'agents.0', 'agents.1', 'agent_0', 'agent_1', 'player_0', 'player_1'
            agent_cols = [col for col in df.columns if 'agent' in col.lower() or 'player' in col.lower()]
            
            if len(agent_cols) >= 2:
                # Extract agent identifiers
                agent_0 = str(first_row[agent_cols[0]])
                agent_1 = str(first_row[agent_cols[1]])
                team_key = tuple(sorted([agent_0, agent_1]))
            else:
                # If no agent columns found, we'll need to infer from the data structure
                # For now, mark as unknown team
                team_key = f"unknown_team_{subject_id}"
            
            team_mapping[team_key].append(subject_id)
            subject_data[subject_id] = {
                'df': df,
                'file_path': file_path,
                'team_key': team_key
            }
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return team_mapping, subject_data


def check_reward_consistency(team_mapping, subject_data):
    """
    Check if the sum of infos.0.delivery_reward per episode matches between team pairs.
    Returns detailed episode-by-episode data for all teams.
    """
    all_team_episodes = []
    
    for team_key, subject_ids in team_mapping.items():
        if len(subject_ids) != 2:
            continue
        
        subject_1_id, subject_2_id = subject_ids
        df1 = subject_data[subject_1_id]['df']
        df2 = subject_data[subject_2_id]['df']
        
        # Check if infos.0.delivery_reward column exists
        reward_col = 'infos.0.delivery_reward'
        if reward_col not in df1.columns or reward_col not in df2.columns:
            continue
        
        # Calculate sum of delivery rewards per episode for each subject
        rewards_1 = df1.groupby('episode_num')[reward_col].sum().sort_index()
        rewards_2 = df2.groupby('episode_num')[reward_col].sum().sort_index()
        
        # Get common episodes
        common_episodes = sorted(set(rewards_1.index) & set(rewards_2.index))
        
        if not common_episodes:
            continue
        
        # Record all episodes for this team
        for episode in common_episodes:
            r1 = rewards_1.loc[episode] if episode in rewards_1.index else 0
            r2 = rewards_2.loc[episode] if episode in rewards_2.index else 0
            diff = abs(r1 - r2)
            
            all_team_episodes.append({
                'team_key': str(team_key),
                'subject_1': subject_1_id,
                'subject_2': subject_2_id,
                'episode': episode,
                'subject_1_reward': r1,
                'subject_2_reward': r2,
                'difference': diff
            })
    
    return all_team_episodes


def export_results(all_team_episodes):
    """
    Export all team episode-by-episode results to CSV file.
    """
    output_dir = Path(agg_data_dir).parent / "consistency_check_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export all teams with episode details
    if all_team_episodes:
        episodes_df = pd.DataFrame(all_team_episodes)
        output_path = output_dir / "all_teams_episode_rewards.csv"
        episodes_df.to_csv(output_path, index=False)
        print(f"✓ All teams episode rewards exported to: {output_path}")
        
        # Print summary statistics
        print(f"\nTotal episodes recorded: {len(episodes_df)}")
        print(f"Number of teams: {episodes_df['team_key'].nunique()}")
        print(f"Episodes with differences: {(episodes_df['difference'] > 1e-6).sum()}")
        print(f"Episodes with matching rewards: {(episodes_df['difference'] < 1e-6).sum()}")
        
        # Create and export team summary with total differences
        team_summary = []
        for team_key in episodes_df['team_key'].unique():
            team_data = episodes_df[episodes_df['team_key'] == team_key]
            
            team_summary.append({
                'team_key': team_key,
                'subject_1': team_data.iloc[0]['subject_1'],
                'subject_2': team_data.iloc[0]['subject_2'],
                'total_episodes': len(team_data),
                'total_difference': team_data['difference'].sum(),
                'episodes_with_mismatch': (team_data['difference'] > 1e-6).sum(),
                'episodes_matching': (team_data['difference'] < 1e-6).sum(),
                'max_difference_in_single_episode': team_data['difference'].max(),
                'mean_difference_per_episode': team_data['difference'].mean()
            })
        
        summary_df = pd.DataFrame(team_summary)
        summary_path = output_dir / "team_total_differences_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Team summary with total differences exported to: {summary_path}")
    else:
        print("No episode data to export.")


if __name__ == "__main__":
    print("=" * 100)
    print("TEAM REWARD CONSISTENCY CHECK - CATEGORY 4 (COMPLETED SUBJECTS)")
    print("=" * 100)
    print(f"\nData directory: {agg_data_dir}\n")
    
    # Identify team pairs
    team_mapping, subject_data = identify_team_pairs()
    
    print(f"Total subjects found: {len(subject_data)}")
    print(f"Total teams identified: {len(team_mapping)}\n")
    
    # Check reward consistency and collect all episode data
    all_team_episodes = check_reward_consistency(team_mapping, subject_data)
    
    # Export results
    export_results(all_team_episodes)
    
    print("\n" + "=" * 100)
    print("CONSISTENCY CHECK COMPLETE")
    print("=" * 100)
