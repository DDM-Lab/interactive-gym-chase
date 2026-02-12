import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
experiment_type = "human-only"  # Change to "AI-only" or "Human-AI" as needed
agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\run_3_and_4"
fig_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\run_3_and_4\generated_figs"
aligned_team_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\run_3_and_4\aligned_team_data"

# Category folder names
CATEGORY_FOLDERS = {
    1: "category_1_failed_to_reach_main_session",
    2: "category_2_failed_to_play_main_session",
    3: "category_3_team_ended_during_main_session",
    4: "category_4_completed_experiment",
    5: "category_5_uncategorized"
}

# Get subject IDs from all category folders
agg_dir = Path(agg_data_dir)
subject_data = []  # List of (subject_id, category, file_path)

if agg_dir.exists():
    for category_num, folder_name in CATEGORY_FOLDERS.items():
        category_dir = agg_dir / folder_name
        if category_dir.exists():
            for file_path in category_dir.glob("*_aggregated.csv"):
                # Extract subject ID by removing "_aggregated.csv"
                subject_id = file_path.stem.replace("_aggregated", "")
                subject_data.append((subject_id, category_num, file_path))
    subject_data.sort(key=lambda x: (x[1], x[0]))  # Sort by category, then subject_id

# Plot mean delivery_act_reward and onion_in_pot_reward across all subjects per episode
# COMMENTED OUT FOR NOW - visualization not needed

# output_path = Path(fig_dir) / "mean_rewards_by_episode.png"
# 
# if not output_path.exists():
#     all_delivery_act_rewards = []
#     all_onion_in_pot_rewards = []
#     all_episode_nums = set()
# 
#     # Iterate through each subject and collect reward data (only category 4 - completed subjects)
#     for subject_id, category_num, file_path in subject_data:
#         # Only process category 4 (completed subjects with full episode data)
#         if category_num != 4:
#             continue
#         
#         df = pd.read_csv(file_path)
#         
#         # Calculate sum of rewards for each episode
#         delivery_act_reward = df.groupby("episode_num")["infos.0.delivery_act_reward"].sum()
#         onion_in_pot_reward = df.groupby("episode_num")["infos.0.onion_in_pot_reward"].sum()
#         
#         all_delivery_act_rewards.append(delivery_act_reward)
#         all_onion_in_pot_rewards.append(onion_in_pot_reward)
#         all_episode_nums.update(delivery_act_reward.index)
# 
#     # Calculate mean across all subjects
#     if all_delivery_act_rewards and all_onion_in_pot_rewards:
#         mean_delivery_act = pd.concat(all_delivery_act_rewards, axis=1).mean(axis=1)
#         mean_onion_in_pot = pd.concat(all_onion_in_pot_rewards, axis=1).mean(axis=1)
#         episode_nums = sorted(mean_delivery_act.index)
#         
#         # Create the plot
#         plt.figure(figsize=(12, 6))
#         plt.plot(episode_nums, mean_delivery_act, marker='o', linewidth=2, label='Mean Delivery Act Reward')
#         plt.plot(episode_nums, mean_onion_in_pot, marker='s', linewidth=2, label='Mean Onion in Pot Reward')
#         
#         plt.xlabel('Episode', fontsize=12)
#         plt.ylabel('Mean Reward', fontsize=12)
#         plt.title('Mean Rewards by Episode (Category 4: Completed Subjects)', fontsize=14, fontweight='bold')
#         plt.legend(fontsize=11)
#         plt.grid(True, alpha=0.3)
#         plt.xticks(episode_nums)
#         
#         # Save the figure
#         output_path.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(output_path, dpi=300, bbox_inches='tight')
#         plt.show()

# Generate CSV with subject_id and bonus

csv_output_path = Path(agg_data_dir) / "subject_compensations.csv"

# Dynamically determine threshold based on max episodes
max_episode = 0
for subject_id, category_num, file_path in subject_data:
    if category_num in [3, 4]:
        df = pd.read_csv(file_path)
        if 'episode_num' in df.columns:
            subject_max_episode = df['episode_num'].max()
            if subject_max_episode > max_episode:
                max_episode = subject_max_episode

bonus_threshold = max_episode

# Category mapping
category_mapping = {
    1: "unpaired",
    2: "lag",
    3: "quit",
    4: "completed",
    5: "unknown"
}

bonus_data = []
team_delivery_breakdown = []  # Track deliveries per episode per team
    
# Process categories 3 and 4 from aligned_team_data directory
aligned_team_dir = Path(aligned_team_data_dir)
processed_teams = set()  # Track teams already processed

if aligned_team_dir.exists():
    for team_file in aligned_team_dir.glob("team_*_aligned.csv"):
        df = pd.read_csv(team_file)
        
        # Extract team pair from first row
        first_row = df.iloc[0]
        player_0 = first_row['player_subjects.0']
        player_1 = first_row['player_subjects.1']
        category_num = int(first_row['category'])
        
        # Only process categories 3 and 4
        if category_num not in [3, 4]:
            continue
        
        # Create team_id from the filename
        team_id = team_file.stem.replace("_aligned", "")
        
        # Skip if already processed
        if team_id in processed_teams:
            continue
        processed_teams.add(team_id)
        
        # Get quitter_id if it exists (for category 3 teams)
        quitter_id = None
        if category_num == 3 and 'quitter_id' in df.columns:
            quitter_id = first_row['quitter_id']
            
        # Calculate bonus for the team (same logic as before)
        if category_num == 3:
            # Category 3 (quit): Calculate bonus based on their own max episode
            subject_episodes = df['episode_num'].nunique() if 'episode_num' in df.columns else 0
            
            # Check if infos.0.delivery_reward exists, otherwise use fallback
            if 'infos.0.delivery_reward' in df.columns:
                deliveries_per_episode = df.groupby('episode_num')['infos.0.delivery_reward'].sum()
            else:
                # Fallback: sum(rewards.0) per episode (equivalent to infos.0.delivery_reward)
                deliveries_per_episode = df.groupby('episode_num')['rewards.0'].sum()
            num_deliveries = (deliveries_per_episode > 0).sum()
            total_delivery_reward = deliveries_per_episode.sum()
            bonus = total_delivery_reward * 0.02 if num_deliveries >= subject_episodes else 0
        elif category_num == 4:
            # Category 4 (completed): Calculate bonus based on full episode count (19)
            
            # Check if infos.0.delivery_reward exists, otherwise use fallback
            if 'infos.0.delivery_reward' in df.columns:
                deliveries_per_episode = df.groupby('episode_num')['infos.0.delivery_reward'].sum()
            else:
                # Fallback: sum(rewards.0) per episode (equivalent to infos.0.delivery_reward)
                deliveries_per_episode = df.groupby('episode_num')['rewards.0'].sum()
            num_deliveries = (deliveries_per_episode > 0).sum()
            total_delivery_reward = deliveries_per_episode.sum()
            bonus = total_delivery_reward * 0.02 if num_deliveries >= bonus_threshold else 0
        
        # Store delivery breakdown for this team
        for episode, delivery_total in deliveries_per_episode.items():
            team_delivery_breakdown.append({
                "team_id": team_id,
                "category": category_mapping[category_num],
                "episode_num": episode,
                "total_deliveries": delivery_total
            })
        
        # Add two rows: one for each partner
        # For category 3, check if quitter was identified and apply penalty only if identified
        if category_num == 3 and quitter_id not in [None, "None", "unknown"]:
            # Quitter identified: penalize the quitter
            player_0_bonus = 0 if quitter_id == player_0 else bonus
            player_1_bonus = 0 if quitter_id == player_1 else bonus
            quitter_display_id = quitter_id
        elif category_num == 3:
            # Quitter not identified: both get full bonus
            player_0_bonus = bonus
            player_1_bonus = bonus
            quitter_display_id = "unidentified"
        else:
            # Categories 1, 2, 4, 5: no quitter logic
            player_0_bonus = bonus
            player_1_bonus = bonus
            quitter_display_id = None
        
        bonus_data.append({
            "subject_id": player_0,
            "category": category_mapping[category_num],
            "compensation": player_0_bonus,
            "team_id": team_id,
            "partner_num": 0,
            "quitter_ID": quitter_display_id if category_num == 3 else None
        })
        bonus_data.append({
            "subject_id": player_1,
            "category": category_mapping[category_num],
            "compensation": player_1_bonus,
            "team_id": team_id,
            "partner_num": 1,
            "quitter_ID": quitter_display_id if category_num == 3 else None
        })

# Process categories 1, 2, and 5 from individual subject files
for subject_id, category_num, file_path in subject_data:
    # Skip categories 3 and 4 as they're handled above
    if category_num in [3, 4]:
        continue
        
    df = pd.read_csv(file_path)
    
    if category_num == 1:
        # Category 1 (unpaired): 50 cents
        bonus = 0.50
    elif category_num == 2:
        # Category 2 (lag): No compensation
        bonus = 0.00
    elif category_num == 5:
        # Category 5: No compensation
        bonus = 0.00
    else:
        bonus = 0.00
    
    bonus_data.append({
        "subject_id": subject_id,
        "category": category_mapping[category_num],
        "compensation": bonus,
        "team_id": None,
        "partner_num": None,
        "quitter_ID": None
    })

# Create DataFrame
bonus_df = pd.DataFrame(bonus_data)
bonus_df = bonus_df.sort_values(by=['category', 'team_id', 'subject_id'], na_position='last')

# Sanity check: report total deliveries per team
print("\n" + "=" * 80)
print("SANITY CHECK: TOTAL DELIVERIES PER TEAM")
print("=" * 80)

if team_delivery_breakdown:
    delivery_df = pd.DataFrame(team_delivery_breakdown)
    
    for team_id in sorted(delivery_df['team_id'].unique()):
        team_data = delivery_df[delivery_df['team_id'] == team_id]
        category = team_data['category'].iloc[0]
        total_deliveries = team_data['total_deliveries'].sum()
        
        print(f"\n{team_id} ({category})")
        for _, row in team_data.sort_values('episode_num').iterrows():
            print(f"  Episode {int(row['episode_num']):2d}: {int(row['total_deliveries']):3d} deliveries")
        print(f"  Total: {int(total_deliveries):3d} deliveries")

print("\n" + "=" * 80)

# Save to CSV
csv_output_path.parent.mkdir(parents=True, exist_ok=True)
bonus_df.to_csv(csv_output_path, index=False)

# Generate list of subjects who received compensation
compensation_recipients = bonus_df[bonus_df['compensation'] > 0].copy()

# Save compensation recipients to separate CSV
compensation_recipients_path = Path(fig_dir) / "compensation_recipients.csv"
compensation_recipients_path.parent.mkdir(parents=True, exist_ok=True)
compensation_recipients.to_csv(compensation_recipients_path, index=False)

# Save compensation recipients as a string list to text file
compensation_recipients_list = compensation_recipients['subject_id'].tolist()
compensation_recipients_txt_path = Path(fig_dir) / "compensation_recipients_list.txt"
with open(compensation_recipients_txt_path, 'w') as f:
    f.write(str(compensation_recipients_list))

# Generate histogram of compensation distribution
histogram_path = Path(fig_dir) / "compensation_distribution_histogram.png"

if not histogram_path.exists():
    plt.figure(figsize=(10, 6))
    plt.hist(bonus_df['compensation'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Compensation Amount ($)', fontsize=12)
    plt.ylabel('Number of Subjects', fontsize=12)
    plt.title('Distribution of Compensation by Category\n(Cat1=$0, Cat2=$0.50, Cat3&4=performance-based)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add vertical line at 0.50 to highlight category 2 subjects
    cat2_count = (bonus_df['compensation'] == 0.50).sum()
    if cat2_count > 0:
        plt.axvline(x=0.50, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Category 2 ($0.50, {cat2_count} subjects)')
        plt.legend(fontsize=11)

    # Save the histogram
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    plt.show()
