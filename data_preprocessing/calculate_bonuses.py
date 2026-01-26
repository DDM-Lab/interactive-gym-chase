import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Configuration
experiment_type = "human-only"  # Change to "AI-only" or "Human-AI" as needed
agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data"
fig_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\generated_figs"

# Get subject IDs from aggregated data folder
agg_dir = Path(agg_data_dir)
subject_ids = []

if agg_dir.exists():
    for file_path in agg_dir.glob("*_aggregated.csv"):
        # Extract subject ID by removing "_aggregated.csv"
        subject_id = file_path.stem.replace("_aggregated", "")
        subject_ids.append(subject_id)
    subject_ids.sort()
else:
    print(f"Warning: Aggregated data directory not found: {agg_dir}")



# Plot mean delivery_act_reward and onion_in_pot_reward across all subjects per episode
print("\n" + "=" * 100)
print("PLOTTING MEAN REWARDS BY EPISODE")
print("=" * 100)

output_path = Path(fig_dir) / "mean_rewards_by_episode.png"

if output_path.exists():
    print(f"\nPlot already exists: {output_path}")
    print("Skipping regeneration...")
else:
    all_delivery_act_rewards = []
    all_onion_in_pot_rewards = []
    episode_nums = None

    # Iterate through each subject and collect reward data (only completed subjects)
    for subject_id in subject_ids:
        file_path = Path(agg_data_dir) / f"{subject_id}_aggregated.csv"
        
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        
        # Skip unpaired subjects (they won't have episode data)
        if 'status' in df.columns and df.iloc[0]['status'] == 'unpaired':
            continue
        
        # Calculate sum of rewards for each episode
        delivery_act_reward = df.groupby("episode_num")["infos.0.delivery_act_reward"].sum()
        onion_in_pot_reward = df.groupby("episode_num")["infos.0.onion_in_pot_reward"].sum()
        
        all_delivery_act_rewards.append(delivery_act_reward)
        all_onion_in_pot_rewards.append(onion_in_pot_reward)
        
        if episode_nums is None:
            episode_nums = sorted(delivery_act_reward.index)

    # Calculate mean across all subjects
    if all_delivery_act_rewards and all_onion_in_pot_rewards:
        mean_delivery_act = pd.concat(all_delivery_act_rewards, axis=1).mean(axis=1)
        mean_onion_in_pot = pd.concat(all_onion_in_pot_rewards, axis=1).mean(axis=1)
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(episode_nums, mean_delivery_act, marker='o', linewidth=2, label='Mean Delivery Act Reward')
        plt.plot(episode_nums, mean_onion_in_pot, marker='s', linewidth=2, label='Mean Onion in Pot Reward')
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Mean Reward', fontsize=12)
        plt.title('Mean Rewards by Episode (All Subjects)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xticks(episode_nums)
        
        # Save the figure
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        
        plt.show()
    else:
        print("No data available to plot")

# Generate CSV with subject_id and bonus
print("\n" + "=" * 100)
print("GENERATING BONUS CSV")
print("=" * 100)

csv_output_path = Path(fig_dir) / "subject_bonuses.csv"

if csv_output_path.exists():
    print(f"\nBonus CSV already exists: {csv_output_path}")
    print("Skipping regeneration...")
    bonus_df = pd.read_csv(csv_output_path)
else:
    # Dynamically determine threshold based on max episodes
    max_episode = 0
    for subject_id in subject_ids:
        file_path = Path(agg_data_dir) / f"{subject_id}_aggregated.csv"
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'episode_num' in df.columns:
                subject_max_episode = df['episode_num'].max()
                if subject_max_episode > max_episode:
                    max_episode = subject_max_episode
    
    bonus_threshold = max_episode
    print(f"Dynamically determined bonus threshold: {bonus_threshold} (based on max episodes)")
    
    bonus_data = []

    for subject_id in subject_ids:
        file_path = Path(agg_data_dir) / f"{subject_id}_aggregated.csv"
        
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        
        # Check if subject is unpaired
        if 'status' in df.columns and df.iloc[0]['status'] == 'unpaired':
            # Unpaired subjects get fixed $0.50
            bonus = 0.50
            status = 'unpaired'
        else:
            # Completed subjects: calculate bonus based on delivery_reward
            delivery_reward = df.groupby("episode_num")["infos.0.delivery_reward"].sum()
            total_delivery_reward = delivery_reward.sum()
            bonus = total_delivery_reward * 0.02 if total_delivery_reward >= bonus_threshold else 0
            status = 'completed'
        
        bonus_data.append({"subject_id": subject_id, "status": status, "bonus": bonus})

    # Create DataFrame and save to CSV
    bonus_df = pd.DataFrame(bonus_data)
    # Sort by status (unpaired first, then completed) and by subject_id within each group
    bonus_df = bonus_df.sort_values(by=['status', 'subject_id'], key=lambda x: x.map({'unpaired': 0, 'completed': 1}) if x.name == 'status' else x)
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    bonus_df.to_csv(csv_output_path, index=False)
    print(f"\nBonus CSV saved to: {csv_output_path}")

# Generate list of subjects who received bonuses
print("\n" + "=" * 100)
print("SUBJECTS WHO RECEIVED BONUSES")
print("=" * 100)

bonus_recipients = bonus_df[bonus_df['bonus'] > 0].copy()
print(f"\nNumber of subjects with bonuses: {len(bonus_recipients)}")
print(f"Total subjects: {len(bonus_df)}")

# Save bonus recipients to separate CSV
bonus_recipients_path = Path(fig_dir) / "bonus_recipients.csv"
bonus_recipients_path.parent.mkdir(parents=True, exist_ok=True)
bonus_recipients.to_csv(bonus_recipients_path, index=False)
print(f"\nBonus recipients CSV saved to: {bonus_recipients_path}")

# Save bonus recipients as a string list to text file
bonus_recipients_list = bonus_recipients['subject_id'].tolist()
bonus_recipients_txt_path = Path(fig_dir) / "bonus_recipients_list.txt"
with open(bonus_recipients_txt_path, 'w') as f:
    f.write(str(bonus_recipients_list))
print(f"Bonus recipients list (as string) saved to: {bonus_recipients_txt_path}")
print(f"\nBonus Recipients: {bonus_recipients_list}")

# Generate histogram of bonus distribution
print("\n" + "=" * 100)
print("GENERATING BONUS DISTRIBUTION HISTOGRAM")
print("=" * 100)

histogram_path = Path(fig_dir) / "bonus_distribution_histogram.png"

if histogram_path.exists():
    print(f"\nHistogram already exists: {histogram_path}")
    print("Skipping regeneration...")
else:
    plt.figure(figsize=(10, 6))
    plt.hist(bonus_df['bonus'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Bonus Amount ($)', fontsize=12)
    plt.ylabel('Number of Subjects', fontsize=12)
    plt.title('Distribution of Bonuses (Unpaired=$0.50, Completed subjects based on performance)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add vertical line at 0.50 to highlight unpaired subjects
    unpaired_count = (bonus_df['bonus'] == 0.50).sum()
    if unpaired_count > 0:
        plt.axvline(x=0.50, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Unpaired ($0.50, {unpaired_count} subjects)')
        plt.legend(fontsize=11)

    # Save the histogram
    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {histogram_path}")

    plt.show()
