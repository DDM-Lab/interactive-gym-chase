import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt

agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\AI_speed\aggregated_data\AI_speed_5"
data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-AI Speed\Data\AI-Speed-5FPS\ai_speed_5_human_ai_condition_data"
fig_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\AI_speed\aggregated_data\AI_speed_5\generated_figs"
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

print("=" * 100)
print("AGENT 0 DELIVERY REWARDS BY SUBJECT AND EPISODE")
print("=" * 100)

# Iterate through each subject
for subject_id in subject_ids:
    file_path = Path(agg_data_dir) / f"{subject_id}_aggregated.csv"
    
    if not file_path.exists():
        print(f"\nWarning: File not found for {subject_id}")
        continue
    
    df = pd.read_csv(file_path)
    
    # Calculate sum of delivery_reward, delivery_act_reward, and onion_in_pot_reward for each episode
    delivery_reward = df.groupby("episode_num")["infos.0.delivery_reward"].sum()
    delivery_act_reward = df.groupby("episode_num")["infos.0.delivery_act_reward"].sum()
    onion_in_pot_reward = df.groupby("episode_num")["infos.0.onion_in_pot_reward"].sum()
    
    print(f"\n{subject_id}:")
    print(f"{'Episode':<10} {'delivery_reward':<20} {'delivery_act_reward':<20} {'onion_in_pot_reward':<20}")
    print("-" * 70)
    
    for episode_num in sorted(delivery_reward.index):
        reward = delivery_reward[episode_num]
        act_reward = delivery_act_reward[episode_num]
        onion_reward = onion_in_pot_reward[episode_num]
        print(f"{int(episode_num):<10} {reward:<20.1f} {act_reward:<20.1f} {onion_reward:<20.1f}")
    
    # Calculate total delivery_reward across episodes 0-19 and bonus
    total_delivery_reward = delivery_reward.sum()
    bonus = total_delivery_reward * 0.02 if total_delivery_reward >= 20 else 0
    print("-" * 70)
    print(f"{'TOTAL':<10} {total_delivery_reward:<20.1f}")
    print(f"{'BONUS (×0.02)':<10} ${bonus:<20.2f}")

print("\n" + "=" * 100)

# Plot mean delivery_act_reward and onion_in_pot_reward across all subjects per episode
print("\n" + "=" * 100)
print("PLOTTING MEAN REWARDS BY EPISODE")
print("=" * 100)

all_delivery_act_rewards = []
all_onion_in_pot_rewards = []
episode_nums = None

# Iterate through each subject and collect reward data
for subject_id in subject_ids:
    file_path = Path(agg_data_dir) / f"{subject_id}_aggregated.csv"
    
    if not file_path.exists():
        continue
    
    df = pd.read_csv(file_path)
    
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
    output_path = Path(fig_dir) / "mean_rewards_by_episode.png"
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

bonus_data = []

for subject_id in subject_ids:
    file_path = Path(agg_data_dir) / f"{subject_id}_aggregated.csv"
    
    if not file_path.exists():
        continue
    
    df = pd.read_csv(file_path)
    
    # Calculate total delivery_reward
    delivery_reward = df.groupby("episode_num")["infos.0.delivery_reward"].sum()
    total_delivery_reward = delivery_reward.sum()
    bonus = total_delivery_reward * 0.02 if total_delivery_reward >= 20 else 0
    
    bonus_data.append({"subject_id": subject_id, "bonus": bonus})

# Create DataFrame and save to CSV
bonus_df = pd.DataFrame(bonus_data)
csv_output_path = Path(fig_dir) / "subject_bonuses.csv"
csv_output_path.parent.mkdir(parents=True, exist_ok=True)
bonus_df.to_csv(csv_output_path, index=False)
print(f"\nBonus CSV saved to: {csv_output_path}")
print(f"\nBonus Summary:")
print(bonus_df.to_string(index=False))

# Generate histogram of bonus distribution
print("\n" + "=" * 100)
print("GENERATING BONUS DISTRIBUTION HISTOGRAM")
print("=" * 100)

plt.figure(figsize=(10, 6))
plt.hist(bonus_df['bonus'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
plt.xlabel('Bonus Amount ($)', fontsize=12)
plt.ylabel('Number of Subjects', fontsize=12)
plt.title('Distribution of Bonuses (Subjects with <20 pts reward = $0)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

# Add vertical line at 0 to highlight threshold
zero_count = (bonus_df['bonus'] == 0).sum()
if zero_count > 0:
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Zero bonus ({zero_count} subjects)')
    plt.legend(fontsize=11)

# Save the histogram
histogram_path = Path(fig_dir) / "bonus_distribution_histogram.png"
histogram_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
print(f"\nHistogram saved to: {histogram_path}")

plt.show()
