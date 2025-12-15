import pandas as pd
import json
from pathlib import Path

agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\aggregated_data\pilot_2_aggregated"
data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed\Data\Pilot2\data"

# Get subject IDs from end_completion_code_scene folder
scene_dir = Path(data_dir) / "end_completion_code_scene"
subject_ids = []

if scene_dir.exists():
    for file_path in scene_dir.glob("*_metadata.json"):
        # Extract subject ID by removing "_metadata"
        subject_id = file_path.stem.replace("_metadata", "")
        subject_ids.append(subject_id)
    subject_ids.sort()
else:
    print(f"Warning: Scene directory not found: {scene_dir}")

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
    bonus = total_delivery_reward * 0.02
    print("-" * 70)
    print(f"{'TOTAL':<10} {total_delivery_reward:<20.1f}")
    print(f"{'BONUS (×0.02)':<10} ${bonus:<20.2f}")

print("\n" + "=" * 100)
