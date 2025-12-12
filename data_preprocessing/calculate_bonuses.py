import pandas as pd
import json
from pathlib import Path

agg_data_dir = r"C:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\aggregated_data"

# List of subject IDs
subject_ids = ["A11YS0T8MV3Q7C", "A1IID9HW53QMHF", "A3CZCEBX4KG54A", "AMOBC1D09Q381", "AVRV5IDK5O4S6"]

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
    bonus = total_delivery_reward * 0.20
    print("-" * 70)
    print(f"{'TOTAL':<10} {total_delivery_reward:<20.1f}")
    print(f"{'BONUS (×0.20)':<10} ${bonus:<20.2f}")

print("\n" + "=" * 100)
