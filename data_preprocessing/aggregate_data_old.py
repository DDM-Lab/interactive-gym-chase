import os
import pandas as pd


def calculate_noop_to_action_mean_time(action_col, time_col):
    """
    Calculate the mean time from the start of a Noop sequence to when a different action is taken.

    Parameters:
    -----------
    action_col : pd.Series
        Column containing action strings (e.g., "Noop", "MoveLeft", etc.)
    time_col : pd.Series
        Column containing elapsed time in seconds

    Returns:
    --------
    float
        Mean time (in seconds) from the first Noop in a sequence to the next different action.
        Returns NaN if no transitions found.

    Example:
    --------
    Action:    Noop  → Noop  → Noop → Noop → MoveLeft → Noop
    Time (s):  1.0   → 1.05  → 1.10 → 2.0  → 2.04    → 2.5

    Would calculate: (2.04 - 1.0) = 1.04 seconds for the first sequence
    """
    # Reset indices to ensure proper iteration order
    action_col = action_col.reset_index(drop=True)
    time_col = time_col.reset_index(drop=True)

    transition_times = []
    i = 0

    while i < len(action_col):
        # Check if current action is Noop
        if action_col.iloc[i] == "Noop":
            noop_start_time = time_col.iloc[i]
            noop_start_idx = i

            # Find the end of this Noop sequence
            while i < len(action_col) and action_col.iloc[i] == "Noop":
                i += 1

            # If we found a different action after the Noop sequence
            if i < len(action_col):
                action_time = time_col.iloc[i]
                time_diff = action_time - noop_start_time
                # Only record positive time differences (ignore backwards time)
                if time_diff > 0:
                    transition_times.append(time_diff)
        else:
            i += 1

    if transition_times:
        return sum(transition_times) / len(transition_times)
    else:
        return float('nan')


def data_aggregation(PROCESSED_DATA_DIR,filtered_df_path):
    # Ensure the processed directory exists
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    episode_agg_df_path = os.path.join(PROCESSED_DATA_DIR, "episode_agg.csv")
    
    # Check if both output files already exist
    if os.path.exists(episode_agg_df_path):
        print("No need for episode aggregation -- aggregated files already exist.")
        return episode_agg_df_path

    # Reload the data created in the last step
    filtered_df = pd.read_csv(filtered_df_path)
    [rr for rr in filtered_df["agent-0_reward"].to_list() if rr not in [0.0, 1.0]]

    # Define NUMBER_TO_ACTION mapping
    NUMBER_TO_ACTION = {
        0: "MoveLeft",
        1: "MoveRight",
        2: "MoveUp",
        3: "MoveDown",
        4: "Toggle",
        5: "PickupDrop",
        6: "Noop"
    }

    # Convert action columns from numbers to strings
    filtered_df["agent-0_action"] = filtered_df["agent-0_action"].map(NUMBER_TO_ACTION)
    filtered_df["agent-1_action"] = filtered_df["agent-1_action"].map(NUMBER_TO_ACTION)

    # Create variables
    for agent_id in ["agent-0", "agent-1"]:
        filtered_df = filtered_df[~filtered_df[f"{agent_id}_pos"].isna()]


        filtered_df[[f"{agent_id}_pos_row", f"{agent_id}_pos_col"]] = (filtered_df[f"{agent_id}_pos"]
            .str.strip('[]')  # Remove the square brackets
            .str.split(expand=True)# Split the string into separate columns
        )

        filtered_df[f"{agent_id}_pos_row"] = filtered_df[f"{agent_id}_pos_row"].apply(lambda x: int(x) if isinstance(x, str) else None).astype(int)
        filtered_df[f"{agent_id}_pos_col"] = filtered_df[f"{agent_id}_pos_col"].apply(lambda x: int(x) if isinstance(x, str) else None).astype(int)

        filtered_df[f"{agent_id}_onion_in_pot"] = ((filtered_df[f"{agent_id}_pos_row"] == 2) & (filtered_df[f"{agent_id}_pos_col"] == 3) & (filtered_df[f"{agent_id}_action"] == "Toggle") & (filtered_df[f"{agent_id}_reward"].isin([0.1, 1.1]))).astype(int)
        filtered_df[f"{agent_id}_plated_dish"] = ((filtered_df[f"{agent_id}_pos_row"] == 2) & (filtered_df[f"{agent_id}_pos_col"] == 3) & (filtered_df[f"{agent_id}_action"] == "Toggle") & (filtered_df[f"{agent_id}_reward"].isin([0.3, 1.3]))).astype(int)

        # Make sure we have no missing values for the document being in focus
        filtered_df[f"{agent_id}_doc_in_focus"] = filtered_df[f"{agent_id}_doc_in_focus"].map(lambda x: x if not pd.isna(x) else 0).astype(int)

        # Track if the action is no-op
        filtered_df[f"{agent_id}_action_is_noop"] = filtered_df[f"{agent_id}_action"] == "Noop"

        # Calculate delivery events before modifying reward
        filtered_df[f"{agent_id}_delivered_dish"] = ((filtered_df[f"{agent_id}_pos_row"] == 3) & (filtered_df[f"{agent_id}_pos_col"] == 4) & (filtered_df[f"{agent_id}_action"] == "Toggle") & (filtered_df[f"{agent_id}_reward"] == 1.0)).astype(int)
        
        # Create individual agent reward: only count reward if THIS agent performed the rewarding action
        # For onion placement: agent at [2,3] with action 4 and reward 0.1 or 1.1
        # For dish plating: agent at [2,3] with action 4 and reward 0.3 or 1.3  
        # For delivery: agent at [3,4] with action 4 and reward 1.0 or 1.1
        filtered_df[f"{agent_id}_individual_reward"] = 0.0
        
        # Onion placement rewards (0.1 or 1.1)
        filtered_df.loc[
            (filtered_df[f"{agent_id}_pos_row"] == 2) & 
            (filtered_df[f"{agent_id}_pos_col"] == 3) & 
            (filtered_df[f"{agent_id}_action"] == "Toggle") & 
            (filtered_df[f"{agent_id}_reward"].isin([0.1, 1.1])),
            f"{agent_id}_individual_reward"
        ] = filtered_df[f"{agent_id}_reward"]
        
        # Dish plating rewards (0.3 or 1.3)
        filtered_df.loc[
            (filtered_df[f"{agent_id}_pos_row"] == 2) & 
            (filtered_df[f"{agent_id}_pos_col"] == 3) & 
            (filtered_df[f"{agent_id}_action"] == "Toggle") & 
            (filtered_df[f"{agent_id}_reward"].isin([0.3, 1.3])),
            f"{agent_id}_individual_reward"
        ] = filtered_df[f"{agent_id}_reward"]
        
        # Delivery rewards (1.0 or 1.1)
        filtered_df.loc[
            (filtered_df[f"{agent_id}_pos_row"] == 3) & 
            (filtered_df[f"{agent_id}_pos_col"] == 4) & 
            (filtered_df[f"{agent_id}_action"] == "Toggle") & 
            (filtered_df[f"{agent_id}_reward"].isin([1.0, 1.1])),
            f"{agent_id}_individual_reward"
        ] = filtered_df[f"{agent_id}_reward"]

    # Create additional columns for reward sums by action type
    # Initialize reward sum columns
    filtered_df["agent-0_delivery_reward_sum"] = 0.0
    filtered_df["agent-0_onion_reward_sum"] = 0.0
    filtered_df["agent-0_plated_reward_sum"] = 0.0
    
    filtered_df["agent-1_delivery_reward_sum"] = 0.0
    filtered_df["agent-1_onion_reward_sum"] = 0.0
    filtered_df["agent-1_plated_reward_sum"] = 0.0
    
    # Calculate reward sums for each agent and action type
    for agent_id in ["agent-0", "agent-1"]:
        # Delivery reward sum
        filtered_df.loc[
            (filtered_df[f"{agent_id}_pos_row"] == 3) & 
            (filtered_df[f"{agent_id}_pos_col"] == 4) & 
            (filtered_df[f"{agent_id}_action"] == "Toggle") & 
            (filtered_df[f"{agent_id}_reward"].isin([1.0, 1.1])),
            f"{agent_id}_delivery_reward_sum"
        ] = filtered_df[f"{agent_id}_reward"]
        
        # Onion reward sum
        filtered_df.loc[
            (filtered_df[f"{agent_id}_pos_row"] == 2) & 
            (filtered_df[f"{agent_id}_pos_col"] == 3) & 
            (filtered_df[f"{agent_id}_action"] == "Toggle") & 
            (filtered_df[f"{agent_id}_reward"].isin([0.1, 1.1])),
            f"{agent_id}_onion_reward_sum"
        ] = filtered_df[f"{agent_id}_reward"]
        
        # Plated reward sum
        filtered_df.loc[
            (filtered_df[f"{agent_id}_pos_row"] == 2) & 
            (filtered_df[f"{agent_id}_pos_col"] == 3) & 
            (filtered_df[f"{agent_id}_action"] == "Toggle") & 
            (filtered_df[f"{agent_id}_reward"].isin([0.3, 1.3])),
            f"{agent_id}_plated_reward_sum"
        ] = filtered_df[f"{agent_id}_reward"]

    # Calculate mean speed for each episode (time from Noop to next action)
    speed_data_0 = []
    speed_data_1 = []

    for (game_id, episode_num), group in filtered_df.groupby(["game_id", "episode_num"]):
        # Sort by time to ensure chronological order
        group = group.sort_values("episode_s_elapsed")

        agent_0_speed = calculate_noop_to_action_mean_time(group["agent-0_action"], group["episode_s_elapsed"])
        agent_1_speed = calculate_noop_to_action_mean_time(group["agent-1_action"], group["episode_s_elapsed"])

        speed_data_0.append({"game_id": game_id, "episode_num": episode_num, "agent_0_mean_speed": agent_0_speed})
        speed_data_1.append({"game_id": game_id, "episode_num": episode_num, "agent_1_mean_speed": agent_1_speed})

    speed_df_0 = pd.DataFrame(speed_data_0)
    speed_df_1 = pd.DataFrame(speed_data_1)

    episode_agg = filtered_df.groupby(["game_id", "episode_num"]).agg(
        agent_0_identifier=("agent-0_identifier", "first"),
        agent_0_is_human=("agent-0_is_human", "first"),
        agent_0_doc_in_focus_mean=("agent-0_doc_in_focus", "mean"),
        agent_0_action_is_noop_mean=("agent-0_action_is_noop", "mean"),
        agent_0_reward_sum=("agent-0_individual_reward", "sum"),
        agent_0_deliveries_frequency=("agent-0_delivered_dish", "sum"),
        agent_0_deliveries_sum=("agent-0_delivery_reward_sum", "sum"),
        agent_0_onion_in_pot_frequency=("agent-0_onion_in_pot", "sum"),
        agent_0_onion_in_pot_sum=("agent-0_onion_reward_sum", "sum"),
        agent_0_plated_dish_frequency=("agent-0_plated_dish", "sum"),
        agent_0_plated_dish_sum=("agent-0_plated_reward_sum", "sum"),

        agent_1_identifier=("agent-1_identifier", "first"),
        agent_1_is_human=("agent-1_is_human", "first"),
        agent_1_doc_in_focus_mean=("agent-1_doc_in_focus", "mean"),
        agent_1_action_is_noop_mean=("agent-1_action_is_noop", "mean"),
        agent_1_reward_sum=("agent-1_individual_reward", "sum"),
        agent_1_deliveries_frequency=("agent-1_delivered_dish", "sum"),
        agent_1_deliveries_sum=("agent-1_delivery_reward_sum", "sum"),
        agent_1_onion_in_pot_frequency=("agent-1_onion_in_pot", "sum"),
        agent_1_onion_in_pot_sum=("agent-1_onion_reward_sum", "sum"),
        agent_1_plated_dish_frequency=("agent-1_plated_dish", "sum"),
        agent_1_plated_dish_sum=("agent-1_plated_reward_sum", "sum"),

        episode_duration_s=("episode_s_elapsed", "max"),
        episode_num_ticks=("tick_num", "max"),
    ).reset_index()
    
    # Calculate total reward frequency for each agent (sum of all action frequencies)
    episode_agg["agent_0_reward_frequency"] = (
        episode_agg["agent_0_deliveries_frequency"] + 
        episode_agg["agent_0_onion_in_pot_frequency"] + 
        episode_agg["agent_0_plated_dish_frequency"]
    )
    
    episode_agg["agent_1_reward_frequency"] = (
        episode_agg["agent_1_deliveries_frequency"] + 
        episode_agg["agent_1_onion_in_pot_frequency"] + 
        episode_agg["agent_1_plated_dish_frequency"]
    )
    
    # Add cumulative reward column (sum of both agents' individual rewards)
    episode_agg["cumulative_reward"] = episode_agg["agent_0_reward_sum"] + episode_agg["agent_1_reward_sum"]
    
    # Add cumulative frequency column (sum of both agents' reward frequencies)
    episode_agg["cumulative_frequency"] = episode_agg["agent_0_reward_frequency"] + episode_agg["agent_1_reward_frequency"]
    
    # Add team-level delivery reward sum
    episode_agg["team_delivery_reward_sum"] = episode_agg["agent_0_deliveries_sum"] + episode_agg["agent_1_deliveries_sum"]
    
    # Add team-level cumulative rewards by action type
    episode_agg["cumulative_team_delivery_reward"] = episode_agg["team_delivery_reward_sum"]
    episode_agg["cumulative_team_onion_reward"] = episode_agg["agent_0_onion_in_pot_sum"] + episode_agg["agent_1_onion_in_pot_sum"]
    episode_agg["cumulative_team_plated_reward"] = episode_agg["agent_0_plated_dish_sum"] + episode_agg["agent_1_plated_dish_sum"]

    # Merge in the mean speed data
    episode_agg = episode_agg.merge(speed_df_0, on=["game_id", "episode_num"], how="left")
    episode_agg = episode_agg.merge(speed_df_1, on=["game_id", "episode_num"], how="left")

    episode_agg.to_csv(os.path.join(PROCESSED_DATA_DIR, "episode_agg.csv"), index=False)
    return episode_agg_df_path