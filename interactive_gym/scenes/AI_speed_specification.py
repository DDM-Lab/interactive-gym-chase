import json
import random
import os
import math


def get_random_frame_skip():
    """
    Randomly selects a key from agent_0_mean_speeds.json and returns
    the second value (index 1) from the corresponding tuple.
    
    :return: The second value from a randomly selected agent's mean speeds tuple
    :rtype: float
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "subject_mean_speeds.json")
    
    # Load the JSON file
    with open(json_path, "r") as f:
        agent_speeds = json.load(f)
    
    # Randomly select a key
    random_key = random.choice(list(agent_speeds.keys()))
    
    # Get the second value (index 1) from the tuple
    frame_skip_value = agent_speeds[random_key][1]
    
    # Log the value for debugging
    # print(f"[get_random_frame_skip] Selected key: {random_key}, Raw value: {frame_skip_value}, int() value: {int(frame_skip_value)}")
    
    # Return the second value (index 1) from the tuple
    return frame_skip_value


def get_unique_frame_skip_values():
    """
    Extracts unique FPS (frame skip) values from the subject_mean_speeds.json file.
    For each subject in the JSON, extracts the FPS (second value in the tuple),
    rounds up to integers, and returns only the unique values.
    
    :return: A sorted list of unique frame skip values rounded up to integers
    :rtype: list
    """
    # Get the directory where this file is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, "subject_mean_speeds.json")
    
    # Load the JSON file
    with open(json_path, "r") as f:
        agent_speeds = json.load(f)
    
    # Extract FPS values (second value in tuple) for each subject
    fps_values = []
    for subject_key, speed_data in agent_speeds.items():
        fps = speed_data[1]  # Get the second value (FPS)
        fps_rounded = math.ceil(fps)  # Round up to integer
        fps_values.append(fps_rounded)
    
    # Get unique values and sort for consistent ordering
    unique_fps_values = sorted(list(set(fps_values)))
    
    return unique_fps_values


if __name__ == "__main__":
    unique_values = get_unique_frame_skip_values()
    print("Unique frame skip values:", unique_values)
