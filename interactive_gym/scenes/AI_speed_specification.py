import json
import random
import os


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
    
    # Return the second value (index 1) from the tuple
    return agent_speeds[random_key][1]
