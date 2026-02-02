from pathlib import Path

#data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\Pilot1-2\human-only-data-pilot-3"
data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-data_run_1"

# Parameter: "human-only", "AI-only", or "Human-AI"
experiment_type = "human-only"
suffix = "hh" if experiment_type.lower() == "human-only" else "sp"

 


def get_episode_data():
    """
    Extract all unique IDs from cramped_room_{suffix} and find max episode per subject.
    Returns: dict with all_ids, max_episode_per_subject, and overall_max_episode
    """
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    all_ids = set()
    max_episode_per_subject = {}
    
    if cramped_room_dir.exists():
        # First, collect all IDs from any file in the directory (process all files)
        for file_path in cramped_room_dir.glob("*"):
            if file_path.is_file():
                filename = file_path.stem
                id_part = filename.split("_")[0]
                all_ids.add(id_part)
        
        # Then, extract episode numbers from episode files
        for file_path in cramped_room_dir.glob("*_ep*.csv"):
            filename = file_path.stem  # Remove .csv
            # Extract ID and episode number from pattern like "ID_ep5"
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
    
    return {
        "all_ids": sorted(list(all_ids)),
        "max_episode_per_subject": max_episode_per_subject,
        "overall_max_episode": overall_max_episode
    }


def get_unpaired_subjects():
    """
    Check for subjects in start_scene but not in cramped_room.
    """
    start_scene_dir = Path(data_dir) / f"overcooked_{suffix}_start_scene"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    start_scene_ids = set()
    cramped_room_ids = set()
    
    if start_scene_dir.exists():
        for file_path in start_scene_dir.glob("*"):
            id_part = file_path.stem.split("_")[0]
            start_scene_ids.add(id_part)
    
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            id_part = file_path.stem.split("_")[0]
            cramped_room_ids.add(id_part)
    
    unpaired_ids = sorted(list(start_scene_ids - cramped_room_ids))
    
    return {
        "total_start_scene": len(start_scene_ids),
        "unpaired_ids": unpaired_ids,
        "count_unpaired": len(unpaired_ids)
    }


def get_sanity_checks(max_episode_per_subject, overall_max_episode):
    """
    Sanity checks for data integrity.
    """
    end_scene_dir = Path(data_dir) / "end_completion_code_scene"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    # Get IDs from end_completion_code_scene
    end_scene_ids = set()
    if end_scene_dir.exists():
        for file_path in end_scene_dir.glob("*"):
            id_part = file_path.stem.split("_")[0]
            end_scene_ids.add(id_part)
    
    # Get all IDs in cramped_room
    all_cramped_room_ids = set()
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            id_part = file_path.stem.split("_")[0]
            all_cramped_room_ids.add(id_part)
    
    # Check 1: Prematurely ended (has completion code, but didn't finish all episodes)
    prematurely_ended = []
    for id_str in end_scene_ids:
        if id_str in max_episode_per_subject:
            if max_episode_per_subject[id_str] < overall_max_episode:
                prematurely_ended.append(id_str)
    
    # Check 2: No episodes but got completion code
    no_episodes_but_completion = []
    for id_str in end_scene_ids:
        if id_str not in all_cramped_room_ids:
            no_episodes_but_completion.append(id_str)
    
    return {
        "prematurely_ended": sorted(prematurely_ended),
        "no_episodes_but_completion": sorted(no_episodes_but_completion)
    }


if __name__ == "__main__":
    # Get all data
    episode_data = get_episode_data()
    
    max_episode_per_subject = episode_data["max_episode_per_subject"]
    overall_max_episode = episode_data["overall_max_episode"]
    
    # Get IDs from each scene
    start_scene_dir = Path(data_dir) / "overcooked_hh_start_scene"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    end_scene_dir = Path(data_dir) / "end_completion_code_scene"
    
    start_scene_ids = set()
    if start_scene_dir.exists():
        for file_path in start_scene_dir.glob("*"):
            id_part = file_path.stem.split("_")[0]
            start_scene_ids.add(id_part)
    
    cramped_room_ids = set()
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            id_part = file_path.stem.split("_")[0]
            cramped_room_ids.add(id_part)
    
    end_scene_ids = set()
    if end_scene_dir.exists():
        for file_path in end_scene_dir.glob("*"):
            id_part = file_path.stem.split("_")[0]
            end_scene_ids.add(id_part)
    
    # Categorize subjects based on the 4 categories
    # (1) Subject failed to reach main session: In start_scene, NOT in cramped_room, NOT in end_completion
    failed_to_reach = sorted([id_str for id_str in start_scene_ids 
                             if id_str not in cramped_room_ids and id_str not in end_scene_ids])
    
    # (2) Subject failed to play main session: In start_scene, in cramped_room, but NO episode data
    failed_to_play = sorted([id_str for id_str in start_scene_ids 
                            if id_str in cramped_room_ids and id_str not in end_scene_ids
                            and id_str not in max_episode_per_subject])
    
    # (3) Subject's team ended during main session: In start_scene, in cramped_room (any amount of data), NOT in end_completion
    team_ended = sorted([id_str for id_str in start_scene_ids 
                        if id_str in cramped_room_ids and id_str not in end_scene_ids
                        and id_str in max_episode_per_subject])
    
    # (5) Subject completed: Has max number of episodes
    completed_subjects = sorted([id_str for id_str in start_scene_ids 
                                if id_str in max_episode_per_subject 
                                and max_episode_per_subject[id_str] == overall_max_episode])
    
    # Print report
    print("\n" + "=" * 70)
    print("DATA COMPLETENESS REPORT")
    print("=" * 70)
    print(f"Total unique IDs in overcooked_hh_start_scene: {len(start_scene_ids)}")
    print(f"Max episode number: {overall_max_episode}")
    
    # Category 1: Subject failed to reach main session
    print("\n(1) SUBJECT FAILED TO REACH MAIN SESSION")
    print("-" * 70)
    print(f"Count: {len(failed_to_reach)}")
    print("Criteria: In start_scene, NOT in cramped_room_hh, NOT in end_completion_code_scene")
    if failed_to_reach:
        print("\nSubject IDs:")
        for id_str in failed_to_reach:
            print(f"  - {id_str}")
    else:
        print("\n[OK] None found")
    
    # Category 2: Subject failed to play main session
    print("\n(2) SUBJECT FAILED TO PLAY MAIN SESSION")
    print("-" * 70)
    print(f"Count: {len(failed_to_play)}")
    print("Criteria: In start_scene, in cramped_room_hh, but NO episode data recorded")
    if failed_to_play:
        print("\nSubject IDs:")
        for id_str in failed_to_play:
            print(f"  - {id_str}")
    else:
        print("\n[OK] None found")
    
    # Category 3: Subject's team ended during main session
    print("\n(3) SUBJECT'S TEAM ENDED DURING MAIN SESSION")
    print("-" * 70)
    print(f"Count: {len(team_ended)}")
    print("Criteria: In start_scene, HAS data in cramped_room_hh, NOT in end_completion_code_scene")
    if team_ended:
        print("\nSubject IDs:")
        for id_str in team_ended:
            max_ep = max_episode_per_subject.get(id_str, 0)
            print(f"  - {id_str} (max episode: {max_ep})")
    else:
        print("\n[OK] None found")
    
    # Category 4: Completed subjects
    print("\n(4) SUBJECT COMPLETED THE EXPERIMENT")
    print("-" * 70)
    print(f"Count: {len(completed_subjects)}")
    print(f"Criteria: Has max number of episodes ({overall_max_episode}) in cramped_room_hh")
    if completed_subjects:
        print("\nSubject IDs:")
        for id_str in completed_subjects:
            print(f"  - {id_str}")
    else:
        print("\n[OK] None found")
    
    # Category 5: Uncategorized - IDs that don't fall into any of the above categories
    all_categorized = set(failed_to_reach + failed_to_play + team_ended + completed_subjects)
    # Build universe of IDs from all discovered sources
    all_ids_universe = start_scene_ids.union(cramped_room_ids).union(end_scene_ids).union(set(max_episode_per_subject.keys()))
    uncategorized = sorted([id_str for id_str in all_ids_universe if id_str not in all_categorized])
    
    print("\n(5) UNCATEGORIZED (NOT IN ANY ABOVE CATEGORY)")
    print("-" * 70)
    print(f"Count: {len(uncategorized)}")
    print("Criteria: Valid MTurk ID but doesn't fit any of the above categories")
    if uncategorized:
        print("\nSubject IDs:")
        for id_str in uncategorized:
            in_start = "✓" if id_str in start_scene_ids else "✗"
            in_cramped = "✓" if id_str in cramped_room_ids else "✗"
            in_end = "✓" if id_str in end_scene_ids else "✗"
            has_episodes = "✓" if id_str in max_episode_per_subject else "✗"
            print(f"  - {id_str} [start:{in_start} cramped:{in_cramped} end:{in_end} episodes:{has_episodes}]")
    else:
        print("\n[OK] None found")
    
    print("\n" + "=" * 70)
