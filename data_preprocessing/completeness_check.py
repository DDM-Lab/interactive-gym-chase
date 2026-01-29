from pathlib import Path

data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\Pilot1-2\human-only-data-pilot-3"

# Parameter: "human-only", "AI-only", or "Human-AI"
experiment_type = "human-only"
suffix = "hh" if experiment_type.lower() == "human-only" else "sp"

# Load valid MTurk IDs from file
valid_mturk_ids = set()
mturk_file = Path(__file__).parent.parent / "mturk_ids_unique_to_dir_2.txt"
if mturk_file.exists():
    with open(mturk_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and line.startswith('- '):
                valid_mturk_ids.add(line[2:])
            elif line and not line.startswith('-'):
                valid_mturk_ids.add(line)


def get_episode_data():
    """
    Extract all unique IDs from cramped_room_{suffix} and find max episode per subject.
    Returns: dict with all_ids, max_episode_per_subject, and overall_max_episode
    """
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    all_ids = set()
    max_episode_per_subject = {}
    
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*_ep*.csv"):
            filename = file_path.stem  # Remove .csv
            # Extract ID and episode number from pattern like "ID_ep5"
            parts = filename.split("_ep")
            if len(parts) == 2:
                id_part = parts[0]
                # Only include IDs that are in the valid MTurk IDs list
                if not valid_mturk_ids or id_part in valid_mturk_ids:
                    all_ids.add(id_part)
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
    start_scene_dir = Path(data_dir) / "overcooked_hh_start_scene"
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    start_scene_ids = set()
    cramped_room_ids = set()
    
    if start_scene_dir.exists():
        for file_path in start_scene_dir.glob("*"):
            id_part = file_path.name.split("_")[0]
            # Only include IDs that are in the valid MTurk IDs list
            if not valid_mturk_ids or id_part in valid_mturk_ids:
                start_scene_ids.add(id_part)
    
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            id_part = file_path.name.split("_")[0]
            # Only include IDs that are in the valid MTurk IDs list
            if not valid_mturk_ids or id_part in valid_mturk_ids:
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
            id_part = file_path.name.split("_")[0]
            # Only include IDs that are in the valid MTurk IDs list
            if not valid_mturk_ids or id_part in valid_mturk_ids:
                end_scene_ids.add(id_part)
    
    # Get all IDs in cramped_room
    all_cramped_room_ids = set()
    if cramped_room_dir.exists():
        for file_path in cramped_room_dir.glob("*"):
            id_part = file_path.name.split("_")[0]
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
    unpaired_data = get_unpaired_subjects()
    sanity_data = get_sanity_checks(episode_data["max_episode_per_subject"], episode_data["overall_max_episode"])
    
    all_ids = episode_data["all_ids"]
    max_episode_per_subject = episode_data["max_episode_per_subject"]
    overall_max_episode = episode_data["overall_max_episode"]
    
    # Categorize subjects
    fully_completed = [id_str for id_str in all_ids if max_episode_per_subject[id_str] == overall_max_episode]
    partially_completed = [id_str for id_str in all_ids if max_episode_per_subject[id_str] < overall_max_episode]
    
    # Print report
    print("\n" + "=" * 70)
    print("DATA COMPLETENESS REPORT")
    print("=" * 70)
    
    # Section 1: Unpaired subjects
    print("\n(1) UNPAIRED SUBJECTS")
    print("-" * 70)
    print(f"Total unique IDs in overcooked_hh_start_scene: {unpaired_data['total_start_scene']}")
    print(f"Subjects NOT in cramped_room_hh: {unpaired_data['count_unpaired']}")
    if unpaired_data['unpaired_ids']:
        print("\nUnpaired Subject IDs:")
        for id_str in unpaired_data['unpaired_ids']:
            print(f"  - {id_str}")
    else:
        print("\n[OK] All subjects in start_scene are in cramped_room_hh")
    
    # Section 2: Partially completed
    print("\n(2) SUBJECTS WITH PARTIAL COMPLETION")
    print("-" * 70)
    print(f"Max episode number: {overall_max_episode}")
    print(f"Subjects who did NOT complete episode {overall_max_episode}: {len(partially_completed)}")
    if partially_completed:
        print("\nPartially Completed Subject IDs:")
        for id_str in partially_completed:
            print(f"  - {id_str} (max episode: {max_episode_per_subject[id_str]})")
    else:
        print("\n[OK] All subjects completed the maximum episode")
    
    # Section 3: Fully completed
    print("\n(3) SUBJECTS WITH FULL COMPLETION")
    print("-" * 70)
    print(f"Subjects who completed episode {overall_max_episode}: {len(fully_completed)}")
    if fully_completed:
        print("\nFully Completed Subject IDs:")
        for id_str in fully_completed:
            print(f"  - {id_str}")
    else:
        print("\n✓ No subjects completed the maximum episode")
    
    # Section 4: Sanity checks
    print("\n(4) SANITY CHECKS")
    print("-" * 70)
    
    print("\nA. Subjects who prematurely ended (incomplete episodes but got completion code):")
    if sanity_data['prematurely_ended']:
        print(f"   Count: {len(sanity_data['prematurely_ended'])}")
        for id_str in sanity_data['prematurely_ended']:
            print(f"   - {id_str}")
    else:
        print("   [OK] None found")
    
    print("\nB. Subjects with completion code but NO episode data:")
    if sanity_data['no_episodes_but_completion']:
        print(f"   Count: {len(sanity_data['no_episodes_but_completion'])}")
        for id_str in sanity_data['no_episodes_but_completion']:
            print(f"   - {id_str}")
    else:
        print("   [OK] None found")
    
    print("\n" + "=" * 70)
