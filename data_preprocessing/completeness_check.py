import os
from pathlib import Path
from collections import defaultdict

data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed\Data\Pilot2\data"


def analyze_data_completeness():
    """
    Analyze the data in the data directory and report:
    (1) IDs present in cramped_room_sp_0 but NOT in end_completion_code_scene
    (2) IDs that do not have a CSV file with "_19" suffix
    
    Returns:
        dict: A dictionary containing 'missing_from_end_scene' and 'missing_episode_19' lists
    """
    
    # Define directory paths
    sp_0_dir = Path(data_dir) / "cramped_room_sp_0"
    end_scene_dir = Path(data_dir) / "end_completion_code_scene"
    
    # Extract IDs from cramped_room_sp_0 CSV files
    sp_0_ids = set()
    if sp_0_dir.exists():
        for csv_file in sp_0_dir.glob("*.csv"):
            # Extract ID (part before first underscore)
            filename = csv_file.stem  # Remove .csv extension
            id_part = filename.split("_")[0]
            sp_0_ids.add(id_part)
    
    # Extract IDs from end_completion_code_scene (from metadata files or CSV if present)
    end_scene_ids = set()
    if end_scene_dir.exists():
        # First try to find CSV files
        csv_files = list(end_scene_dir.glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                filename = csv_file.stem
                id_part = filename.split("_")[0]
                end_scene_ids.add(id_part)
        else:
            # If no CSV files, extract from metadata files
            for meta_file in end_scene_dir.glob("*_metadata.json"):
                filename = meta_file.stem.replace("_metadata", "")
                id_part = filename.split("_")[0]
                end_scene_ids.add(id_part)
    
    # (1) IDs in sp_0 but NOT in end_scene
    missing_from_end_scene = sorted(list(sp_0_ids - end_scene_ids))
    
    # (2) IDs that don't have episode 19
    ids_with_episode_19 = set()
    if sp_0_dir.exists():
        for csv_file in sp_0_dir.glob("*_19.csv"):
            filename = csv_file.stem
            id_part = filename.split("_")[0]
            ids_with_episode_19.add(id_part)
    
    missing_episode_19 = sorted(list(sp_0_ids - ids_with_episode_19))
    
    return {
        "missing_from_end_scene": missing_from_end_scene,
        "missing_episode_19": missing_episode_19,
        "total_ids_in_sp_0": len(sp_0_ids),
        "ids_in_end_scene": len(end_scene_ids)
    }


if __name__ == "__main__":
    results = analyze_data_completeness()
    
    print("=" * 60)
    print("DATA COMPLETENESS ANALYSIS")
    print("=" * 60)
    print(f"\nTotal unique IDs in cramped_room_sp_0: {results['total_ids_in_sp_0']}")
    print(f"IDs in end_completion_code_scene: {results['ids_in_end_scene']}")
    
    print("\n" + "-" * 60)
    print("(1) IDs in cramped_room_sp_0 but NOT in end_completion_code_scene:")
    print("-" * 60)
    if results['missing_from_end_scene']:
        for id_str in results['missing_from_end_scene']:
            print(f"  - {id_str}")
    else:
        print("  (None - all IDs completed the end scene)")
    
    print("\n" + "-" * 60)
    print("(2) IDs that do NOT have a CSV file with episode '_19':")
    print("-" * 60)
    if results['missing_episode_19']:
        for id_str in results['missing_episode_19']:
            print(f"  - {id_str}")
    else:
        print("  (None - all IDs have episode 19)")
    
    print("\n" + "=" * 60)
