import os
from pathlib import Path
from collections import defaultdict

#data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed\Data\Pilot2\data"
data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed\Data\FullRun1+2+3\data"

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
    end_scene_files_found = []
    if end_scene_dir.exists():
        # First try to find CSV files
        csv_files = list(end_scene_dir.glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                filename = csv_file.stem
                id_part = filename.split("_")[0]
                end_scene_ids.add(id_part)
                end_scene_files_found.append(filename)
        else:
            # If no CSV files, extract from metadata files
            for meta_file in end_scene_dir.glob("*_metadata.json"):
                filename = meta_file.stem.replace("_metadata", "")
                id_part = filename.split("_")[0]
                end_scene_ids.add(id_part)
                end_scene_files_found.append(filename)
    
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
        "ids_in_end_scene": len(end_scene_ids),
        "end_scene_files_found": end_scene_files_found,
        "all_end_scene_ids": sorted(list(end_scene_ids))
    }


def check_specific_ids_in_end_scene():
    """
    Simple function to list all files in end_completion_code_scene,
    extract IDs, and check if the four specific IDs are present.
    """
    end_scene_dir = Path(data_dir) / "end_completion_code_scene"
    specific_ids = ["ADZK6MRVYIR5D", "A1UP5HLWPM1LAC", "A2WPFFUE15XC92", "A36DK2OP86BTSO"]
    
    print("\n" + "=" * 70)
    print("DIRECT CHECK: ALL FILES IN end_completion_code_scene")
    print("=" * 70)
    
    all_files = list(end_scene_dir.glob("*"))
    print(f"\nTotal files/folders in directory: {len(all_files)}")
    print("\nFile listing (first 30):")
    for i, file_path in enumerate(sorted(all_files)[:30]):
        print(f"  {file_path.name}")
    
    if len(all_files) > 30:
        print(f"  ... and {len(all_files) - 30} more files")
    
    # Extract IDs from all files
    ids_found = set()
    print("\n" + "-" * 70)
    print("Extracting IDs from filenames:")
    print("-" * 70)
    
    for file_path in sorted(all_files):
        filename = file_path.name
        # Try to extract ID (everything before first underscore)
        id_part = filename.split("_")[0]
        ids_found.add(id_part)
    
    print(f"Unique IDs extracted: {len(ids_found)}")
    print(f"Sample IDs: {sorted(list(ids_found))[:10]}")
    
    # Check for specific IDs
    print("\n" + "-" * 70)
    print("Checking for specific IDs:")
    print("-" * 70)
    
    for id_str in specific_ids:
        is_present = id_str in ids_found
        status = "✓ FOUND" if is_present else "✗ NOT FOUND"
        print(f"  {id_str}: {status}")
    
    print("\n" + "=" * 70)


def check_episode_19_completion():
    """
    Check if specific IDs have episode 19 data in cramped_room_sp_0 directory.
    """
    sp_0_dir = Path(data_dir) / "cramped_room_sp_0"
    specific_ids = ["ADZK6MRVYIR5D", "A1UP5HLWPM1LAC", "A2WPFFUE15XC92", "A36DK2OP86BTSO"]
    
    print("\n" + "=" * 70)
    print("EPISODE 19 COMPLETION CHECK: cramped_room_sp_0")
    print("=" * 70)
    
    # Get all CSV files for each specific ID
    for id_str in specific_ids:
        print(f"\nID: {id_str}")
        print("-" * 70)
        
        # Find all files matching this ID
        matching_files = list(sp_0_dir.glob(f"{id_str}*.csv"))
        
        if not matching_files:
            print("  ✗ NO FILES FOUND")
        else:
            print(f"  Files found: {len(matching_files)}")
            
            # Extract episode numbers
            episodes = []
            for file_path in sorted(matching_files):
                filename = file_path.stem  # Remove .csv
                # Extract episode number (everything after the first underscore)
                parts = filename.split("_")
                if len(parts) == 1:
                    # No episode number = episode 0
                    episodes.append(0)
                    print(f"    {file_path.name} → Episode 0")
                else:
                    try:
                        ep_num = int(parts[-1])
                        episodes.append(ep_num)
                        print(f"    {file_path.name} → Episode {ep_num}")
                    except ValueError:
                        print(f"    {file_path.name} → (could not parse episode number)")
            
            # Check if episode 19 exists
            has_ep_19 = 19 in episodes
            status = "✓ YES" if has_ep_19 else "✗ NO"
            print(f"\n  Has Episode 19: {status}")
            print(f"  Episodes present: {sorted(set(episodes))}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run the direct check first
    check_specific_ids_in_end_scene()
    
    # Check episode 19 completion
    check_episode_19_completion()
    
    # Then run the full analysis
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
    
    # Debug: Show all IDs found in end_completion_code_scene
    print("\nDEBUG: All IDs found in end_completion_code_scene:")
    print("-" * 60)
    if results['end_scene_files_found']:
        print(f"Files found: {len(results['end_scene_files_found'])}")
        for file in sorted(results['end_scene_files_found'])[:20]:  # Show first 20
            print(f"  {file}")
        if len(results['end_scene_files_found']) > 20:
            print(f"  ... and {len(results['end_scene_files_found']) - 20} more")
    else:
        print("  (No files found in end_completion_code_scene)")
    
    print(f"\nUnique IDs extracted: {results['ids_in_end_scene']}")
    print(f"Sample IDs: {results['all_end_scene_ids'][:10]}")
    
    print("\n" + "=" * 60)
    
    # Check specific IDs
    specific_ids = ["ADZK6MRVYIR5D", "A1UP5HLWPM1LAC", "A2WPFFUE15XC92", "A36DK2OP86BTSO"]
    print("SPECIFIC ID CHECK")
    print("=" * 60)
    
    for id_str in specific_ids:
        missing_end_scene = id_str in results['missing_from_end_scene']
        missing_ep_19 = id_str in results['missing_episode_19']
        in_end_scene = id_str in results['all_end_scene_ids']
        
        print(f"\nID: {id_str}")
        print(f"  In end_completion_code_scene: {in_end_scene}")
        print(f"  Missing from end_completion_code_scene: {missing_end_scene}")
        print(f"  Missing episode 19: {missing_ep_19}")
    
    print("\n" + "=" * 60)
