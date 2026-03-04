"""
Compare how completeness_check.py and data_aggregation.py categorize subjects.
This script shows the key differences in categorization logic.
"""

from pathlib import Path

data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-post-pilot"
suffix = "hh"


def extract_subject_id(filename):
    """
    Extract subject ID from filename, handling multi-underscore IDs like 'grace_t_6'.
    For episode files (e.g., 'grace_t_6_ep0.csv'), splits on '_ep' to get the ID.
    For other files (e.g., 'grace_t_6_metadata.json'), removes known suffixes.
    """
    # For episode files, split on '_ep'
    if '_ep' in filename:
        id_part = filename.split('_ep')[0]
    else:
        # Remove known suffixes for non-episode files
        known_suffixes = ['_metadata', '_globals', '_multiplayer_metrics']
        id_part = filename
        for suffix_str in known_suffixes:
            if id_part.endswith(suffix_str):
                id_part = id_part[:-len(suffix_str)]
                break
    return id_part

# ================================================================
# COMPLETENESS_CHECK APPROACH
# ================================================================
print("\n" + "=" * 80)
print("COMPLETENESS_CHECK APPROACH")
print("=" * 80)

# Get all scene IDs
start_scene_dir = Path(data_dir) / f"overcooked_{suffix}_start_scene"
cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
end_scene_dir = Path(data_dir) / "end_completion_code_scene"

start_scene_ids = set()
if start_scene_dir.exists():
    for file_path in start_scene_dir.glob("*"):
        id_part = extract_subject_id(file_path.stem)
        start_scene_ids.add(id_part)
        print(f"  Found in start_scene: {file_path.stem} -> ID: {id_part}")

cramped_room_ids = set()
if cramped_room_dir.exists():
    for file_path in cramped_room_dir.glob("*"):
        if file_path.is_file():
            id_part = extract_subject_id(file_path.stem)
            cramped_room_ids.add(id_part)

end_scene_ids = set()
if end_scene_dir.exists():
    for file_path in end_scene_dir.glob("*"):
        id_part = extract_subject_id(file_path.stem)
        end_scene_ids.add(id_part)

# Get episode data
max_episode_per_subject = {}
if cramped_room_dir.exists():
    for file_path in cramped_room_dir.glob("*_ep*.csv"):
        filename = file_path.stem
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

print(f"\nstart_scene_ids: {start_scene_ids}")
print(f"cramped_room_ids: {cramped_room_ids}")
print(f"max_episode_per_subject: {max_episode_per_subject}")
print(f"overall_max_episode: {overall_max_episode}")

# Categorization in completeness_check.py (lines 306-321)
failed_to_reach = sorted([id_str for id_str in start_scene_ids 
                         if id_str not in cramped_room_ids and id_str not in end_scene_ids])

failed_to_play = sorted([id_str for id_str in start_scene_ids 
                        if id_str in cramped_room_ids and id_str not in end_scene_ids
                        and id_str not in max_episode_per_subject])

team_ended = sorted([id_str for id_str in start_scene_ids 
                    if id_str in cramped_room_ids and id_str not in end_scene_ids
                    and id_str in max_episode_per_subject])

completed_subjects = sorted([id_str for id_str in start_scene_ids 
                            if id_str in max_episode_per_subject 
                            and max_episode_per_subject[id_str] == overall_max_episode])

all_categorized = set(failed_to_reach + failed_to_play + team_ended + completed_subjects)
all_ids_universe_cc = start_scene_ids.union(cramped_room_ids).union(end_scene_ids).union(set(max_episode_per_subject.keys()))
uncategorized_cc = sorted([id_str for id_str in all_ids_universe_cc if id_str not in all_categorized])

print(f"\n✓ Category 1 (failed_to_reach): {failed_to_reach}")
print(f"✓ Category 2 (failed_to_play): {failed_to_play}")
print(f"✓ Category 3 (team_ended): {team_ended}")
print(f"✓ Category 4 (completed_subjects): {completed_subjects}")
print(f"✓ Category 5 (uncategorized): {uncategorized_cc}")

# ================================================================
# DATA_AGGREGATION APPROACH
# ================================================================
print("\n" + "=" * 80)
print("DATA_AGGREGATION APPROACH")
print("=" * 80)

# Build team mapping
import csv

id_to_teammate = {}
processed_ids = set()

if cramped_room_dir.exists():
    for csv_file in sorted(cramped_room_dir.glob("*_ep0.csv")):
        filename = csv_file.stem
        id_part = filename.split("_ep")[0]
        
        if id_part in processed_ids:
            continue
        processed_ids.add(id_part)
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                first_row = next(reader, None)
                if first_row:
                    player_0 = first_row.get("player_subjects.0", "").strip()
                    player_1 = first_row.get("player_subjects.1", "").strip()
                    if player_0 and player_1:
                        id_to_teammate[player_0] = player_1
                        id_to_teammate[player_1] = player_0
                        print(f"  Team mapping from {csv_file.name}: {player_0} <-> {player_1}")
        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")

print(f"\nBuilt id_to_teammate: {id_to_teammate}")

# Categorization in data_aggregation.py - categorize_subjects() function
# Get IDs from each scene (same logic as completeness_check)
start_scene_ids_da = set()
if start_scene_dir.exists():
    for file_path in start_scene_dir.glob("*"):
        id_part = extract_subject_id(file_path.stem)
        start_scene_ids_da.add(id_part)

cramped_room_ids_da = set()
if cramped_room_dir.exists():
    for file_path in cramped_room_dir.glob("*"):
        if file_path.is_file():
            id_part = extract_subject_id(file_path.stem)
            cramped_room_ids_da.add(id_part)

end_scene_ids_da = set()
if end_scene_dir.exists():
    for file_path in end_scene_dir.glob("*"):
        id_part = extract_subject_id(file_path.stem)
        end_scene_ids_da.add(id_part)

max_episode_per_subject_da = {}
if cramped_room_dir.exists():
    for file_path in cramped_room_dir.glob("*_ep*.csv"):
        filename = file_path.stem
        parts = filename.split("_ep")
        if len(parts) == 2:
            id_part = parts[0]
            try:
                ep_num = int(parts[1])
                if id_part not in max_episode_per_subject_da:
                    max_episode_per_subject_da[id_part] = ep_num
                else:
                    max_episode_per_subject_da[id_part] = max(max_episode_per_subject_da[id_part], ep_num)
            except ValueError:
                pass

overall_max_episode_da = max(max_episode_per_subject_da.values()) if max_episode_per_subject_da else 0

# Category 1: Failed to reach main session
category_1 = sorted([id_str for id_str in start_scene_ids_da 
                     if id_str not in cramped_room_ids_da and id_str not in end_scene_ids_da])

# Category 2: Failed to play main session
category_2 = sorted([id_str for id_str in start_scene_ids_da 
                    if id_str in cramped_room_ids_da and id_str not in end_scene_ids_da
                    and id_str not in max_episode_per_subject_da])

# Category 4: Completed experiment
category_4 = []
for id_str in start_scene_ids_da:
    # Own data complete
    if id_str in max_episode_per_subject_da and max_episode_per_subject_da[id_str] == overall_max_episode_da:
        category_4.append(id_str)
        print(f"  {id_str}: Added to category_4 (own data complete: {max_episode_per_subject_da[id_str]} == {overall_max_episode_da})")
    # Own data incomplete, but partner completed all episodes
    elif id_str in id_to_teammate:
        partner_id = id_to_teammate[id_str]
        if partner_id in max_episode_per_subject_da and max_episode_per_subject_da[partner_id] == overall_max_episode_da:
            category_4.append(id_str)
            print(f"  {id_str}: Added to category_4 (partner {partner_id} completed)")
    else:
        print(f"  {id_str}: NOT added to category_4 (own: {max_episode_per_subject_da.get(id_str, 'missing')} vs {overall_max_episode_da}, no partner)")

category_4 = sorted(category_4)

# Category 3: Team ended during main session
category_3 = []
for id_str in start_scene_ids_da:
    # Not in category 1 or 2, and not in category 4
    if id_str not in category_1 and id_str not in category_2 and id_str not in category_4:
        # Must have some episode data
        if id_str in cramped_room_ids_da and id_str in max_episode_per_subject_da:
            # If subject is in a team, check partner status
            if id_str in id_to_teammate:
                partner_id = id_to_teammate[id_str]
                # Only add to category 3 if partner also has incomplete data
                if partner_id in max_episode_per_subject_da and max_episode_per_subject_da[partner_id] < overall_max_episode_da:
                    category_3.append(id_str)
                # If partner not in data at all, still add to category 3
                elif partner_id not in max_episode_per_subject_da:
                    category_3.append(id_str)
            else:
                # Not in a team pair, but has incomplete data
                category_3.append(id_str)

category_3 = sorted(category_3)

# Category 5: Uncategorized
all_categorized_da = set(category_1 + category_2 + category_3 + category_4)
all_ids_universe_da = start_scene_ids_da.union(cramped_room_ids_da).union(end_scene_ids_da).union(set(max_episode_per_subject_da.keys()))
category_5 = sorted([id_str for id_str in all_ids_universe_da if id_str not in all_categorized_da])

print(f"\n✓ Category 1 (failed_to_reach): {category_1}")
print(f"✓ Category 2 (failed_to_play): {category_2}")
print(f"✓ Category 3 (team_ended): {category_3}")
print(f"✓ Category 4 (completed): {category_4}")
print(f"✓ Category 5 (uncategorized): {category_5}")

# ================================================================
# COMPARISON
# ================================================================
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print(f"\nCategory 4 match? {set(completed_subjects) == set(category_4)}")
if set(completed_subjects) != set(category_4):
    print(f"  completeness_check: {completed_subjects}")
    print(f"  data_aggregation:   {category_4}")
    print(f"  Missing in data_agg: {set(completed_subjects) - set(category_4)}")
    print(f"  Extra in data_agg: {set(category_4) - set(completed_subjects)}")

print(f"\nCategory 5 match? {set(uncategorized_cc) == set(category_5)}")
if set(uncategorized_cc) != set(category_5):
    print(f"  completeness_check: {uncategorized_cc}")
    print(f"  data_aggregation:   {category_5}")
    print(f"  Missing in data_agg: {set(uncategorized_cc) - set(category_5)}")
    print(f"  Extra in data_agg: {set(category_5) - set(uncategorized_cc)}")
