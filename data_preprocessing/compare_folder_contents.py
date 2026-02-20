# Script will compare the contents of two folders and report:
# 1. Files that are in folder A but not in folder B
# 2. Files that are in folder B but not in folder A
# 3. The participant IDs that are in both folders
# 4. The particiapnt IDs that are only in folder A
# 5. The participant IDs that are only in folder B

import os
from pathlib import Path

folder_a = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-run-7\cramped_room_hh"
folder_b = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-run-8\cramped_room_hh"

def extract_participant_id(filename):
    """
    Extract participant ID from filename.
    If filename contains underscore, take part before first underscore.
    Otherwise, return entire filename (without extension).
    """
    name_without_ext = Path(filename).stem
    if "_" in name_without_ext:
        return name_without_ext.split("_")[0]
    return name_without_ext

def get_files_without_extension(folder_path):
    """Get all files in folder as filenames without extensions."""
    if not os.path.exists(folder_path):
        print(f"Warning: Folder does not exist: {folder_path}")
        return set()
    
    files = set()
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            name_without_ext = Path(filename).stem
            files.add(name_without_ext)
    return files

def get_all_files(folder_path):
    """Get all files in folder with full filenames (without extension)."""
    if not os.path.exists(folder_path):
        print(f"Warning: Folder does not exist: {folder_path}")
        return set()
    
    files = set()
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            name_without_ext = Path(filename).stem
            files.add(name_without_ext)
    return files

def get_participant_ids(folder_path):
    """Get all participant IDs from files in folder."""
    if not os.path.exists(folder_path):
        print(f"Warning: Folder does not exist: {folder_path}")
        return set()
    
    participant_ids = set()
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            participant_id = extract_participant_id(filename)
            participant_ids.add(participant_id)
    return participant_ids

# Get files from both folders
files_a = get_all_files(folder_a)
files_b = get_all_files(folder_b)

# Get participant IDs from both folders
ids_a = get_participant_ids(folder_a)
ids_b = get_participant_ids(folder_b)

# 1. Files in folder A but not in folder B
files_only_in_a = files_a - files_b
print("=" * 80)
print("1. FILES IN FOLDER A BUT NOT IN FOLDER B:")
print("=" * 80)
if files_only_in_a:
    for file in sorted(files_only_in_a):
        print(f"  {file}")
else:
    print("  (none)")
print()

# 2. Files in folder B but not in folder A
files_only_in_b = files_b - files_a
print("=" * 80)
print("2. FILES IN FOLDER B BUT NOT IN FOLDER A:")
print("=" * 80)
if files_only_in_b:
    for file in sorted(files_only_in_b):
        print(f"  {file}")
else:
    print("  (none)")
print()

# 3. Participant IDs in both folders
ids_in_both = ids_a & ids_b
print("=" * 80)
print("3. PARTICIPANT IDS IN BOTH FOLDERS:")
print("=" * 80)
print(f"Count: {len(ids_in_both)}")
if ids_in_both:
    for participant_id in sorted(ids_in_both):
        print(f"  {participant_id}")
else:
    print("  (none)")
print()

# 4. Participant IDs only in folder A
ids_only_in_a = ids_a - ids_b
print("=" * 80)
print("4. PARTICIPANT IDS ONLY IN FOLDER A:")
print("=" * 80)
print(f"Count: {len(ids_only_in_a)}")
if ids_only_in_a:
    for participant_id in sorted(ids_only_in_a):
        print(f"  {participant_id}")
else:
    print("  (none)")
print()

# 5. Participant IDs only in folder B
ids_only_in_b = ids_b - ids_a
print("=" * 80)
print("5. PARTICIPANT IDS ONLY IN FOLDER B:")
print("=" * 80)
print(f"Count: {len(ids_only_in_b)}")
if ids_only_in_b:
    for participant_id in sorted(ids_only_in_b):
        print(f"  {participant_id}")
else:
    print("  (none)")
print()

