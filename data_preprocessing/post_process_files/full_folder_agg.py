import os
import shutil
import json
from pathlib import Path
from datetime import datetime

data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns"
agg_folder_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\full_data"
folder_names = ["console_logs", "cramped_room_hh", "end_completion_code_scene", "match_logs", "multiplayer_feedback_scene", "overcooked_hh_start_scene", "overcooked_tutorial"]

# Log file to track successful and failed copies
LOG_FILE = os.path.join(os.path.dirname(__file__), "copy_log.json")

def load_copy_log():
    """Load the copy log from file."""
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load log file: {e}")
            return {"successful": [], "failed": []}
    return {"successful": [], "failed": []}

def save_copy_log(log_data):
    """Save the copy log to file."""
    try:
        with open(LOG_FILE, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save log file: {e}")

def get_file_key(human_folder, subfolder_name, file):
    """Generate a unique key for a file for tracking purposes."""
    return f"{human_folder}/{subfolder_name}/{file}"

def get_long_path(path):
    """Convert a path to Windows extended-length path format for pathlib.Path."""
    abs_path = str(Path(path).resolve())
    # On Windows, add the \\?\ prefix for network paths (and long paths)
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        # Handle UNC paths (network paths)
        if abs_path.startswith('\\\\'):
            abs_path = '\\\\?\\UNC\\' + abs_path[2:]
        # Handle regular paths
        else:
            abs_path = '\\\\?\\' + abs_path
    return abs_path

def aggregate_data():
    """
    Iterate through folders in data_dir starting with 'human-only',
    copy files from specified subfolders to agg_folder_dir,
    and rename files by appending the source folder name to avoid conflicts.
    Only copies files that haven't been successfully copied before.
    """
    # Load existing log
    log_data = load_copy_log()
    successful_files = set(log_data.get("successful", []))
    failed_files = set(log_data.get("failed", []))
    
    # Get all folders in data_dir that start with 'human-only'
    human_only_folders = [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("human-only")
    ]
    
    print(f"Found {len(human_only_folders)} 'human-only' folders")
    print(f"Previously successful copies: {len(successful_files)}")
    print(f"Previously failed copies: {len(failed_files)}")
    
    files_skipped = 0
    files_copied = 0
    files_failed = 0
    
    # Iterate through each human-only folder
    for human_folder in human_only_folders:
        human_folder_path = os.path.join(data_dir, human_folder)
        print(f"\nProcessing: {human_folder}")
        
        # Iterate through each subfolder specified in folder_names
        for subfolder_name in folder_names:
            source_subfolder = os.path.join(human_folder_path, subfolder_name)
            dest_subfolder = os.path.join(agg_folder_dir, subfolder_name)
            
            # Check if source subfolder exists
            if not os.path.isdir(source_subfolder):
                print(f"  ⚠ Source subfolder not found: {subfolder_name}")
                continue
            
            # Create destination subfolder if it doesn't exist
            dest_path = Path(get_long_path(dest_subfolder))
            dest_path.mkdir(parents=True, exist_ok=True)
            
            # Copy files from source to destination
            files = os.listdir(source_subfolder)
            for file in files:
                source_file = os.path.join(source_subfolder, file)
                
                # Only copy files, not directories
                if not os.path.isfile(source_file):
                    continue
                
                # Generate file key for tracking
                file_key = get_file_key(human_folder, subfolder_name, file)
                
                # Skip if already successfully copied
                if file_key in successful_files:
                    files_skipped += 1
                    continue
                
                # Split filename and extension
                file_name, file_ext = os.path.splitext(file)
                
                # Create new filename with source folder appended
                new_filename = f"{file_name}_{human_folder}{file_ext}"
                dest_file = os.path.join(dest_subfolder, new_filename)
                
                # Check if destination file already exists
                dest_file_path = Path(get_long_path(dest_file))
                if dest_file_path.exists():
                    # File already exists, mark as successful and skip
                    successful_files.add(file_key)
                    if file_key in failed_files:
                        failed_files.remove(file_key)
                    files_skipped += 1
                    continue
                
                # Copy the file
                try:
                    # Use long path format for destination to handle long file paths on Windows
                    dest_file_long = get_long_path(dest_file)
                    shutil.copy2(source_file, dest_file_long)
                    print(f"  ✓ Copied: {file} → {new_filename}")
                    
                    # Mark as successful
                    successful_files.add(file_key)
                    if file_key in failed_files:
                        failed_files.remove(file_key)
                    files_copied += 1
                    
                except Exception as e:
                    print(f"  ✗ Error copying {file}: {e}")
                    failed_files.add(file_key)
                    files_failed += 1
    
    # Save updated log
    log_data["successful"] = sorted(list(successful_files))
    log_data["failed"] = sorted(list(failed_files))
    log_data["last_run"] = datetime.now().isoformat()
    save_copy_log(log_data)
    
    print(f"\n✓ Aggregation complete!")
    print(f"  Files skipped (already copied): {files_skipped}")
    print(f"  Files copied: {files_copied}")
    print(f"  Files failed: {files_failed}")
    print(f"  Total tracked successful: {len(successful_files)}")
    print(f"  Total tracked failed: {len(failed_files)}")

def analyze_subject_ids():
    """
    Analyze and report the number of unique subject IDs in each subfolder of agg_folder_dir.
    Subject IDs are extracted from the beginning of filenames (before the first underscore).
    """
    print(f"Analyzing subject IDs in: {agg_folder_dir}\n")
    
    overall_unique_ids = set()
    results = {}
    
    # Iterate through each subfolder
    for subfolder_name in folder_names:
        subfolder_path = Path(agg_folder_dir) / subfolder_name
        
        if not subfolder_path.exists():
            print(f"⚠ Subfolder not found: {subfolder_name}")
            results[subfolder_name] = 0
            continue
        
        unique_ids = set()
        file_count = 0
        
        # List all files in the subfolder
        try:
            files = list(subfolder_path.iterdir())
            for file_path in files:
                if file_path.is_file():
                    file_count += 1
                    # Extract subject ID (everything before the first underscore)
                    filename = file_path.name
                    subject_id = filename.split('_')[0]
                    unique_ids.add(subject_id)
                    overall_unique_ids.add(subject_id)
        except Exception as e:
            print(f"✗ Error reading {subfolder_name}: {e}")
            results[subfolder_name] = 0
            continue
        
        results[subfolder_name] = len(unique_ids)
        print(f"  {subfolder_name}")
        print(f"    Unique subject IDs: {len(unique_ids)}")
        print(f"    Total files: {file_count}")
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for subfolder_name, unique_count in results.items():
        print(f"  {subfolder_name}: {unique_count} unique IDs")
    
    print(f"\nTotal unique subject IDs across all folders: {len(overall_unique_ids)}")
    print("=" * 60)



if __name__ == "__main__":
    analyze_subject_ids()
