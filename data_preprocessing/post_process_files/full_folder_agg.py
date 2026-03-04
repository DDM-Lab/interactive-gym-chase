import os
import shutil
from pathlib import Path

data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns"
agg_folder_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\full_data"
folder_names = ["console_logs", "cramped_room_hh", "end_completion_code_scene", "match_logs", "multiplayer_feedback_scene", "overcooked_hh_start_scene", "overcooked_tutorial"]

def aggregate_data():
    """
    Iterate through folders in data_dir starting with 'human-only',
    copy files from specified subfolders to agg_folder_dir,
    and rename files by appending the source folder name to avoid conflicts.
    """
    # Get all folders in data_dir that start with 'human-only'
    human_only_folders = [
        f for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f)) and f.startswith("human-only")
    ]
    
    print(f"Found {len(human_only_folders)} 'human-only' folders")
    
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
            os.makedirs(dest_subfolder, exist_ok=True)
            
            # Copy files from source to destination
            files = os.listdir(source_subfolder)
            for file in files:
                source_file = os.path.join(source_subfolder, file)
                
                # Only copy files, not directories
                if not os.path.isfile(source_file):
                    continue
                
                # Split filename and extension
                file_name, file_ext = os.path.splitext(file)
                
                # Create new filename with source folder appended
                new_filename = f"{file_name}_{human_folder}{file_ext}"
                dest_file = os.path.join(dest_subfolder, new_filename)
                
                # Copy the file
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"  ✓ Copied: {file} → {new_filename}")
                except Exception as e:
                    print(f"  ✗ Error copying {file}: {e}")
    
    print("\n✓ Aggregation complete!")

if __name__ == "__main__":
    aggregate_data()