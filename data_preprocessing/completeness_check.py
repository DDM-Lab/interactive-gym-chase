from pathlib import Path
import csv
from quit_initiator_detector import detect_quit_initiator

#data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-data_run_2"
data_dir = r"G:\.shortcut-targets-by-id\1n7peZVybcw0B7smQ0VFfiXWcIbYjxZ96\2025ControllableCollaborationChaseGrace\Experiments\2025-ControllableCollaboration-Human Speed-HH\Data\FullRuns\human-only-run-1-aws"
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


def get_team_pair_from_csv(csv_file_path):
    """
    Extract team pair (player_subjects.0 and player_subjects.1) from first row of CSV.
    Returns: tuple (player_0_id, player_1_id) or None if extraction fails
    """
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            first_row = next(reader, None)
            if first_row:
                player_0 = first_row.get("player_subjects.0", "").strip()
                player_1 = first_row.get("player_subjects.1", "").strip()
                if player_0 and player_1:
                    return (player_0, player_1)
    except Exception:
        pass
    return None


def build_team_mapping(max_episode_per_subject, overall_max_episode):
    """
    Build team mapping by reading CSV files to extract player_subjects.
    Returns dict with team pairs and related info.
    """
    cramped_room_dir = Path(data_dir) / f"cramped_room_{suffix}"
    
    id_to_teammates = {}  # Maps each ID to its teammate
    team_pairs = {}  # Maps frozenset{id1, id2} -> episode info
    
    # Find one CSV per ID and extract team pair
    if cramped_room_dir.exists():
        processed_ids = set()
        
        for csv_file in cramped_room_dir.glob("*_ep0.csv"):
            filename = csv_file.stem
            id_part = filename.split("_ep")[0]
            
            if id_part in processed_ids:
                continue
            processed_ids.add(id_part)
            
            team_pair = get_team_pair_from_csv(csv_file)
            if team_pair:
                player_0, player_1 = team_pair
                id_to_teammates[player_0] = player_1
                id_to_teammates[player_1] = player_0
                
                # Normalize team pair (use frozenset to avoid duplicates)
                team_key = frozenset([player_0, player_1])
                if team_key not in team_pairs:
                    team_pairs[team_key] = {
                        "players": (player_0, player_1),
                        "episodes": {}
                    }
                
                # Store episode info for both players
                team_pairs[team_key]["episodes"][player_0] = max_episode_per_subject.get(player_0, -1)
                team_pairs[team_key]["episodes"][player_1] = max_episode_per_subject.get(player_1, -1)
    
    return {
        "id_to_teammates": id_to_teammates,
        "team_pairs": team_pairs
    }


def get_completed_teams(max_episode_per_subject, overall_max_episode, team_mapping):
    """
    Get teams where BOTH members completed all episodes.
    """
    completed_teams = []
    single_member_teams = []
    incomplete_teams = []
    
    for team_key, team_info in team_mapping["team_pairs"].items():
        player_0, player_1 = team_info["players"]
        ep_0 = team_info["episodes"].get(player_0, -1)
        ep_1 = team_info["episodes"].get(player_1, -1)
        
        # Both members completed
        if ep_0 == overall_max_episode and ep_1 == overall_max_episode:
            completed_teams.append((player_0, player_1))
        # Only one member completed
        elif ep_0 == overall_max_episode or ep_1 == overall_max_episode:
            incomplete_teams.append({
                "players": (player_0, player_1),
                "completed": player_0 if ep_0 == overall_max_episode else player_1,
                "incomplete": player_1 if ep_0 == overall_max_episode else player_0,
                "ep_0": ep_0,
                "ep_1": ep_1
            })
    
    return {
        "completed_teams": sorted(completed_teams),
        "incomplete_teams": incomplete_teams
    }


def get_team_ended(max_episode_per_subject, overall_max_episode, team_mapping):
    """
    Get teams that ended prematurely (at least one member with data but max_episode < overall_max).
    """
    team_ended_teams = []
    
    for team_key, team_info in team_mapping["team_pairs"].items():
        player_0, player_1 = team_info["players"]
        ep_0 = team_info["episodes"].get(player_0, -1)
        ep_1 = team_info["episodes"].get(player_1, -1)
        
        # At least one has data, but neither completed all episodes
        if (ep_0 >= 0 or ep_1 >= 0) and ep_0 < overall_max_episode and ep_1 < overall_max_episode:
            team_ended_teams.append({
                "players": (player_0, player_1),
                "ep_0": ep_0,
                "ep_1": ep_1
            })
    
    return sorted(team_ended_teams, key=lambda x: (x["ep_0"], x["ep_1"]), reverse=True)


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


def get_quitter_for_team(player_0, player_1, console_logs_dir):
    """
    Attempt to determine which player quit first by analyzing console logs.
    
    Args:
        player_0: First player's subject ID
        player_1: Second player's subject ID
        console_logs_dir: Path to directory containing console.jsonl files
    
    Returns:
        tuple (quitter_id, reason) or (None, reason_why_not_found)
    """
    console_logs_dir = Path(console_logs_dir)
    
    # Build paths to console logs
    log_file_1 = console_logs_dir / f"{player_0}_console.jsonl"
    log_file_2 = console_logs_dir / f"{player_1}_console.jsonl"
    
    # Check if both files exist
    if not log_file_1.exists():
        return None, f"Console log not found for {player_0}"
    if not log_file_2.exists():
        return None, f"Console log not found for {player_1}"
    
    try:
        quitter, details = detect_quit_initiator(str(log_file_1), str(log_file_2))
        return quitter, details.get('reason', 'Unknown reason')
    except Exception as e:
        return None, f"Error analyzing logs: {str(e)}"


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
    
    # Build team mapping
    team_mapping = build_team_mapping(max_episode_per_subject, overall_max_episode)
    completed_team_data = get_completed_teams(max_episode_per_subject, overall_max_episode, team_mapping)
    team_ended_data = get_team_ended(max_episode_per_subject, overall_max_episode, team_mapping)
    
    # Category 3: Teams that ended prematurely
    print("\n(3) TEAMS THAT ENDED PREMATURELY")
    print("-" * 70)
    print(f"Count: {len(team_ended_data)} teams")
    print("Criteria: At least one team member has data, but neither completed all episodes")
    
    # Determine console logs directory
    console_logs_dir = Path(data_dir) / "console_logs"
    
    if team_ended_data:
        print("\nTeam Pairs (Player0, Player1) [Episodes reached by each player]:")
        for team_info in team_ended_data:
            p0, p1 = team_info["players"]
            ep0, ep1 = team_info["ep_0"], team_info["ep_1"]
            print(f"  - ({p0}, {p1}) [Player0: {ep0}, Player1: {ep1}]", end="")
            
            # Try to determine quitter
            if console_logs_dir.exists():
                quitter, reason = get_quitter_for_team(p0, p1, console_logs_dir)
                if quitter:
                    print(f" → QUITTER: {quitter}")
                else:
                    print(f" → {reason}")
            else:
                print(f" → Console logs directory not found at {console_logs_dir}")
    else:
        print("\n[OK] None found")
    
    # Category 4: Completed teams
    print("\n(4) TEAMS THAT COMPLETED THE EXPERIMENT")
    print("-" * 70)
    print(f"Count: {len(completed_team_data['completed_teams'])} teams ({len(completed_team_data['completed_teams']) * 2} individuals)")
    print(f"Criteria: BOTH team members have max number of episodes ({overall_max_episode})")
    if completed_team_data["completed_teams"]:
        print("\nCompleted Team Pairs:")
        for p0, p1 in completed_team_data["completed_teams"]:
            print(f"  - ({p0}, {p1})")
    else:
        print("\n[OK] None found")
    
    # Data integrity warning: incomplete teams
    if completed_team_data["incomplete_teams"]:
        print(f"\n⚠️  DATA INTEGRITY WARNING: {len(completed_team_data['incomplete_teams'])} teams with only ONE member completing")
        print("   These teams have unequal data and may need to be excluded from analysis:")
        for team in completed_team_data["incomplete_teams"]:
            p0, p1 = team["players"]
            completed = team["completed"]
            incomplete = team["incomplete"]
            print(f"  - ({p0}, {p1}): {completed} completed, {incomplete} completed {team['ep_' + ('0' if p0 == incomplete else '1')]} episodes")
    
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
