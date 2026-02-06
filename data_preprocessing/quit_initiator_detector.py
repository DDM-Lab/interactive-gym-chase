import json
from pathlib import Path
from typing import Tuple, Optional


def detect_quit_initiator(console_log_1: str, console_log_2: str) -> Tuple[Optional[str], dict]:
    """
    Analyzes two console log files from a multiplayer game session and determines
    which player quit/disconnected first.
    
    Args:
        console_log_1: Path to first player's console.jsonl file
        console_log_2: Path to second player's console.jsonl file
    
    Returns:
        Tuple of (quitter_subject_id, analysis_details_dict)
        - quitter_subject_id: The subject ID of the player who quit first, or None if unclear
        - analysis_details_dict: Dictionary containing:
            - 'subject_1': Subject ID from first log
            - 'subject_2': Subject ID from second log
            - 'quitter': Subject ID of the player who quit
            - 'reason': Explanation of how quitter was determined
            - 'disconnection_timestamp_1': Timestamp of disconnect event in log 1
            - 'disconnection_timestamp_2': Timestamp of disconnect event in log 2
            - 'lost_players': Player slot(s) that disconnected (from game end message)
            - 'player_subjects_mapping': Mapping of player slots to subject IDs
    """
    
    details = {
        'subject_1': None,
        'subject_2': None,
        'quitter': None,
        'reason': None,
        'disconnection_timestamp_1': None,
        'disconnection_timestamp_2': None,
        'lost_players': None,
        'player_subjects_mapping': None
    }
    
    try:
        # Parse both log files
        log1_data = _parse_console_log(console_log_1)
        log2_data = _parse_console_log(console_log_2)
        
        if not log1_data or not log2_data:
            details['reason'] = "Could not parse one or both log files"
            return None, details
        
        # Extract subject IDs
        subject_1 = log1_data.get('subject_id')
        subject_2 = log2_data.get('subject_id')
        details['subject_1'] = subject_1
        details['subject_2'] = subject_2
        
        # Find disconnection events
        disconnect_1 = _find_disconnect_event(log1_data)
        disconnect_2 = _find_disconnect_event(log2_data)
        
        details['disconnection_timestamp_1'] = disconnect_1.get('timestamp') if disconnect_1 else None
        details['disconnection_timestamp_2'] = disconnect_2.get('timestamp') if disconnect_2 else None
        
        # Check for partner disconnection event (indicates this player stayed connected)
        partner_disconnect_1 = disconnect_1 and 'Partner disconnected' in disconnect_1.get('message', '')
        partner_disconnect_2 = disconnect_2 and 'Partner disconnected' in disconnect_2.get('message', '')
        
        # Extract lost_players and player mapping from game end message
        lost_players = None
        player_subjects_mapping = None
        
        if partner_disconnect_1:
            # Subject 1 saw partner disconnect, so subject 2 quit
            game_end_msg = disconnect_1.get('message', '')
            lost_players, player_subjects_mapping = _extract_game_end_info(game_end_msg)
            details['lost_players'] = lost_players
            details['player_subjects_mapping'] = player_subjects_mapping
            details['quitter'] = subject_2
            details['reason'] = f"Subject {subject_2} detected as partner disconnect in {subject_1}'s log"
            return subject_2, details
        
        elif partner_disconnect_2:
            # Subject 2 saw partner disconnect, so subject 1 quit
            game_end_msg = disconnect_2.get('message', '')
            lost_players, player_subjects_mapping = _extract_game_end_info(game_end_msg)
            details['lost_players'] = lost_players
            details['player_subjects_mapping'] = player_subjects_mapping
            details['quitter'] = subject_1
            details['reason'] = f"Subject {subject_1} detected as partner disconnect in {subject_2}'s log"
            return subject_1, details
        
        # If no clear partner disconnect, check for lost_players info
        if disconnect_1:
            lost_players, player_subjects_mapping = _extract_game_end_info(disconnect_1.get('message', ''))
            if lost_players and player_subjects_mapping:
                details['lost_players'] = lost_players
                details['player_subjects_mapping'] = player_subjects_mapping
                
                # Map lost player to subject ID
                for lost_slot in lost_players:
                    if lost_slot in player_subjects_mapping:
                        quitter = player_subjects_mapping[lost_slot]
                        details['quitter'] = quitter
                        details['reason'] = f"Subject {quitter} identified from lost_players field: {lost_players}"
                        return quitter, details
        
        # Fallback 1: Use explicit disconnect timestamp if available
        if disconnect_1 and disconnect_2:
            ts1 = disconnect_1.get('timestamp', float('inf'))
            ts2 = disconnect_2.get('timestamp', float('inf'))
            
            if ts1 < ts2:
                details['quitter'] = subject_1
                details['reason'] = f"Subject {subject_1} had earlier disconnect timestamp ({ts1} vs {ts2})"
                return subject_1, details
            elif ts2 < ts1:
                details['quitter'] = subject_2
                details['reason'] = f"Subject {subject_2} had earlier disconnect timestamp ({ts2} vs {ts1})"
                return subject_2, details
        
        # Fallback 2: Compare last log entry timestamps (for incomplete logs)
        if log1_data and log2_data and log1_data.get('entries') and log2_data.get('entries'):
            last_entry_1 = log1_data['entries'][-1]
            last_entry_2 = log2_data['entries'][-1]
            
            ts1_last = last_entry_1.get('timestamp', float('inf'))
            ts2_last = last_entry_2.get('timestamp', float('inf'))
            
            if ts1_last != ts2_last:
                if ts1_last < ts2_last:
                    details['quitter'] = subject_1
                    details['reason'] = f"Subject {subject_1} had earlier final log entry timestamp ({ts1_last} vs {ts2_last}) - logs appear incomplete"
                    return subject_1, details
                else:
                    details['quitter'] = subject_2
                    details['reason'] = f"Subject {subject_2} had earlier final log entry timestamp ({ts2_last} vs {ts1_last}) - logs appear incomplete"
                    return subject_2, details
        
        details['reason'] = "Could not definitively determine quitter from available data"
        return None, details
        
    except Exception as e:
        details['reason'] = f"Error during analysis: {str(e)}"
        return None, details


def _parse_console_log(file_path: str) -> Optional[dict]:
    """Parse a JSONL console log file and extract relevant data."""
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return None
        
        entries = []
        subject_id = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                    if subject_id is None and 'subject_id' in entry:
                        subject_id = entry['subject_id']
                except json.JSONDecodeError:
                    continue
        
        return {
            'subject_id': subject_id,
            'entries': entries,
            'total_entries': len(entries)
        }
    except Exception:
        return None


def _find_disconnect_event(log_data: dict) -> Optional[dict]:
    """
    Find the disconnect event in a log.
    Looks for key messages indicating disconnection.
    """
    if not log_data or 'entries' not in log_data:
        return None
    
    # Keywords that indicate disconnection/game end
    disconnect_keywords = [
        'Partner disconnected',
        'P2P connection lost',
        'Connection state: disconnected',
        'Game ended due to partner disconnection',
        'Reconnection timeout reached'
    ]
    
    # Search from the end backwards to find the final disconnect event
    for entry in reversed(log_data['entries']):
        message = entry.get('message', '')
        for keyword in disconnect_keywords:
            if keyword in message:
                return {
                    'timestamp': entry.get('timestamp'),
                    'message': message,
                    'level': entry.get('level'),
                    'original_entry': entry
                }
    
    return None


def _extract_game_end_info(message: str) -> Tuple[Optional[list], Optional[dict]]:
    """
    Extract lost_players list and player_subjects mapping from a game end message.
    
    Example message format:
    "[P2P] Game ended due to partner disconnection {\"game_id\":\"...\",\"reason\":\"...\",
    \"reconnection_data\":{...},\"lost_players\":[\"1\"],\"player_subjects\":{\"0\":\"A1AFPXC0FVLK44\",\"1\":\"AIC2D8A9UQO8S\"},...}"
    """
    try:
        # Try to extract JSON from the message
        # Look for the JSON object in curly braces
        start_idx = message.find('{')
        if start_idx == -1:
            return None, None
        
        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(message)):
            if message[i] == '{':
                brace_count += 1
            elif message[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count != 0:
            return None, None
        
        json_str = message[start_idx:end_idx]
        data = json.loads(json_str)
        
        # Extract lost_players and player_subjects
        lost_players = data.get('lost_players')
        reconnection_data = data.get('reconnection_data', {})
        if reconnection_data and 'lost_players' in reconnection_data:
            lost_players = reconnection_data['lost_players']
        
        player_subjects = data.get('player_subjects')
        
        return lost_players, player_subjects
        
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, None


if __name__ == "__main__":
    # Example usage
    log_file_1 = r"c:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\run_1\A1AFPXC0FVLK44_console.jsonl"
    log_file_2 = r"c:\Users\groessli\Documents\GitHub\interactive-gym-chase\data_preprocessing\human_only\aggregated_data\run_1\AIC2D8A9UQO8S_console.jsonl"
    
    quitter, details = detect_quit_initiator(log_file_1, log_file_2)
    
    print("=" * 70)
    print("QUIT INITIATOR DETECTION RESULTS")
    print("=" * 70)
    print(f"Subject 1: {details['subject_1']}")
    print(f"Subject 2: {details['subject_2']}")
    print(f"\nQuitter: {quitter}")
    print(f"Reason: {details['reason']}")
    
    if details['lost_players']:
        print(f"Lost Players (slots): {details['lost_players']}")
    if details['player_subjects_mapping']:
        print(f"Player Subjects Mapping: {details['player_subjects_mapping']}")
    
    print(f"\nDisconnection Timestamp 1: {details['disconnection_timestamp_1']}")
    print(f"Disconnection Timestamp 2: {details['disconnection_timestamp_2']}")
    print("=" * 70)
