# Categorization Bug Analysis and Fix

## Problem Summary

Both `completeness_check.py` and `data_aggregation.py` had a subtle bug in how they extracted subject IDs from filenames with multiple underscores (like `grace_t_6`).

## Root Cause

The issue was in ID extraction from non-episode files:

**Problematic code:**
```python
id_part = file_path.stem.split("_")[0]
```

**What happened:**
- File: `grace_t_6_metadata.json`
- After `.stem`: `grace_t_6_metadata`
- After `split("_")[0]`: `grace` ← **WRONG!**

Meanwhile, episode files were parsed correctly:
```python
id_part = filename.split("_ep")[0]  # Correctly returns 'grace_t_6'
```

This created **two different ID representations**:
- `'grace'` in start_scene_ids (from metadata files)
- `'grace_t_6'` in max_episode_per_subject (from episode files)

**Result:**
- `grace_t_6` was never found when looking for completed subjects in start_scene_ids
- Both `'grace'` and `'grace_t_6'` ended up in Category 5 (uncategorized)

## Solution

Introduced a new `extract_subject_id()` function that handles multi-underscore IDs correctly:

```python
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
```

## Files Modified

1. **completeness_check.py**
   - Added `extract_subject_id()` function
   - Updated `get_unpaired_subjects()` to use it
   - Updated `get_sanity_checks()` to use it
   - Main block already updated to use it

2. **data_aggregation.py**
   - Added `extract_subject_id()` function
   - Updated `categorize_subjects()` to use it

## Verification

Before fix:
- `grace_t_6`: Category 5 (uncategorized)
- `jf0209t1`: Category 4 (completed)

After fix:
- `grace_t_6`: Category 4 (completed) ✓
- `jf0209t1`: Category 4 (completed) ✓
- Both scripts now produce identical categorizations ✓

## Impact

This fix ensures that:
- Subject IDs with multiple underscores are parsed consistently
- Both scripts generate the same categorizations
- Data aggregation correctly processes all subject data
- Pilot data with atypical ID formats (like `grace_t_6`) is handled properly
