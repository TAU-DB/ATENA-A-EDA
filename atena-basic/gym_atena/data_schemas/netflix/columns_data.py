"""
This file contains all data with regard to columns in the netflix schema
"""

KEYS = ['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added', 'release_year', 'rating',
        'duration', 'listed_in', 'description']
KEYS_ANALYST_STR = KEYS
assert len(KEYS) == len(KEYS_ANALYST_STR)
FILTER_COLS = KEYS  # Note: changing this from KEYS require to change other occurrences of KEYS in the codebase
GROUP_COLS = KEYS  # Note: changing this from KEYS require to change other occurrences of KEYS in the codebase

NUMERIC_KEYS = {'show_id', 'release_year'}
AGG_KEYS = ['show_id']
AGG_KEYS_ANALYST_STR = ['show_id']
DONT_FILTER_FIELDS = {'show_id'}
