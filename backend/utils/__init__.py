"""
Internal trace:
- Wrong before: runtime utility logic was scattered across unrelated packages and cleanup/error behavior was inconsistent.
- Fixed now: shared logging and file helpers live in a single utility package.
"""
