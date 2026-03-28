"""
Internal trace:
- Wrong before: router package mixed upload, live, social, and interview entrypoints without a clear production boundary.
- Fixed now: runtime only imports the health and analyse routers from this package.
"""
