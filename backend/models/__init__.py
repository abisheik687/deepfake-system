"""
Internal trace:
- Wrong before: the models package contained mock/demo implementations alongside production code without a clean loader boundary.
- Fixed now: the new runtime uses dedicated loader, ensemble, image, and audio modules from this package.
"""
