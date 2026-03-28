"""
Internal trace:
- Wrong before: analysis code lived across unrelated packages and mixed low-level media handling with HTTP concerns.
- Fixed now: image, video, and audio inference each have a dedicated pipeline module.
"""
