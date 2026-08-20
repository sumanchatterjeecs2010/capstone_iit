"""Project folder layout for the assistant (Codebase runtime only)."""

import os

CODEBASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODEBASE)
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data")
UPLOADS_DIR = os.path.join(CODEBASE, "uploads", "processed")
