"""Project folder layout for submission."""

import os

CODEBASE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODEBASE)
SAMPLE_DIR = os.path.join(PROJECT_ROOT, "sample_data")
REPORT_DIR = os.path.join(PROJECT_ROOT, "Report")
UPLOADS_DIR = os.path.join(CODEBASE, "uploads", "processed")
