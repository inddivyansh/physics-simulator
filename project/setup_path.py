# setup_path.py - Helper module to fix import paths across the project

import sys
import os

def setup_project_path():
    """Add the project root directory to Python path"""
    # Get the directory containing this file (project/)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add to Python path if not already there
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    
    return project_dir

# Auto-setup when imported
setup_project_path()
