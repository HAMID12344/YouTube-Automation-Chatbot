import os
import sys
import runpy

# Ensure project directory is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

app_path = os.path.join(PROJECT_DIR, "app.py")
runpy.run_path(app_path, run_name="__main__")
