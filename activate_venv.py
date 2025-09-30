import os
import sys
import subprocess

VENV_DIR = 'venv'
activate_script = os.path.join(VENV_DIR, 'Scripts', 'Activate.ps1')

# Check if venv exists, if not, create it
if not os.path.isdir(VENV_DIR):
    print(f"Virtual environment not found. Creating '{VENV_DIR}'...")
    subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
    print(f"Virtual environment '{VENV_DIR}' created.")
else:
    print(f"Virtual environment '{VENV_DIR}' already exists.")

if not os.path.isfile(activate_script):
    print(f"Activation script not found: {activate_script}")
    sys.exit(1)

print("\nTo activate your virtual environment, run this command in PowerShell:")
print(f".\\{VENV_DIR}\\Scripts\\Activate.ps1")
