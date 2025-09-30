import os
import subprocess
import sys

VENV_DIR = 'venv'

# Check if venv exists, if not, create it
if not os.path.isdir(VENV_DIR):
    print(f"Virtual environment not found. Creating '{VENV_DIR}'...")
    subprocess.check_call([sys.executable, '-m', 'venv', VENV_DIR])
    print(f"Virtual environment '{VENV_DIR}' created.")
else:
    print(f"Virtual environment '{VENV_DIR}' already exists.")

# Build the activation command for PowerShell
activate_script = os.path.join(VENV_DIR, 'Scripts', 'Activate.ps1')
if not os.path.isfile(activate_script):
    print(f"Activation script not found: {activate_script}")
    sys.exit(1)

# Open a new PowerShell window with the venv activated
print("Opening a new PowerShell window with the virtual environment activated...")
subprocess.Popen([
    'powershell',
    '-NoExit',
    '-Command', f'& {{ . "{activate_script}" }}'
])
