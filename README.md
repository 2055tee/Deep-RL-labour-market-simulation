# Peer-Simulation

A MESA-based agent-based modeling (ABM) framework for peer simulation.

## Project Structure

- `model.py` - Main MESA model class for the peer simulation
- `agent.py` - Peer agent class with behavior definitions
- `server.py` - Visualization server for interactive simulation
- `run.py` - Script to run simulation without visualization
- `requirements.txt` - Python dependencies
- `setup_venv.sh` - Virtual environment setup script (Linux/Mac)
- `setup_venv.bat` - Virtual environment setup script (Windows)
- `.gitignore` - Python gitignore template

## Getting Started

### Setting Up Virtual Environment

#### On Linux/Mac:
```bash
./setup_venv.sh
```

#### On Windows:
```cmd
setup_venv.bat
```

### Manual Setup

If you prefer to set up manually:

1. Create virtual environment:
```bash
python3 -m venv venv
```

2. Activate virtual environment:
   - Linux/Mac: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate.bat`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation

#### With Visualization:
```bash
python server.py
```
Then open your browser to `http://localhost:8521`

#### Without Visualization:
```bash
python run.py
```

## Requirements

- Python 3.8+
- MESA 2.1.0+
- NumPy 1.24.0+
- Matplotlib 3.7.0+
- Pandas 2.0.0+

## Development

The project uses MESA framework for agent-based modeling. Key components:

- **Model**: Defines the simulation environment, grid, and scheduling
- **Agent**: Defines individual peer behavior and interactions
- **Server**: Provides web-based visualization interface
- **Run**: Batch simulation for data collection

## License

This project is open source and available under standard licensing terms.