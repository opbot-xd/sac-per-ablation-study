# SAC vs SAC-PER Ablation Study

A rigorous ablation study of Soft Actor-Critic (SAC) and Prioritized Experience Replay (PER), evaluated natively on the `gym-anm` suite (`ANM6Easy-v0`).

This codebase tests whether the improvements in SAC-PER are due to PER or co-bundled fixes like N-step returns, clipping, and rewards normalization. 

## Features
- **True Parallel Training**: Scale experiments perfectly across multiple seeds and variants with multi-core parallel processing. 
- Designed explicitly to unleash 100% capacity on high-end hardware (e.g., Core i9 processors, 24 threads, 100GB RAM).
- Includes an exploratory Jupyter Notebook equipped with rich multi-seed plotting/CI graphing tools.

## Repository Structure 
- `SAC_SACPER_Ablation_Study.ipynb`: Main interactive notebook detailing the architecture and final visualizations.
- `beast_script.py and laptop_script.py`: The exported pure-Python multithreading script designed for zero-bottleneck terminal execution.
- `requirements.txt`: Necessary dependencies (`torch`, `gymnasium`, etc.).

## Setup

1. Clone the repository natively on your beast machine.
2. Initialize a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Experiments

To run all 24 configurations simultaneously (8 experiments × 3 seeds):

```bash
python beast_script.py and laptop_script.py
```

This will run silently and natively lock Python to 1 PyTorch thread per worker so the CPU handles the 24 parallel streams effortlessly. All completed runs will automatically deposit `.pkl` log data into the `results/` directory.

## Graphing

Once `beast_script.py and laptop_script.py` finishes your full simulation sweeps:
1. Open `SAC_SACPER_Ablation_Study.ipynb` locally.
2. Skip the cell that runs the agent experiments. 
3. Execute the final plotting cells at the bottom to aggregate everything directly from your loaded `results/` folder and generate the graphs.
