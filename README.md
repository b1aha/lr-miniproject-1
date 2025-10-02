# MP1: Quadruped Jumping

This repository contains code to implement and optimize a quadruped jumping controller in simulation.
The controller is inspired from the Quadruped-Frog paper [1].
You will **only modify a few specific files**, the  rest provide the environment and helper code.

## Quick start (TL;DR)

```bash
# 1. Create environment (with conda, recommended)
conda create -n quadruped python=3.9
conda activate quadruped

# 2. Install dependencies
pip install -r requirements.txt
# If pybullet fails, install through conda before installing requirements again:
conda install conda-forge::pybullet

# 3. Run your first controller script
python quadruped_jump.py
```

## Repository structure

```bash
.
├── env/                     # Library code (DO NOT MODIFY)
│   ├── utils.py             # Filepath utilities
│   └── simulation.py        # Quadruped simulation class (read to see available functions)
├── quadruped_jump.py        # Main script: implement your jumping controller here
├── quadruped_jump_opt.py    # Optimization script: optimize your controller with Optuna
├── profiles.py              # Define force profiles for your controller
└── requirements.txt         # Dependencies
```

## Installation

Installation follows the same structure as the [practicals](https://gitlab.epfl.ch/lgevers/lr-practicals).


### 1. Create a virtual environment

We recommend using **conda** (preferred) or **virtualenv** with **Python 3.9** or higher.
It is a good practice to keep separate environments for different projects, so we encourage you to create a new environment for this project rather than reusing the one for the practicals (especially since the dependencies are not exactly the same).

After downloading the repository, create the virtual environment as follows:

```bash
# With conda (recommended)
conda create -n quadruped python=3.9

# With virtualenv (alternative)
python -m venv venv
```

Then, activate your environment every time you intend on using it for this project:

```bash
# With conda (recommended)
conda activate quadruped

# With virtualenv (alternative)
# Note that you need to be in the project directory containing the venv/ folder
source venv/bin/activate    # Linux/Mac OS
venv\Scripts\Activate       # Windows
```

### 2. Install dependencies


```bash
pip install -r requirements.txt
```

⚠️ If pybullet fails to install (compilation issues), install through conda and then install the `requirements.txt` again:

```bash
conda install conda-forge::pybullet
```

## What you need to do

**Do not modify** `env/`: this is library code.
It is a good idea to check out the functions of the `QuadSimulator` in `env/simulation.py` to check what states you can retrieve from the simulation.

**Modify the following files:**
- `quadruped_jump.py`: implement your quadruped jumping controller by completing the `# TODO`s in the file. 
Run your implementation with:
```bash
python quadruped_jump.py
```
- `profiles.py`: implement the equations to generate the force profiles for your controller. 
Work on this in parallel with `quadruped_jump.py`.
- `quadruped_jump_opt.py`: after finishing your controller, extend it with an optimization framework. 
Here we use Optuna [2], you can check out the associated [documentation](https://optuna.readthedocs.io/en/stable/) if needed.
The skeleton of the optimization framework is already prepared for you, you only need to implement the `# TODO`s:
    - The variables to optimize (and their bounds)
    - The objective function
    - run the script with:
```bash
python quadruped_jump_opt.py
```

Creating new files and scripts (e.g., for plotting) is allowed as long as the structure is clear and concise (please document your changes if you do so).
Do not forget to bundle these potential new files with the final code you are submitting.

## Submission

When submitting zip all your code files together.
These should be your **code files only** (e.g., no `venv` directory for the virtual environment).
You should include the `env` folder and the `requirements.txt` in case you are adding new dependencies.
Videos should be renamed to more descriptive names (e.g., `forward_jump.MP4`) that you refer to in your report.
The structure of your submission folder should look like this:

```bash
lr_mp1_group_{group number}.zip
├── env/                     # Unmodified
│   ├── __init__.py          
│   ├── utils.py             
│   └── simulation.py        
├── videos/                  # Only include videos that are relevant to your report!
│   ├── forward_jump.MP4     # Descriptive name example        
│   └── ...        
├── quadruped_jump.py        # Script with your changes
├── quadruped_jump_opt.py    # Script with your changes
├── profiles.py              # Script with your changes
└── requirements.txt         # Dependencies
```

**Note that the max submission size is 100 MB, which means you may have to compress your videos.**

## References

```
[1] G. Bellegarda, M. Shafiee, M. E. Özberk and A. Ijspeert, "Quadruped-Frog: Rapid Online Optimization of Continuous Quadruped Jumping," 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan, 2024, pp. 1443-1450.

[2] Akiba, T., et al, "Optuna: A Next-generation Hyperparameter Optimization Framework," in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2623–2631.
```
