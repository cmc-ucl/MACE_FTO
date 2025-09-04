# MACE_FTO
MACE fine tuning and optimisation tools

## Installation
conda create -n mace_fto python=3.11
conda activate mace_fto

Then either:
python -m pip uninstall mace-torch
python -m pip install -U torch torchvision torchaudio
python -m pip install git+https://github.com/ACEsuit/mace.git
python -m pip install uv  
uv pip install 'janus-core[mace]@git+https://github.com/stfc/janus-core.git' 

python -m pip install pymatgen
python -m pip install jupyter

or

pip install -r requirements.txt

## Jupyter notebook collaboration
jupytext --set-formats ipynb,py:percent path/to/notebooks/*.ipynb

vi .jupytext.toml

PASTE:
default_jupytext_formats = "ipynb,py:percent"
notebook_metadata_filter = "kernelspec,language_info"
cell_metadata_filter = "-all"

vi .pre-commit-config.yaml

PASTE:
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.7.1
    hooks:
      - id: nbstripout
  - repo: https://github.com/mwouts/jupytext
    rev: v1.16.2
    hooks:
      - id: jupytext
        args: ["--sync"]  # keeps .ipynb and .py paired on commit

vi .gitattributes

PASTE:
*.ipynb  merge=jupyternb
nbdime config-git --enable

### For collaborators only:
pre-commit install
nbdime config-git --enable

### Conflict
When there is a conflict work on the .py file and then opening the .ipynb should have the correct version, otherwise:

run: jupytext --sync EDA.ipynb (ensures both files match).

### When creating a new jn
It should be automatic with .jupytext.toml
	1.	Create a new MyNotebook.ipynb.
	2.	Open it.
	3.	Open Command Palette (Ctrl+Shift+P or Cmd+Shift+P on Mac).
	4.	Search for “Jupytext: Pair Notebook with percent Script”.
	5.	A file MyNotebook.py appears next to it.

### Committing
There is an error message when you commit the nb. Just click ok and then restage the notebooks and this second time it works. This is because the first time you try to committ is changes the nbs, so it looks like there are unstaged changes.