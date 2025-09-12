# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: kernelspec,language_info
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: mace_fto
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.11.13
# ---

# %% [markdown]
# # TMP notebook for all things MACE and janus

# %% [markdown]
# ### To make descriptors
#
# janus descriptors --config descriptors.yml --struct file_name.xyz
#
# ### Visualise distribution of descriptors
#
# distributions_dft.py
#
# ### Custom FPS
#
# python fps_from_exyz.py file_name.xyz -k n_final_struct
#
# ### Alin FPS
#
# python script/sample_split.py --n_samples 2000 --config_types random --pre AlGaN AlGaN_super3_all-descriptors.extxyz

# %%
# %load_ext autoreload
# %reload_ext autoreload
# %autoreload 2

from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import read
from ase.data import chemical_symbols
import copy
import numpy as np
import pandas as pd
from scipy.constants import physical_constants
HARTREE_TO_EV = physical_constants['Hartree energy in eV'][0]
BOHR_TO_ANGSTROM = physical_constants['Bohr radius'][0] * 1e10  # Convert meters to Ångstrom
BOHR_CUBED_TO_ANGSTROM_CUBED = BOHR_TO_ANGSTROM**3

import os
import json
import re
import shutil as sh

from janus_core.calculations.single_point import SinglePoint
from janus_core.calculations.geom_opt import GeomOpt
import sys
sys.path.append('../src')   # add src/ to Python path
from functions import *
from structure_generation import *
from helper_functions import *
from crystal_helper_functions import *

current_dir = os.path.dirname(os.path.abspath("__file__"))
sys.path.append(current_dir)

def vview(structure):
    from ase.visualize import view
    from pymatgen.io.ase import AseAtomsAdaptor
    
    view(AseAtomsAdaptor().get_atoms(structure))

# %%

# %%

# %%
