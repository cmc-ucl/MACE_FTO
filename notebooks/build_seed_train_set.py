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
# # Build the seed train and test sets
#

# %% [markdown]
# - occupation of orbitals in isolated atoms
# - values of the eigenvalues (no 1/3 occupation)
# - PBE + %HF
# - read r2SCAN paper

# %% [markdown]
# #### Families
#
# Initial seed name:
#
# AlGaN_333_K_F_e_md_o
#
# K = composition
# F = family (0-19)
# e = eos (0-11)
# md = MD
# o = optimisation
#
# Folder structure:
#
# supercell_size
#     |
#     -- functional r2SCAN
#         |
#         composition 7 (0,0.1,0.25,0.5,0.75,0.9,1) -> 0,5,14,27,40,49,54 Ga atoms
#             |
#             -- family (configuration) generate 20 - use 5
#                 |
#                 -- initial - this contains the initial structure
#                 -- MD 1000 steps - select ~10
#                 -- EOS (from optimised?) select N structures from the MD ones and do 10 steps 
#                 -- optgeom (starting from MACE optimised and save) all steps - use 1?
#
#
# So we can select the F using random split and then select a subset of structures within that group.

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


# %% [markdown]
# ### AlN and GaN experimental structures

# %%
AlN_exp = Structure.from_file('../data/bulk_structures/experimental/AlN_experimental.cif')

supercell_matrix = np.eye(3)*3

AlN_333_exp = copy.deepcopy(AlN_exp)

AlN_333_exp.make_supercell(supercell_matrix)

AlN_333_exp.num_sites

# %%
GaN_exp = Structure.from_file('../data/bulk_structures/experimental/GaN_experimental.cif')

supercell_matrix = np.eye(3)*3

GaN_333_exp = copy.deepcopy(GaN_exp)

GaN_333_exp.make_supercell(supercell_matrix)

GaN_333_exp.num_sites

# %% [markdown]
# ## Symmetry analysis

# %%
atom_indices_aln_333 = get_all_configurations_pmg(AlN_333_exp)
np.savetxt('../data/symmetry/aln_333_indices.csv',atom_indices_aln_333,delimiter=',',fmt='%d')

# %%
atom_indices_aln = np.genfromtxt('../data/symmetry/aln_333_indices.csv',delimiter=',').astype('int')

# %% [markdown]
# ## Generate SIC random structures
#
# This saves the data into a json file, I'm not sure we need it.

# %%
active_sites=np.where(np.array(AlN_333_exp.atomic_numbers) == 13)[0]
num_active_sites=len(active_sites)

N_atom = 31
num_families = 20
all_config_atom_number = {}
compositions = [0.1,0.25,0.5,0.75,0.9]
for n,comp in enumerate(compositions):
   
    N_Ga = int(np.round(num_active_sites*comp))

    structures_random = generate_random_structures(AlN_333_exp,atom_indices=atom_indices_aln,
                                                   N_atoms=N_Ga,new_species=31,N_config=500,
                                                   DFT_config=num_families,active_sites=active_sites)

    atom_number_tmp = []
    for structure in structures_random:
        atom_number_tmp.append(list(structure.atomic_numbers))

    all_config_atom_number[str(N_Ga)] = atom_number_tmp

with open('../data/seed_structures/333/AlGaN_super3.json', 'w') as json_file:
    json.dump(all_config_atom_number, json_file)

# %%
with open('../data/seed_structures/333/AlGaN_super3.json', 'r', encoding='utf-8') as json_file:
    AlGaN_super3_all_config = json.load(json_file)


# %%
len(AlGaN_super3_all_config['5'])

# %%
# Generate the Extended XYZ files
AlN_lattice_matrix = np.round(AlN_exp.lattice.matrix[0:3], 6)
GaN_lattice_matrix = np.round(GaN_exp.lattice.matrix[0:3], 6)

# %%
# Generate the Extended XYZ files
AlN_lattice_matrix = np.round(AlN_333_exp.lattice.matrix[0:3], 6)
GaN_lattice_matrix = np.round(GaN_333_exp.lattice.matrix[0:3], 6)


positions = AlN_333_exp.frac_coords
for N_atoms in AlGaN_super3_all_config.keys():

    Ga_comp = int(N_atoms)/num_active_sites
    AlGaN_lattice_matrix = (AlN_lattice_matrix*(1-Ga_comp) + GaN_lattice_matrix*Ga_comp)

    folder_name = f'../data/seed_structures/333/r2SCAN/{N_atoms}Ga/initial/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for i,config in enumerate(AlGaN_super3_all_config[N_atoms]):
        structure = Structure(AlGaN_lattice_matrix,config,positions) # here we use the AlN positions

        write_extended_xyz(structure,os.path.join(folder_name,f'AlGaN_333_K{N_atoms}_F{i}_e0_md0_o0.xyz'),
                           comment='initial=True md=False eos=False opt=False')

# %% [markdown]
# ### Concatenate files

# %%
# Ensure the output folder exists
output_file = '../data/seed_structures/333/r2SCAN/concatenated_files/initial/AlGaN_333_KX_FX_e0_md0_o0.xyz'

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, 'w') as outfile:
    for comp in compositions:
        N_Ga = int(np.round(num_active_sites*comp)) 
        for f in range(num_families):
            file_path = f'../data/seed_structures/333/r2SCAN/{N_Ga}Ga/initial/AlGaN_333_K{N_Ga}_F{f}_e0_md0_o0.xyz'
        # for file_name in sorted(os.listdir(input_folder)):
        #     if file_name.endswith(".xyz"):
        #         file_path = os.path.join(input_folder, file_name)

            # Read the content of the current .xyz file
            with open(file_path, 'r') as infile:
                content = infile.read()
            
            # Append content to the output file
            outfile.write(content)


# %% [markdown]
# ## Convergence test - % HF

# %% [markdown]
# ### r2SCAN

# %% [markdown]
# #### AlN

# %%
results = []

for i in np.arange(0,26,5):
    
    with open(f'../data/convergence_tests/hf_exchange/Al_{i}HF.out', "r") as f:
        file_content = f.readlines()
    Al_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/hf_exchange/N_{i}HF.out', "r") as f:
        file_content = f.readlines()
    N_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/hf_exchange/AlN_r2scan_{i}HF.out', "r") as f:
        file_content = f.readlines()

    # Parse the file content
    parsed_structures, opt_end_converged_seen = parse_crystal_output(file_content, num_atoms=4) 
    
    if opt_end_converged_seen == True:
        structure_tmp = parsed_structures[-1]
        a,b,c,alpha,beta,gamma = matrix_to_lattice_params(structure_tmp['lattice_matrix'])
        energy = structure_tmp['energy_ev']
        formation_energy = (energy-2*Al_energy-2*N_energy)/2
        band_gap = structure_tmp['band_gap_ev']

        results.append(
            {
                "%HF": i,
                #"Al_energy": Al_energy,
                #"Ga_energy": Ga_energy,
                # "AlN_energy": energy,
                "formation_energy": formation_energy,
                "band_gap": band_gap,
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            })
        

df = pd.DataFrame(results).set_index("%HF")
df.to_csv("../data/convergence_tests/hf_exchange/AlN_hf_convergence_results.csv")
df


# %% [markdown]
# #### GaN

# %%
results = []

for i in np.arange(0,26,5):
    
    with open(f'../data/convergence_tests/hf_exchange/Ga_{i}HF.out', "r") as f:
        file_content = f.readlines()
    Ga_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/hf_exchange/N_{i}HF.out', "r") as f:
        file_content = f.readlines()
    N_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/hf_exchange/GaN_r2scan_{i}HF.out', "r") as f:
        file_content = f.readlines()

    # Parse the file content
    parsed_structures, opt_end_converged_seen = parse_crystal_output(file_content, num_atoms=4) 
    
    if opt_end_converged_seen == True:
        structure_tmp = parsed_structures[-1]
        a,b,c,alpha,beta,gamma = matrix_to_lattice_params(structure_tmp['lattice_matrix'])
        energy = structure_tmp['energy_ev']
        formation_energy = (energy-2*Ga_energy-2*N_energy)/2
        band_gap = structure_tmp['band_gap_ev']

        results.append(
            {
                "%HF": i,
                #"Al_energy": Al_energy,
                #"Ga_energy": Ga_energy,
                # "AlN_energy": energy,
                "formation_energy": formation_energy,
                "band_gap": band_gap,
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            })
        

df = pd.DataFrame(results).set_index("%HF")
df.to_csv("../data/convergence_tests/hf_exchange/GaN_hf_convergence_results.csv")
df


# %% [markdown]
# #### Standard states

# %% [markdown]
# N2

# %%
with open(f'../data/convergence_tests/hf_exchange/standard_state/N2_15HF.out', "r") as f:
        file_content = f.readlines()
N2_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/hf_exchange/N_15HF.out', "r") as f:
        file_content = f.readlines()
N_energy = read_last_scf_energy(file_content) 
N2_energy-2*N_energy

# %% [markdown]
# Al

# %%
with open(f'../data/convergence_tests/hf_exchange/standard_state/Al_15HF.out', "r") as f:
        file_content = f.readlines()
Al_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/hf_exchange/Al_15HF.out', "r") as f:
        file_content = f.readlines()
Al_energy = read_last_scf_energy(file_content)
Al_metal_energy-Al_energy

# %% [markdown]
# Ga

# %%
with open(f'../data/convergence_tests/hf_exchange/standard_state/Ga_15HF.out', "r") as f:
        file_content = f.readlines()
Ga_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/hf_exchange/Ga_15HF.out', "r") as f:
        file_content = f.readlines()
Ga_energy = read_last_scf_energy(file_content) 
(Ga_metal_energy-4*Ga_energy)/4

# %% [markdown]
# #### Formation energy wrt standard state

# %%
with open(f'../data/convergence_tests/hf_exchange/standard_state/Al_15HF.out', "r") as f:
        file_content = f.readlines()
Al_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/hf_exchange/standard_state/N2_15HF.out', "r") as f:
        file_content = f.readlines()
N2_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/hf_exchange/standard_state/Ga_15HF.out', "r") as f:
        file_content = f.readlines()
Ga_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/hf_exchange/AlN_r2SCAN_15HF.out', "r") as f:
        file_content = f.readlines()
AlN_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/hf_exchange/GaN_r2SCAN_15HF.out', "r") as f:
        file_content = f.readlines()
GaN_energy = read_last_scf_energy(file_content) 
print('AlN energy:',(AlN_energy-2*Al_metal_energy-N2_energy)/2)
print('GaN energy:',(GaN_energy-Ga_metal_energy/2-N2_energy)/2)


# %% [markdown]
# ### PBE

# %% [markdown]
# #### AlN

# %%
results = []

for i in np.arange(0,26,5):
    
    with open(f'../data/convergence_tests/pbe_hf_exchange/Al_{i}HF.out', "r") as f:
        file_content = f.readlines()
    Al_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/pbe_hf_exchange/N_{i}HF.out', "r") as f:
        file_content = f.readlines()
    N_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/pbe_hf_exchange/AlN_r2scan_{i}HF.out', "r") as f:
        file_content = f.readlines()

    # Parse the file content
    parsed_structures, opt_end_converged_seen = parse_crystal_output(file_content, num_atoms=4) 
    
    if opt_end_converged_seen == True:
        structure_tmp = parsed_structures[-1]
        a,b,c,alpha,beta,gamma = matrix_to_lattice_params(structure_tmp['lattice_matrix'])
        energy = structure_tmp['energy_ev']
        formation_energy = (energy-2*Al_energy-2*N_energy)/2
        band_gap = structure_tmp['band_gap_ev']

        results.append(
            {
                "%HF": i,
                #"Al_energy": Al_energy,
                #"Ga_energy": Ga_energy,
                # "AlN_energy": energy,
                "formation_energy": formation_energy,
                "band_gap": band_gap,
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            })
        

df = pd.DataFrame(results).set_index("%HF")
df.to_csv("../data/convergence_tests/pbe_hf_exchange/AlN_hf_convergence_results.csv")
df

# %% [markdown]
# #### GaN

# %%
results = []

for i in np.arange(0,26,5):
    
    with open(f'../data/convergence_tests/pbe_hf_exchange/Ga_{i}HF.out', "r") as f:
        file_content = f.readlines()
    Ga_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/pbe_hf_exchange/N_{i}HF.out', "r") as f:
        file_content = f.readlines()
    N_energy = read_last_scf_energy(file_content) 

    with open(f'../data/convergence_tests/pbe_hf_exchange/GaN_r2scan_{i}HF.out', "r") as f:
        file_content = f.readlines()

    # Parse the file content
    parsed_structures, opt_end_converged_seen = parse_crystal_output(file_content, num_atoms=4) 
    
    if opt_end_converged_seen == True:
        structure_tmp = parsed_structures[-1]
        a,b,c,alpha,beta,gamma = matrix_to_lattice_params(structure_tmp['lattice_matrix'])
        energy = structure_tmp['energy_ev']
        formation_energy = (energy-2*Ga_energy-2*N_energy)/2
        band_gap = structure_tmp['band_gap_ev']

        results.append(
            {
                "%HF": i,
                #"Al_energy": Al_energy,
                #"Ga_energy": Ga_energy,
                # "AlN_energy": energy,
                "formation_energy": formation_energy,
                "band_gap": band_gap,
                "a": a,
                "b": b,
                "c": c,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
            })
        

df = pd.DataFrame(results).set_index("%HF")
df.to_csv("../data/convergence_tests/pbe_hf_exchange/GaN_hf_convergence_results.csv")
df

# %% [markdown]
# #### Standard states

# %% [markdown]
# N2

# %%
with open(f'../data/convergence_tests/pbe_hf_exchange/standard_state/N2_15HF.out', "r") as f:
        file_content = f.readlines()
N2_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/pbe_hf_exchange/N_15HF.out', "r") as f:
        file_content = f.readlines()
N_energy = read_last_scf_energy(file_content) 
N2_energy-2*N_energy

# %% [markdown]
# Al

# %%
with open(f'../data/convergence_tests/pbe_hf_exchange/standard_state/Al_15HF.out', "r") as f:
        file_content = f.readlines()
Al_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/pbe_hf_exchange/Al_15HF.out', "r") as f:
        file_content = f.readlines()
Al_energy = read_last_scf_energy(file_content)
Al_metal_energy-Al_energy

# %% [markdown]
# Ga

# %%
with open(f'../data/convergence_tests/pbe_hf_exchange/standard_state/Ga_15HF.out', "r") as f:
        file_content = f.readlines()
Ga_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/pbe_hf_exchange/Ga_15HF.out', "r") as f:
        file_content = f.readlines()
Ga_energy = read_last_scf_energy(file_content) 
(Ga_metal_energy-4*Ga_energy)/4

# %% [markdown]
# ### Formation energy wrt standard state

# %%
with open(f'../data/convergence_tests/pbe_hf_exchange/standard_state/Al_15HF.out', "r") as f:
        file_content = f.readlines()
Al_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/pbe_hf_exchange/standard_state/N2_15HF.out', "r") as f:
        file_content = f.readlines()
N2_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/pbe_hf_exchange/standard_state/Ga_15HF.out', "r") as f:
        file_content = f.readlines()
Ga_metal_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/pbe_hf_exchange/AlN_r2SCAN_15HF.out', "r") as f:
        file_content = f.readlines()
AlN_energy = read_last_scf_energy(file_content) 
with open(f'../data/convergence_tests/pbe_hf_exchange/GaN_r2SCAN_15HF.out', "r") as f:
        file_content = f.readlines()
GaN_energy = read_last_scf_energy(file_content) 
print('AlN energy:',(AlN_energy-2*Al_metal_energy-N2_energy)/2)
print('GaN energy:',(GaN_energy-Ga_metal_energy/2-N2_energy)/2)


# %% [markdown]
# ### B3LYP

# %%
N_energy = -5.4572189496862E+01*HARTREE_TO_EV
N2_energy = -1.0947321502613E+02*HARTREE_TO_EV
N2_energy-2*N_energy

# %% [markdown]
# ## Test r2SCAN vs r2SCAN0

# %%
random_idx = np.random.choice(np.arange(0, 21), size=12, replace=False)
for i in [5,14,27,40,49]:
    if i == 27:
        for k in random_idx:
            file = f'../data/seed_structures/333/r2SCAN/{i}Ga/initial/AlGaN_333_K{i}_F{k}_e0_md0_o0.xyz'
            sh.copy(file,f'../data/convergence_tests/r2SCAN_vs_r2SCAN0/AlGaN_333_K{i}_F{k}_e0_md0_o0.xyz')
    else:
        k = np.random.randint(0,20)
        file = f'../data/seed_structures/333/r2SCAN/{i}Ga/initial/AlGaN_333_K{i}_F{k}_e0_md0_o0.xyz'
        sh.copy(file,f'../data/convergence_tests/r2SCAN_vs_r2SCAN0/AlGaN_333_K{i}_F{k}_e0_md0_o0.xyz')
    


# %%
path = "../data/convergence_tests/r2SCAN_vs_r2SCAN0"
xyz_files = [f for f in os.listdir(path) if f.endswith(".xyz")]

files = []

for file in xyz_files:
    file_name = os.path.join(path,file)
    atoms = read(file_name)
    structure = AseAtomsAdaptor().get_structure(atoms)

    full_name = file_name[:-4]+'.gui'
    files.append(file[:-4])
    sh.copy('../data/crystal_input_files/sp_r2scan_input.d12',
            full_name[:-4]+'.d12')

    lattice_matrix = structure.lattice.matrix
    atomic_numbers = structure.atomic_numbers
    cart_coords = structure.cart_coords

    write_CRYSTAL_gui_from_data(lattice_matrix,atomic_numbers,
                            cart_coords, full_name, dimensionality = 3)
bash_script = generate_slurm_file(files)

with open("../data/seed_structures/333/r2SCAN/5Ga/initial/slurm_file.slurm", "w") as f:
    for fn in bash_script:
        f.write(fn)

    

# %% [markdown]
# ## Geometry optimisation

# %%
atoms = read('../data/seed_structures/333/r2SCAN/5Ga/initial/AlGaN_333_K5_F0_e0_md0_o0.xyz')
structure = AseAtomsAdaptor().get_structure(atoms)


# %%
files = []
max_family = 4
for comp in compositions:
    N_Ga = int(np.round(num_active_sites*comp)) 
    for f in range(num_families):
        atoms = read(f'../data/seed_structures/333/r2SCAN/{N_Ga}Ga/initial/AlGaN_333_K{N_Ga}_F{f}_e0_md0_o0.xyz')
        structure = AseAtomsAdaptor().get_structure(atoms)

        folder_name = f'../data/seed_structures/333/r2SCAN/{N_Ga}Ga/optgeom/'
        file_name = f'AlGaN_333_K{N_Ga}_F{f}_e0_md0_o1.gui'
        full_name = os.path.join(folder_name,file_name)
        if f <= max_family:
            files.append(full_name)
        sh.copy('../data/crystal_input_files/fulloptg_r2scan_input.d12',
                full_name[:-3]+'d12')
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        lattice_matrix = structure.lattice.matrix
        atomic_numbers = structure.atomic_numbers
        cart_coords = structure.cart_coords

        write_CRYSTAL_gui_from_data(lattice_matrix,atomic_numbers,
                                cart_coords, full_name, dimensionality = 3)
bash_script = generate_slurm_file(files)


# %% [markdown]
# ### Read CRYSTAL output files

# %%
out_xyz = write_extxyz_from_crystal_output(
    "AlGaN_super3_12_7_0.out",
    output_path="AlGaN_super3_12_7_0.xyz",
    num_atoms=108,
    config_type="geometry_optimisation",
    system_name="AlGaN_super3_12_7",
    comment="this=3 that=4"
)

# %%
with open('data/crystal/AlGaN/super3/output_files/AlGaN_super3_1_0_0.out', 'r') as f:
    file_content = f.readlines()

# %% [markdown]
# Example usage

# %%

# num_atoms = 108  
# # Parse the file and extract structures with lattice matrix conversion
# parsed_structures = parse_crystal_output(file_content, num_atoms)

# # Convert to DataFrame for inspection
# df_structures = pd.DataFrame(parsed_structures)

# # Generate extended XYZ files
# generate_extended_xyz_files_from_df(df_structures, 'data/crystal/AlGaN/super3/output_files/test')

# %% [markdown]
# #### Write all the extxyz from all output files

# %%
# Write all the extxyz from all output files
import os
import numpy as np
import pandas as pd

# Folder containing the .out files
# folder_path = "data/crystal/AlGaN/super3/output_files"
# output_folder = "data/crystal/AlGaN/super3/extxyz_files"
folder_path = "data/crystal/AlGaN/pbe0/output_files"
output_folder = "data/crystal/AlGaN/pbe0/extxyz_files"
os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

# Loop over X values
for X in np.arange(54):  # Adjust the range as needed
    for Y in np.arange(10):  # Adjust the range as needed
        # Global index for structures extracted for the current X and Y
        global_index = 0
        Z = 0

        while True:
            # Construct file path for the current Z
            file_name = f"AlGaN_super3_{X}_{Y}_{Z}.out"
            file_path = os.path.join(folder_path, file_name)

            # Check if the file exists
            if not os.path.exists(file_path):
                break  # Exit the loop when no more Z files exist for this X_Y

            # Read the file and process its content
            with open(file_path, "r") as f:
                file_content = f.readlines()

            # Parse the file content
            parsed_structures = parse_crystal_output(file_content, num_atoms=108)  # Replace 108 with your atom count

            # Convert parsed structures to a DataFrame
            df_structures = pd.DataFrame(parsed_structures)

            # Save all extracted structures with unique global indices
            for _, row in df_structures.iterrows():
                # Generate the output file name with the incrementing global index
                output_file = os.path.join(
                    output_folder, f"AlGaN_super3_{X}_{Y}_{global_index}.xyz"
                )

                # Write the structure to an extended XYZ file
                with open(output_file, "w") as out_f:
                    # Write number of atoms
                    num_atoms = len(row['cartesian_coordinates'])
                    out_f.write(f"{num_atoms}\n")

                    # Write metadata
                    lattice_flat = " ".join(f"{value:.12e}" for value in row['lattice_matrix'].flatten())
                    stress_flat = " ".join(f"{value:.12e}" for value in np.array(row['stress']).flatten())
                    out_f.write(
                        f"dft_energy={row['energy_ev']:.12e} "
                        f'Lattice="{lattice_flat}" '
                        f'dft_stress="{stress_flat}" '
                        f'Properties=species:S:1:pos:R:3:dft_forces:R:3 '
                        f'config_type=random '
                        # f'system_name={os.path.basename(output_file[:-4])}\n'
                        f'system_name=random\n'
                    )

                    # Write atomic data
                    for symbol, coord, force in zip(row['atomic_symbols'], row['cartesian_coordinates'], row['forces']):
                        out_f.write(
                            f"{symbol} {coord[0]:.12e} {coord[1]:.12e} {coord[2]:.12e} "
                            f"{force[0]:.12e} {force[1]:.12e} {force[2]:.12e}\n"
                        )

                # Increment the global index
                global_index += 1

            # Increment Z to process the next file
            Z += 1

# %% [markdown]
# Check for dusplicates

# %%
# Folder containing the .out files
folder_path = "data/crystal/AlGaN/pbe0/extxyz_files/"

# Example: Check for duplicates in AlGaN_super3_1_0_*
x = 1
y = 1
pattern_prefix = f"AlGaN_super3_{x}_{y}_"
duplicates = find_duplicate_files(folder_path, pattern_prefix)

if duplicates:
    print("Duplicate files found:")
    for file1, file2 in duplicates:
        print(f"{file1} and {file2}")
else:
    print("No duplicate files found.")

# %% [markdown]
# #### Concatenate files

# %%
# Example usage
input_folder = "data/crystal/AlGaN/pbe0/extxyz_files"
output_file = "data/crystal/AlGaN/pbe0/concatenated_files/AlGaN_super3_all.xyz"
concatenate_xyz_files(input_folder, output_file)

# %% [markdown]
# ### Read structures ASE
#
# The stress is rounded, change to full value from CRYSTAL

# %%
test_file = "data/crystal/AlGaN/super3/concatenated_files/AlGaN_super3_all.xyz"
atoms = read(test_file, index=":")

# %%

# # Directory containing the extxyz files
# directory = 'data/crystal/AlGaN/super3/extxyz_files/'

# # List to store the atoms and stress tensors
# atoms_list = []
# stress_list = []

# # Iterate over all files in the directory
# for filename in os.listdir(directory):
#     if filename.endswith('.xyz'):  # Only process .extxyz files
#         file_path = os.path.join(directory, filename)
        
#         # Read the ASE atoms object
#         atoms = read(file_path, format='extxyz')
#         atoms_list.append(atoms)
        
#         # Extract the stress tensor if it exists
#         stress_flat = atoms.info.get("Stress")
#         if stress_flat is not None:
#             stress = stress_flat.reshape(3, 3)
#             stress_list.append(stress)
#         else:
#             print(f"No stress information found in {filename}")
#             stress_list.append(None)

# %% [markdown]
# #### Test/Train split

# %%
# Convert lists to numpy arrays for easier indexing
atoms_array = np.array(atoms_list, dtype=object)
stress_array = np.array(stress_list, dtype=object)

# Generate random indices for train-test split
n_samples = len(atoms_array)
test_size = 0.2
n_test = int(n_samples * test_size)

# Create a random permutation of indices
indices = np.arange(n_samples)
np.random.shuffle(indices)

# Split indices for train and test sets
test_indices = indices[:n_test]
train_indices = indices[n_test:]

# Split the data
atoms_train = atoms_array[train_indices]
atoms_test = atoms_array[test_indices]
stress_train = stress_array[train_indices]
stress_test = stress_array[test_indices]

# Output information
print(f"Total structures: {n_samples}")
print(f"Training set: {len(atoms_train)} structures")
print(f"Testing set: {len(atoms_test)} structures")
