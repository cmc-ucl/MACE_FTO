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
# #### Families
#
# Initial seed name:
#
# AlGaN_333_K_F_e_r
#
# K = composition
# F = family
# e = expansion
# r = randomised displacements
#
# Folder structure:
#
# supercell_size
#     |
#     -- functional
#         |
#         -- family
#             |
#             -- optgeom
#             -- expansion
#             -- random dispalcements
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
# ### AlGaN - OPTIMISE STRUCTURES WITH CRYSTAL23

# %%
AlN_bulk_r2scan = Structure.from_file('../data/bulk_structures/AlN.cif')

supercell_matrix = np.eye(3)*3

AlN_333_r2scan = copy.deepcopy(AlN_bulk_r2scan)

AlN_333_r2scan.make_supercell(supercell_matrix)

AlN_333_r2scan.num_sites

# %%
# write_CRYSTAL_gui_from_data(AlN_bulk_crystal.lattice.matrix,AlN_bulk_crystal.atomic_numbers,AlN_bulk_crystal.cart_coords,'../data/bulk_structures/crystal/AlN.gui')

# %%
GaN_bulk_r2scan = Structure.from_file('../data/bulk_structures/GaN.cif')

supercell_matrix = np.eye(3)*3

GaN_333_r2scan = copy.deepcopy(GaN_bulk_r2scan)

GaN_333_r2scan.make_supercell(supercell_matrix)

GaN_333_r2scan.num_sites

# %% [markdown]
# ## Symmetry analysis

# %%
atom_indices_aln_333 = get_all_configurations_pmg(AlN_333_r2scan)
np.savetxt('../data/symmetry/aln_333_indices.csv',atom_indices_aln_333,delimiter=',',fmt='%d')

# %%
atom_indices_aln = np.genfromtxt('../data/symmetry/aln_333_indices.csv',delimiter=',').astype('int')

# %% [markdown]
# ## Generate SIC random structures

# %%
active_sites=np.where(np.array(AlN_333_r2scan.atomic_numbers) == 13)[0]
num_active_sites=len(active_sites)

N_atom = 31

all_config_atom_number = {}

for n,N_atoms in enumerate(np.arange(27,28)):

    structures_random = generate_random_structures(AlN_333_r2scan,atom_indices=atom_indices_aln,
                                                   N_atoms=N_atoms,new_species=31,N_config=500,
                                                   DFT_config=20,active_sites=active_sites)

    atom_number_tmp = []
    for structure in structures_random:
        atom_number_tmp.append(list(structure.atomic_numbers))

    all_config_atom_number[str(N_atoms)] = atom_number_tmp

# with open('data/supercell_structures/AlGaN/AlGaN_super3.json', 'w') as json_file:
#     json.dump(all_config_atom_number, json_file)

# %%
all_config_atom_number

# %%
vview(structures_random[0])

# %%
with open('data/supercell_structures/AlGaN/AlGaN_super3.json', 'r', encoding='utf-8') as json_file:
    AlGaN_super3_all_config = json.load(json_file)


# %%
# Generate the Extended XYZ files

lattice = AlN_super3.lattice.matrix
positions = AlN_super3.frac_coords
for N_atoms in AlGaN_super3_all_config.keys():
    
    folder_name = f'data/supercell_structures/AlGaN/AlGaN_super3_{N_atoms}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    for i,config in enumerate(AlGaN_super3_all_config[N_atoms]):
        structure = Structure(lattice,config,positions)

        write_extended_xyz(structure,os.path.join(folder_name,f'AlGaN_super3_{N_atoms}_{i}.xyz'))


# %% [markdown]
# ### Write CRYSTAL input files

# %%
def generate_slurm_file(file_names_list, project_code='e05-algor-smw'):

    bash_script = [
    '#!/bin/bash\n',
    f'#SBATCH --nodes={len(file_names_list)}\n',
    '#SBATCH --ntasks-per-node=128\n',
    '#SBATCH --cpus-per-task=1\n',
    '#SBATCH --time=24:00:00\n\n',
    '# Replace [budget code] below with your full project code\n',
    f'#SBATCH --account={project_code}\n',
    '#SBATCH --partition=standard\n',
    '#SBATCH --qos=standard\n',
    '#SBATCH --export=none\n\n',
    'module load epcc-job-env\n',
    'module load other-software\n',
    'module load crystal\n\n',
    '# Address the memory leak\n',
    'export FI_MR_CACHE_MAX_COUNT=0\n',
    'export SLURM_CPU_FREQ_REQ=2250000\n\n',
    '# Run calculations\n'
]

    for file in file_names_list:
        bash_script.append(f'timeout 1430m /work/e05/e05/bcamino/runCRYSTAL/Pcry_slurm_multi {file[:-4]} &\n')

    bash_script.append('wait')

    return bash_script
    



# %%
AlN_lattice_matrix = np.round(AlN_super3.lattice.matrix[0:3], 6)
GaN_lattice_matrix = np.round(GaN_super3.lattice.matrix[0:3], 6)

AlGaN_lattice_matrix = (AlN_lattice_matrix + GaN_lattice_matrix)/2

# %%
from structure_generation import write_CRYSTAL_gui_from_data


lattice_matrix = AlGaN_lattice_matrix
cart_coords = np.round(AlN_super3.cart_coords,8)


for N_atoms in AlGaN_super3_all_config.keys():
    
    for i,config in enumerate(AlGaN_super3_all_config[N_atoms]):

        atomic_numbers = config

        folder_name = f'data/crystal/AlGaN/super3/config_{i}/'
        file_name = f'AlGaN_super3_{N_atoms}_{i}_0.gui'
        full_name = os.path.join(folder_name,file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        for i,config in enumerate(AlGaN_super3_all_config[N_atoms]):
            structure = Structure(lattice_matrix,config,cart_coords)

            write_CRYSTAL_gui_from_data(lattice_matrix,atomic_numbers,
                                cart_coords, full_name, dimensionality = 3)


# %%
folder_path = 'data/crystal/AlGaN/super3/'

folders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

for folder in folders:

    folder_path_new = os.path.join(folder_path,folder)
    slurm_file_name = os.path.join(folder_path_new,f'{folder}_0.slurm')
    files = [name for name in os.listdir(folder_path_new) 
         if os.path.isfile(os.path.join(folder_path_new, name)) and name.endswith('.gui')]

    # copy .d12
    for file in files:
        input_file = os.path.join(folder_path_new,f'{file[:-4]}.d12')
        sh.copy('data/crystal/AlGaN/super3/super3_input.d12', input_file)

    bash_script = generate_slurm_file(files)
    with open(slurm_file_name, 'w') as file:
        for line in bash_script:
            file.write(f"{line}")



# %%

# %% [markdown]
# ### Read CRYSTAL output files

# %%
with open('data/crystal/AlGaN/super3/output_files/AlGaN_super3_1_0_0.out', 'r') as f:
    file_content = f.readlines()


# %%
def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """
    Convert lattice parameters to a 3x3 lattice matrix.

    Parameters:
        a, b, c (float): Lattice constants.
        alpha, beta, gamma (float): Angles (in degrees) between the lattice vectors.

    Returns:
        numpy.ndarray: 3x3 lattice matrix.
    """
    # Convert angles from degrees to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)

    # Compute the lattice vectors
    v_x = a
    v_y = b * np.cos(gamma_rad)
    v_z = c * np.cos(beta_rad)

    w_y = b * np.sin(gamma_rad)
    w_z = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)

    u_z = np.sqrt(c**2 - v_z**2 - w_z**2)

    # Assemble the lattice matrix
    lattice_matrix = np.array([
        [v_x, 0, 0],
        [v_y, w_y, 0],
        [v_z, w_z, u_z],
    ])

    return lattice_matrix


# %%
# Conversion factors
HARTREE_TO_EV = physical_constants['Hartree energy in eV'][0]
BOHR_TO_ANGSTROM = physical_constants['Bohr radius'][0] * 1e10  # Convert meters to Ångstrom
BOHR_CUBED_TO_ANGSTROM_CUBED = BOHR_TO_ANGSTROM**3

def parse_extended_xyz(file_content, num_atoms):
    """
    Parse the file to extract structures and convert lattice parameters, coordinates, energy, forces,
    and stress tensor to standard units (e.g., eV, Å).
    """
    def extract_floats(line):
        """Helper function to extract floats from a string."""
        return list(map(float, re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", line)))

    results = []
    structure_data = {}

    for i, line in enumerate(file_content):
        line = line.strip()

        # Lattice parameters
        if "ATOM                 X/A                 Y/B                 Z/C" in line:
            lattice_params = extract_floats(file_content[i - 3])
            if len(lattice_params) == 6:
                a, b, c, alpha, beta, gamma = lattice_params
                structure_data['lattice_matrix'] = lattice_params_to_matrix(a, b, c, alpha, beta, gamma)

        # Fractional coordinates and atomic symbols
        if "ATOM                 X/A                 Y/B                 Z/C" in line:
            start = i + 2
            fractional_coords = []
            atomic_symbols = []
            for j in range(num_atoms):
                coord_line = file_content[start + j].strip()
                parts = coord_line.split()
                atomic_number = int(parts[2])  # Third element is the atomic number
                atomic_symbols.append(chemical_symbols[atomic_number])  # Convert to symbol
                fractional_coords.append(extract_floats(coord_line))

            structure_data['fractional_coordinates'] = fractional_coords
            structure_data['atomic_symbols'] = atomic_symbols

            # Calculate Cartesian coordinates
            lattice_matrix = structure_data['lattice_matrix']
            structure_data['cartesian_coordinates'] = [
                np.dot(coord, lattice_matrix) for coord in fractional_coords
            ]

        # Energy
        if "== SCF ENDED - CONVERGENCE ON ENERGY      E(AU)" in line:
            energy_hartree = extract_floats(line)[0]
            structure_data['energy_ev'] = energy_hartree * HARTREE_TO_EV

        # Forces
        if "CARTESIAN FORCES IN HARTREE/BOHR (ANALYTICAL)" in line:
            start = i + 2
            structure_data['forces'] = [
                extract_floats(file_content[start + j])
                for j in range(num_atoms)
            ]

        # Stress tensor
        if "STRESS TENSOR, IN HARTREE/BOHR^3:" in line:
            start = i + 4
            stress_hartree_bohr3 = [
                extract_floats(file_content[start + j]) for j in range(3)
            ]
            stress_ev_angstrom3 = np.array(stress_hartree_bohr3) * (HARTREE_TO_EV / BOHR_CUBED_TO_ANGSTROM_CUBED)
            structure_data['stress'] = stress_ev_angstrom3.tolist()

        # Store the structure if all required fields are found
        if all(key in structure_data for key in ['lattice_matrix', 'fractional_coordinates', 'cartesian_coordinates', 'energy_ev', 'forces', 'stress', 'atomic_symbols']):
            results.append(structure_data.copy())
            structure_data = {}  # Reset for the next structure

    return results


# %%
def generate_extended_xyz_files_from_df(df, seed_name, start_index):
    """
    Generate extended XYZ files from a DataFrame containing structure data in ASE extended format.

    Parameters:
        df (pd.DataFrame): DataFrame containing columns:
            - 'cartesian_coordinates': List of Cartesian coordinates.
            - 'atomic_symbols': List of atomic symbols.
            - 'energy_ev': Energy in eV.
            - 'forces': List of forces for each atom.
            - 'stress': Stress tensor for the structure.
            - 'lattice_matrix': Lattice matrix.
        seed_name (str): Base name for output files.
        start_index (int): Starting index for numbering the output files.

    Returns:
        int: Updated index after processing all structures.
    """
    index = start_index
    for _, row in df.iterrows():
        # Filename with incrementing index
        filename = f"{seed_name}_{index}.xyz"

        # Extract data
        cartesian_coords = row['cartesian_coordinates']
        atomic_symbols = row['atomic_symbols']
        energy = row['energy_ev']
        forces = row['forces']
        stress = row['stress']
        lattice_matrix = row['lattice_matrix']
        num_atoms = len(cartesian_coords)

        # Generate content
        content = []
        # First line: Number of atoms
        content.append(str(num_atoms))
        # Second line: Metadata (energy, lattice matrix, stress tensor, properties, config type)
        lattice_flat = " ".join(f"{value:.12e}" for row in lattice_matrix for value in row)
        stress_flat = " ".join(f"{value:.12e}" for row in stress for value in row)
        content.append(
            f'Energy={energy:.12e} '
            f'Lattice="{lattice_flat}" '
            f'Stress="{stress_flat}" '
            f'Properties=species:S:1:pos:R:3:forces:R:3 '
            f'Config_type={filename}'
        )
        # Atom lines with forces
        for atomic_symbol, atom, force in zip(atomic_symbols, cartesian_coords, forces):
            content.append(
                f"{atomic_symbol} {atom[0]:.12e} {atom[1]:.12e} {atom[2]:.12e} "
                f"{force[0]:.12e} {force[1]:.12e} {force[2]:.12e}"
            )

        # Write to file
        with open(filename, 'w') as file:
            file.write("\n".join(content) + "\n")

        # Increment index
        index += 1

    return index


# %%

# num_atoms = 108  
# # Parse the file and extract structures with lattice matrix conversion
# parsed_structures = parse_extended_xyz(file_content, num_atoms)

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
            parsed_structures = parse_extended_xyz(file_content, num_atoms=108)  # Replace 108 with your atom count

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
import os
import hashlib

def calculate_file_hash(file_path, hash_algo="md5"):
    """
    Calculate the hash of a file using the specified algorithm.
    
    Parameters:
        file_path (str): Path to the file.
        hash_algo (str): Hash algorithm to use (default: "md5").
    
    Returns:
        str: Hexadecimal hash of the file content.
    """
    hash_func = hashlib.new(hash_algo)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def find_duplicate_files(folder_path, pattern_prefix):
    """
    Find duplicate files in a folder for a given pattern prefix.
    
    Parameters:
        folder_path (str): Path to the folder containing files.
        pattern_prefix (str): Prefix pattern for filtering files (e.g., "AlGaN_super3_X_Y_").
    
    Returns:
        list of tuple: List of duplicate file pairs (file1, file2).
    """
    # Filter files matching the pattern
    files = [f for f in os.listdir(folder_path) if f.startswith(pattern_prefix)]
    file_hashes = {}
    duplicates = []

    # Calculate hashes for each file
    for file in files:
        file_path = os.path.join(folder_path, file)
        file_hash = calculate_file_hash(file_path)
        if file_hash in file_hashes:
            # Found a duplicate
            duplicates.append((file_hashes[file_hash], file))
        else:
            file_hashes[file_hash] = file

    return duplicates

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
import os

def concatenate_xyz_files(input_folder, output_file):
    """
    Concatenate all .xyz files in a folder into a single .xyz file.

    Parameters:
        input_folder (str): Path to the folder containing .xyz files.
        output_file (str): Path to the output .xyz file.
    """
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as outfile:
        for file_name in sorted(os.listdir(input_folder)):
            if file_name.endswith(".xyz"):
                file_path = os.path.join(input_folder, file_name)

                # Read the content of the current .xyz file
                with open(file_path, 'r') as infile:
                    content = infile.read()
                
                # Append content to the output file
                outfile.write(content)

    print(f"All .xyz files in '{input_folder}' have been concatenated into '{output_file}'.")

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


# %% [markdown]
# ## mace geometry optimisation

# %%
def mace_geom_opt(atoms):

    atoms_sp = SinglePoint(
        struct=atoms.copy(),
        arch="mace_mp",
        device='cpu',
        calc_kwargs={'model_paths':'small','default_dtype':'float64'},
    )

    atoms_opt = GeomOpt(
        struct=atoms_sp.struct,
        fmax=0.001,
    )

    atoms_opt.run()

    return atoms_opt


# %%
np.round(sAlN_super3_mace_opt.struct.positions[0:],6)


# %%
mgo = Structure.from_file('data/test/MgO_mp-1265_computed.cif')
from pymatgen.io.xyz import XYZ
XYZ(mgo).write_file('data/test/mgo.xyz')

# %%
mgo.lattice.matrix
