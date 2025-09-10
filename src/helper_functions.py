import numpy as np
from scipy.constants import physical_constants

HARTREE_TO_EV = physical_constants['Hartree energy in eV'][0]
BOHR_TO_ANGSTROM = physical_constants['Bohr radius'][0] * 1e10  # Convert meters to Ångstrom
BOHR_CUBED_TO_ANGSTROM_CUBED = BOHR_TO_ANGSTROM**3

import hashlib
import os

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



