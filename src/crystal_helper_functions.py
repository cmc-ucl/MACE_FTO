import numpy as np

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


def parse_crystal_output(file_content, num_atoms):
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


def write_CRYSTAL_gui_from_data(lattice_matrix,atomic_numbers,
                                cart_coords, file_name, dimensionality = 3):

    input_data = [f'{dimensionality} 1 1\n']
    
    identity = np.eye(3).astype('float')

    for row in lattice_matrix:
        input_data.append(' '.join(f'{val:.6f}' for val in row)+'\n')
    input_data.append('1\n')    
    for row in identity:
        input_data.append(' '.join(f'{val:.6f}' for val in row)+'\n')
    input_data.append('0.000000 0.000000 0.000000\n')
    input_data.append(f'{len(cart_coords)}\n')
    for row, row2 in zip(atomic_numbers, cart_coords):
        input_data.append(f'{row} '+' '.join(f'{val:.6f}' for val in row2)+'\n')
    input_data.append('0 0')

    with open(file_name, 'w') as file:
        for line in input_data:
            file.write(f"{line}")



