import numpy as np
import re
from helper_functions import *
from ase.data import chemical_symbols
from typing import Tuple, Optional

HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANGSTROM = 0.529177210903
FORCE_HBOHR_TO_EVANG = HARTREE_TO_EV / BOHR_TO_ANGSTROM        # Hartree/Bohr -> eV/Å
BOHR_CUBED_TO_ANGSTROM_CUBED = BOHR_TO_ANGSTROM ** 3
STRESS_HBOHR3_TO_EVANG3 = HARTREE_TO_EV / BOHR_CUBED_TO_ANGSTROM_CUBED  # Hartree/Bohr^3 -> eV/

def check_converged(content: str) -> bool:
    """
    Returns True if 'OPT END - CONVERGED' is in the file content string,
    otherwise returns False.
    """
    return "OPT END - CONVERGED" in content

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
    Parse the CRYSTAL .out content and return a list[dict] of structures.
    Each dict contains:
      - lattice_matrix: (3,3) in Å
      - fractional_coordinates: (N,3)
      - cartesian_coordinates: (N,3) in Å
      - atomic_symbols: list[str]
      - energy_ev: float
      - forces: (N,3) in eV/Å
      - stress: (3,3) in eV/Å^3
      - band_gap_ev: float (last gap seen in that SCF; 0.0 if none)
    Note: OPT convergence is *not* per structure—handled later for the last frame only.
    """

    def extract_floats(line):
        return list(map(float, re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?", line)))

    results = []
    structure_data = {}
    # Track the last band gap *since previous SCF end*; reset after each SCF ends.
    last_gap_since_prev_scf = None
    # Track OPT convergence anywhere in file (applies to last structure only)
    opt_end_converged_seen = False

    # Optional: patterns
    pat_coords_header = "ATOM                 X/A                 Y/B                 Z/C"
    pat_scf_end = "== SCF ENDED - CONVERGENCE ON ENERGY"
    pat_forces = "CARTESIAN FORCES IN HARTREE/BOHR (ANALYTICAL)"
    pat_stress = "STRESS TENSOR, IN HARTREE/BOHR^3:"
    pat_bandgap_dir = "DIRECT ENERGY BAND GAP:"
    pat_bandgap_ind = "INDIRECT ENERGY BAND GAP:"
    pat_opt_end = "OPT END - CONVERGED"

    i = 0
    nlines = len(file_content)
    while i < nlines:
        line = file_content[i].strip()

        # OPT convergence flag (for the overall optimization)
        if pat_opt_end in line:
            opt_end_converged_seen = True

        # Lattice parameters appear a few lines above coords header
        if pat_coords_header in line:
            # lattice params expected 3 lines above, formatted: a b c alpha beta gamma
            lattice_params = extract_floats(file_content[i - 3]) if i >= 3 else []
            if len(lattice_params) == 6:
                a, b, c, alpha, beta, gamma = lattice_params
                structure_data['lattice_matrix'] = lattice_params_to_matrix(a, b, c, alpha, beta, gamma)

            # Fractional coordinates and symbols: next num_atoms lines (skip 2 header lines)
            start = i + 2
            fractional_coords = []
            atomic_symbols = []
            for j in range(num_atoms):
                coord_line = file_content[start + j].strip()
                parts = coord_line.split()
                atomic_number = int(parts[2])  # 3rd column is atomic number
                atomic_symbols.append(chemical_symbols[atomic_number])
                fractional_coords.append(extract_floats(coord_line))

            structure_data['fractional_coordinates'] = fractional_coords
            structure_data['atomic_symbols'] = atomic_symbols

            # Compute cartesian
            if 'lattice_matrix' in structure_data:
                lattice_matrix = np.array(structure_data['lattice_matrix'])
                cart = np.dot(np.array(fractional_coords), lattice_matrix)
                structure_data['cartesian_coordinates'] = cart.tolist()

            i += num_atoms  # skip past the block we just read
            continue

        # Energy at SCF end (in Hartree)
        if pat_scf_end in line:
            floats = extract_floats(line)
            if floats:
                energy_hartree = floats[0]
                structure_data['energy_ev'] = energy_hartree * HARTREE_TO_EV
            # attach the last gap seen during this SCF (or 0.0)
            structure_data['band_gap_ev'] = float(last_gap_since_prev_scf) if last_gap_since_prev_scf is not None else 0.0
            # reset gap tracker for the next SCF
            last_gap_since_prev_scf = None

        # Forces block (Hartree/Bohr)
        if pat_forces in line:
            start = i + 2
            forces_hb = [extract_floats(file_content[start + j]) for j in range(num_atoms)]
            forces = (np.array(forces_hb) * FORCE_HBOHR_TO_EVANG).tolist()
            structure_data['forces'] = forces

        # Stress tensor (Hartree/Bohr^3) — 3 lines, 3 columns
        if pat_stress in line:
            start = i + 4
            stress_hb3 = [extract_floats(file_content[start + j]) for j in range(3)]
            stress = (np.array(stress_hb3) * STRESS_HBOHR3_TO_EVANG3).tolist()
            structure_data['stress'] = stress

        # Band gap lines (capture the *latest* seen value; used at SCF end)
        if (pat_bandgap_dir in line) or (pat_bandgap_ind in line):
            vals = extract_floats(line)
            if vals:
                last_gap_since_prev_scf = vals[0]  # in eV already

        # If we have all required pieces for one structure, store it.
        required = ['lattice_matrix', 'fractional_coordinates', 'cartesian_coordinates',
                    'energy_ev', 'forces', 'stress', 'atomic_symbols', 'band_gap_ev']
        if all(k in structure_data for k in required):
            results.append(structure_data.copy())
            structure_data.clear()

        i += 1

    # Attach OPT convergence info so the writer can mark the last frame.
    # We store it in a small side-channel: return a tuple (results, opt_flag)
    # If you prefer returning only results, you can also stash it in results[-1]['opt_converged']
    return results, opt_end_converged_seen

_SCFE_LINE = re.compile(
    r"SCF ENDED - CONVERGENCE ON ENERGY\s+E\(AU\)\s+([-+]?\.?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)"
)

def read_last_scf_energy(lines: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Return the last SCF total energy from a CRYSTAL .out file.
    Output: (energy_hartree, energy_eV). If not found, returns (None, None).
    """
    energy_ha = None

    # Read and scan from bottom to top to find the last occurrence quickly

    for line in reversed(lines):
        m = _SCFE_LINE.search(line)
        if m:
            try:
                energy_ha = float(m.group(1))
            except ValueError:
                energy_ha = None
            break

    if energy_ha is None:
        return None, None

    return energy_ha * HARTREE_TO_EV

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


def write_extxyz_from_CRYSTAL_output(
    input_path: str,
    output_path: str | None = None,
    num_atoms: int = 108,
    config_type: str = "random",
    system_name: str | None = None,
    comment: str | None = None,
):
    """
    Parse a single CRYSTAL output and write a multi-frame extended XYZ (.xyz)
    containing energies, forces, stresses, and lattice for each parsed structure.

    Parameters
    ----------
    input_path : str
        Path to the CRYSTAL .out file.
    output_path : str | None
        Where to write the .xyz. If None, uses input basename with .xyz.
    num_atoms : int
        Number of atoms expected in each structure.
    config_type : str
        Value for the EXYZ 'config_type' tag in the comment line.
    system_name : str | None
        Value for the EXYZ 'system_name' tag. If None, uses input stem.
    comment : str | None
        Extra text to append to the EXYZ comment line (second line).
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = f"{stem}.xyz"

    if system_name is None:
        system_name = os.path.splitext(os.path.basename(output_path))[0]

    # Read & parse
    with open(input_path, "r") as f:
        file_content = f.readlines()

    structures = parse_crystal_output(file_content, num_atoms=num_atoms)
    if not structures:
        raise ValueError(f"No structures parsed from: {input_path}")

    with open(output_path, "w") as out_f:
        for row in structures:
            N = len(row["cartesian_coordinates"])
            if N != num_atoms:
                raise ValueError(f"Parsed atom count {N} != expected {num_atoms}")

            out_f.write(f"{N}\n")

            # Flatten lattice and stress
            lattice_flat = " ".join(f"{v:.12e}" for v in np.array(row["lattice_matrix"]).flatten())
            stress_arr = np.array(row["stress"])
            stress_flat = " ".join(f"{v:.12e}" for v in stress_arr.flatten())

            # Build metadata line
            metadata = (
                f"dft_energy={row['energy_ev']:.12e} "
                f'Lattice="{lattice_flat}" '
                f'dft_stress="{stress_flat}" '
                f'Properties=species:S:1:pos:R:3:dft_forces:R:3 '
                f"config_type={config_type} "
                f"system_name={system_name}"
            )
            if comment:  # append user-provided comment
                metadata += " " + comment

            out_f.write(metadata + "\n")

            # Atom lines
            for sym, pos, frc in zip(
                row["atomic_symbols"], row["cartesian_coordinates"], row["forces"]
            ):
                out_f.write(
                    f"{sym} {pos[0]:.12e} {pos[1]:.12e} {pos[2]:.12e} "
                    f"{frc[0]:.12e} {frc[1]:.12e} {frc[2]:.12e}\n"
                )

    return output_path
