import numpy as np

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
