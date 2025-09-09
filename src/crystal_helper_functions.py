import numpy as np

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
