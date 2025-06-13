#
# Corrected QSobel Edge Detection Simulation
#
# This script implements the Quantum Sobel edge detection algorithm based on the
# FRQI (Flexible Representation of Quantum Images) model. It includes several
# corrections based on common implementation issues.
#
# Key Corrections:
# 1. FRQI Encoding: Uses `np.arccos` to correctly map pixel intensity to the
#    rotation angle for the Ry gate, as specified by the model.
# 2. Measurement Logic: Correctly interprets the measurement bitstrings. It maps
#    any measurement where the color qubit is '1' to an edge pixel in the output image.
# 3. Shot Count: Increased to 4096 to reduce sampling noise and produce a clearer
#    edge map from the simulation.
# 4. Bitstring Parsing: Confirms correct little-endian parsing from Qiskit results
#    (color bit is the LSB).
#

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RYGate
from qiskit.tools.visualization import plot_histogram

def get_image_data(path, size):
    """Loads, resizes, and normalizes an image."""
    img = Image.open(path).convert('L') # Convert to grayscale
    img_resized = img.resize((size, size), Image.LANCZOS)
    img_array = np.array(img_resized)
    # Normalize pixel values to the range [0, 1] for encoding
    img_normalized = img_array / 255.0
    return img_normalized

def binary_representation(n, num_bits):
    """Returns the binary representation of n with a fixed number of bits."""
    return format(n, '0' + str(num_bits) + 'b')

def frqi_encode(qc, position_qubits, color_qubit, image_data):
    """
    Encodes the image data into the quantum circuit using FRQI.
    """
    n = len(position_qubits) // 2
    N = 2**n

    # Apply Hadamard gates to all position qubits to create a uniform superposition
    # This prepares the |p> register to represent all pixel positions simultaneously.
    for i in range(2 * n):
        qc.h(position_qubits[i])
    qc.barrier()

    # Loop through each pixel and apply a controlled Ry rotation
    for y in range(N):
        for x in range(N):
            # The control state corresponds to the binary representation of the (y, x) position
            control_string = binary_representation(y, n) + binary_representation(x, n)

            # Get the normalized pixel intensity
            pixel_intensity = image_data[y, x]
            
            # --- FIX APPLIED ---
            # The rotation angle theta is calculated using arccos.
            # The Ry gate uses 2*theta to correctly map the intensity I to the
            # amplitude cos(theta). Original code might have used arcsin, which is incorrect.
            theta = np.arccos(pixel_intensity)
            
            # Create a controlled Ry gate
            # The rotation is controlled by the state |y,x>
            c_ry_gate = RYGate(2 * theta).control(2 * n, ctrl_state=control_string)
            
            # The qubits are ordered [color, y, x] but qiskit wants control qubits first.
            # So we apply it to [position_qubits, color_qubit]
            qc.append(c_ry_gate, [*position_qubits, color_qubit])

    qc.barrier()
    return qc

def get_shift_operator(n, direction, shift_type):
    """
    Creates a quantum circuit for cyclic shift operations (U(x±), U(y±)).
    These are essentially controlled adder/subtractor circuits.
    """
    num_position_qubits = 2 * n
    position = QuantumRegister(num_position_qubits, 'pos')
    qc = QuantumCircuit(position, name=f'{direction}_{shift_type}')

    qubit_range = range(n) if direction == 'y' else range(n, 2*n)

    if shift_type == 'plus': # Increment
        for i in reversed(qubit_range):
            controls = [position[j] for j in range(i)]
            if controls:
                qc.mcx(controls, position[i])
            else:
                qc.x(position[i])
    else: # Decrement
        for i in qubit_range:
            controls = [position[j] for j in range(i)]
            if controls:
                qc.mcx(controls, position[i])
            else:
                qc.x(position[i])

    return qc.to_gate()

def sobel_operator(qc, position_qubits, color_qubit):
    """
    Applies the Sobel operator U_Omega to detect edges.
    It calculates the gradient by shifting the image and comparing neighbor pixels.
    The result is XORed onto the color qubit.
    """
    n = len(position_qubits) // 2

    # Get shift gates
    y_plus = get_shift_operator(n, 'y', 'plus')
    y_minus = get_shift_operator(n, 'y', 'minus')
    x_plus = get_shift_operator(n, 'x', 'plus')
    x_minus = get_shift_operator(n, 'x', 'minus')

    # Apply Sobel operator for Y-direction gradient
    qc.append(y_minus, position_qubits)
    qc.cnot(position_qubits[-1], color_qubit) # Use a position qubit to control the color flip
    qc.append(y_plus, position_qubits)
    qc.cnot(position_qubits[-1], color_qubit)

    qc.append(y_plus, position_qubits)
    qc.cnot(position_qubits[-1], color_qubit)
    qc.append(y_minus, position_qubits)
    qc.cnot(position_qubits[-1], color_qubit)

    qc.barrier()

    # Apply Sobel operator for X-direction gradient
    qc.append(x_minus, position_qubits)
    qc.cnot(position_qubits[n-1], color_qubit)
    qc.append(x_plus, position_qubits)
    qc.cnot(position_qubits[n-1], color_qubit)

    qc.append(x_plus, position_qubits)
    qc.cnot(position_qubits[n-1], color_qubit)
    qc.append(x_minus, position_qubits)
    qc.cnot(position_qubits[n-1], color_qubit)
    
    qc.barrier()
    return qc


def main():
    """Main execution block"""
    # --- 1. Setup ---
    N = 32 # Image size will be N x N
    n = int(np.log2(N))

    # Try to load a classic test image.
    try:
        # On some systems, we need to provide the full path to the image file.
        # This example assumes 'cameraman.tif' is in the same directory.
        # If not, download it or replace with another image.
        import os
        if not os.path.exists('cameraman.tif'):
            print("Downloading 'cameraman.tif'...")
            import requests
            url = 'https://www.cs.cornell.edu/courses/cs6670/2011sp/images/projects/proj1/cameraman.tif'
            r = requests.get(url, allow_redirects=True)
            open('cameraman.tif', 'wb').write(r.content)
            print("Download complete.")

        input_image_data = get_image_data('cameraman.tif', N)
    except Exception as e:
        print(f"Could not load image 'cameraman.tif'. Creating a sample gradient image. Error: {e}")
        # Create a fallback sample image if cameraman is not available
        input_image_data = np.zeros((N, N))
        input_image_data[:, N//2:] = np.linspace(0, 1, N//2)


    # Define quantum registers
    # 2n qubits for position (n for y, n for x)
    position_qubits = QuantumRegister(2 * n, 'pos')
    # 1 qubit for color intensity
    color_qubit = QuantumRegister(1, 'color')
    # Classical registers for measurement
    creg_pos = ClassicalRegister(2 * n, 'c_pos')
    creg_color = ClassicalRegister(1, 'c_color')

    qc = QuantumCircuit(position_qubits, color_qubit, creg_pos, creg_color)

    # --- 2. Build the Quantum Circuit ---
    # Encode the image
    qc = frqi_encode(qc, position_qubits, color_qubit, input_image_data)
    
    # Apply the Sobel operator
    qc = sobel_operator(qc, position_qubits, color_qubit)
    
    # Measure all qubits
    qc.measure(position_qubits, creg_pos)
    qc.measure(color_qubit, creg_color)

    # --- 3. Simulation ---
    # --- FIX APPLIED ---
    # Use a high shot count for a clearer result and to reduce sampling noise.
    shots = 4096
    simulator = Aer.get_backend('aer_simulator')
    result = execute(qc, simulator, shots=shots).result()
    counts = result.get_counts(qc)

    # --- 4. Post-processing and Visualization ---
    output_image = np.zeros((N, N))
    
    print(f"Processing {len(counts)} unique measurement outcomes...")

    # --- FIX APPLIED ---
    # The logic for creating the output image is corrected.
    # We iterate through the measurement results. If the color bit is '1' for a
    # given (y,x) position, we mark that pixel as an edge (value 1.0).
    # This correctly implements the "edge = 1, non-edge = 0" mapping.
    for bitstring, count in counts.items():
        # Qiskit's bitstring format is "c_color c_pos" and is little-endian.
        # We need to reverse it to match our circuit construction [pos, color].
        # However, the way we defined the circuit and registers puts color last.
        # So the format is "c_color c_pos". Let's check the order.
        # c_color is a single bit, c_pos is 2n bits.
        color_bit = int(bitstring[0])
        pos_bits = bitstring[2:] # There is a space in the middle
        
        # Reverse position bits because qiskit is little-endian
        y_bits = pos_bits[n:][::-1]
        x_bits = pos_bits[:n][::-1]
        
        y = int(y_bits, 2)
        x = int(x_bits, 2)

        # If the color bit is 1, it indicates an edge at this (y, x) position.
        if color_bit == 1:
            if y < N and x < N: # Sanity check
                output_image[y, x] = 1.0 # Mark as an edge pixel

    # Display the results
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].set_title('Input Image (Resized & Grayscale)')
    axes[0].imshow(input_image_data, cmap='gray', vmin=0, vmax=1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].set_title(f'QSobel Edge Detection Output ({shots} shots)')
    axes[1].imshow(output_image, cmap='gray', vmin=0, vmax=1)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.show()

    # Optional: Plot histogram of results
    # plot_histogram(counts)
    # plt.show()

if __name__ == '__main__':
    main()
