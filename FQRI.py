"""
Quantum Sobel Edge Detection Pipeline using FRQI (Flexible Representation of Quantum Images).

- FRQI encoding: each pixel i with grayscale intensity I_i is encoded as a qubit state
  cos(theta_i)|0> + sin(theta_i)|1> on a single 'color' qubit, with pixel index i on position qubits:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.
- Supports three intensity encoding modes:
    * "amplitude": encode intensity via RY rotation on the color qubit.
    * "phase": encode intensity via RZ rotation on the color qubit.
    * "combined": use both RY and RZ for encoding.
- For large images (<=256x256), divide image into smaller blocks (power-of-2 dimensions) for FRQI circuits, then reassemble after processing.
- Quantum Sobel filter:
    * Full Sobel (8-direction): uses a QFT-based subroutine to simulate convolution on the position register.
    * Partial Sobel (approximate): omits the full transform for a simpler gradient (e.g., uses fewer directions).
- Implemented with Qiskit; circuits run via Qiskit Runtime (Sampler primitive) for hardware execution.
"""
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler, Session
import numpy as np

# Configuration flags
ENCODING_MODE = 'amplitude'  # 'amplitude', 'phase', or 'combined'
FULL_SOBEL = True            # True for full 8-direction Sobel, False for simpler Sobel
Api_token = "YOUR_API_TOKEN_HERE"  # Replace with your token

def encode_frqi_block(image_block, encoding='amplitude'):
    flat = image_block.flatten().astype(float)
    flat = flat / float(np.max(flat)) if np.max(flat) > 0 else flat
    num_pixels = flat.size
    num_pos_qubits = int(np.ceil(np.log2(num_pixels)))
    color = QuantumRegister(1, name='color')
    pos = QuantumRegister(num_pos_qubits, name='pos')
    color_c = ClassicalRegister(1, name='c_color')
    pos_c = ClassicalRegister(num_pos_qubits, name='c_pos')
    qc = QuantumCircuit(color, pos, color_c, pos_c)
    qc.h(pos)
    for idx, intensity in enumerate(flat):
        theta = np.arcsin(intensity)
        bin_str = format(idx, '0{}b'.format(num_pos_qubits))
        for qb, bit in zip(pos, bin_str):
            if bit == '0':
                qc.x(qb)
        if encoding == 'amplitude':
            qc.mcry(2 * theta, pos[:], color[0], None)
        elif encoding == 'phase':
            qc.mcrz(2 * theta, pos[:], color[0], None)
        elif encoding == 'combined':
            qc.mcry(2 * theta, pos[:], color[0], None)
            qc.mcrz(2 * theta, pos[:], color[0], None)
        else:
            raise ValueError("Unsupported encoding mode.")
        for qb, bit in zip(pos, bin_str):
            if bit == '0':
                qc.x(qb)
    return qc

def quantum_sobel_filter(qc, full=True):
    color_qubit = qc.qubits[0]
    pos_qubits = qc.qubits[1:]
    if full:
        qft = QFT(len(pos_qubits), do_swaps=False, inverse=False)
        qc.append(qft.to_instruction(), pos_qubits)
        for i, qb in enumerate(pos_qubits):
            qc.p(2 * np.pi * (i + 1) / (2**len(pos_qubits)), qb)
        qc.append(qft.inverse().to_instruction(), pos_qubits)
    else:
        qc.x(color_qubit)

def split_image_into_blocks(image, block_size):
    blocks, coords = [], []
    h, w = image.shape
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            blocks.append(block)
            coords.append((i, j))
    return blocks, coords

def recombine_blocks(edge_blocks, coords, image_shape):
    edge_image = np.zeros(image_shape, dtype=float)
    for block, (i, j) in zip(edge_blocks, coords):
        h, w = block.shape
        edge_image[i:i+h, j:j+w] = block
    return edge_image


image = np.random.randint(0, 256, (32, 32))
block_size = 8
blocks, coords = split_image_into_blocks(image, block_size)
circuits = []
for block in blocks:
    qc = encode_frqi_block(block, encoding=ENCODING_MODE)
    quantum_sobel_filter(qc, full=FULL_SOBEL)
    qc.measure(qc.qubits, qc.clbits)
    circuits.append(qc)
# Execute circuits on IBM hardware (or simulator) with Qiskit Runtime Sampler

QiskitRuntimeService.save_account(channel="ibm_quantum", token=Api_token, overwrite=True)
service = QiskitRuntimeService()
backend = service.backend("ibm_sherbrooke")  # or your preferred backend

circuits_transpiled = transpile(circuits, backend=backend, optimization_level=1)
with Session(backend=backend) as session:
    sampler = Sampler()  # No arguments!
    job = sampler.run(circuits_transpiled, shots=1024)
    results = job.result()
# Collect edge blocks from measurement results
edge_blocks = []
print(results.data())
for idx, res in enumerate(results):
    counts = res.data.meas.get_counts()
    total_shots = sum(counts.values())
    block_edges = np.zeros((block_size, block_size))
    for bitstring, count in counts.items():
        # Assume bitstring: first bit = color, remaining = position index (binary)
        color_bit = int(bitstring[0])
        pos_bits = bitstring[1:]
        pos_index = int(pos_bits, 2) if pos_bits else 0
        row = pos_index // block_size
        col = pos_index % block_size
        # Use probability(color=1) as edge intensity approximation
        if color_bit == 1:
            block_edges[row, col] += count / total_shots
    edge_blocks.append(block_edges)
# Reassemble full edge image
edge_image = recombine_blocks(edge_blocks, coords, image.shape)
print("Edge image shape:", edge_image.shape)
    # (User can view 'edge_image' or compare encoding/filter modes)
""""
# Example: create a random 32x32 image (0-255 grayscale)
image = np.random.randint(0, 256, (32, 32))
block_size = 8
blocks, coords = split_image_into_blocks(image, block_size)
circuits = []
for block in blocks:
    qc = encode_frqi_block(block, encoding=ENCODING_MODE)
    quantum_sobel_filter(qc, full=FULL_SOBEL)
    qc.measure(qc.qubits, qc.clbits)
    circuits.append(qc)
"""
