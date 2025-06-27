
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
from qiskit.qasm2 import dumps  # Add this import at the top

from qiskit_aer import Aer
from qiskit import QuantumRegister, ClassicalRegister, transpile
from scipy.ndimage import gaussian_filter, median_filter
import numpy as np
def frqi_encode(image):
    """
    Encode a 2^n×2^n grayscale image into an FRQI circuit.
    `image` is a 2D numpy array of intensities [0,1].
    Returns a QuantumCircuit with n_row+n_col position qubits + 1 color qubit,
    and the row, col, color registers.
    """
    rows, cols = image.shape
    assert (rows & (rows-1) == 0) and (cols & (cols-1) == 0), "Size must be power of 2"
    n = int(np.log2(rows))
    row = QuantumRegister(n, name='r')
    col = QuantumRegister(n, name='col')
    color = QuantumRegister(1, name='color')
    qc = QuantumCircuit(row, col, color)
  

    qc.h(row)
    qc.h(col)
    for i in range(rows):
        for j in range(cols):
            theta = np.arcsin(image[i,j])
            if theta != 0:
                idx_bits = list(map(int, format(i, '0{}b'.format(n)))) + \
                           list(map(int, format(j, '0{}b'.format(n))))
                controls = [row[k] for k,bit in enumerate(idx_bits[:n]) if bit] + \
                           [col[k] for k,bit in enumerate(idx_bits[n:]) if bit]
                if controls:
                    qc.mcry(2*theta, controls, color[0], mode='noancilla')
                else:
                    qc.ry(2*theta, color[0])  # No controls, just apply rotation
    return qc, row, col, color
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
n = 4

thresholds = 0.1
contrast = 95
tol = 7
# Load and preprocess image
image = Image.open('/workspaces/AQA/images/Edge-Test-1.jpg').convert('L')
arr = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
contrast_low = 100-contrast    # Lower percentile for initial stretch
contrast_high = contrast  # Upper percentile for initial stretch
final_low = contrast_low+tol     # Lower percentile for final stretch
final_high = contrast_high-tol    # Upper percentile for final stretch

# Contrast stretching in-place
p2, p98 = np.percentile(arr, (contrast_low, contrast_high))
arr[:] = np.interp(arr, (p2, p98), (0, 1))

# Edge-preserving enhancement in-place
blurred = gaussian_filter(arr, sigma=0.5)
arr += 0.8 * (arr - blurred)
np.clip(arr, 0, 1, out=arr)

# Resize (float32)
arr = resize(arr, (2**n, 2**n), order=3, anti_aliasing=False, mode='reflect').astype(np.float32)

# Final contrast adjustment in-place
p5, p95 = np.percentile(arr, (final_low, final_high))
arr[:] = np.interp(arr, (p5, p95), (0, 1))


qc, row, col, color = frqi_encode(arr)  # <-- pass arr, not image
cr = ClassicalRegister(3, name='c')
qc.add_register(cr)
# Get n from image shape (since n is not global)

# Add ancilla qubits for Sobel results (horizontal and vertical)
anc = QuantumRegister(2, name='anc')
qc.add_register(anc)

# Always reuse existing 'c' register if present

# Sobel X: compare each pixel to its right neighbor
for i in range(4):
    for j in range(3):
        i_bits = list(map(int, format(i, f'0{n}b')))
        j_bits = list(map(int, format(j, f'0{n}b')))
        controls = []
        for k, bit in enumerate(i_bits):
            if bit:
                controls.append(row[k])
        for k, bit in enumerate(j_bits):
            if bit:
                controls.append(col[k])
        if controls:
            qc.mcx(controls, anc[0])

# Sobel Y: compare each pixel to its bottom neighbor
for i in range(3):
    for j in range(4):
        i_bits = list(map(int, format(i, f'0{n}b')))
        j_bits = list(map(int, format(j, f'0{n}b')))
        controls = []
        for k, bit in enumerate(i_bits):
            if bit:
                controls.append(row[k])
        for k, bit in enumerate(j_bits):
            if bit:
                controls.append(col[k])
        if controls:
            qc.mcx(controls, anc[1])

# Measure ancilla or color as needed
qc.measure(anc, cr[:2])    # measure Sobel edges into classical bits
qc.measure(color[0], cr[2])   # measure color qubit (optional)

qasm_str = dumps(qc)

qasm_str = dumps(qc)
