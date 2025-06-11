# New QSED implementation based on eight-direction Sobel operator for NEQR
# This builds upon ideas from the AQA repo and implements the improved method described in the 2023 paper.

import numpy as np
import cv2
import matplotlib.pyplot as plt

# === 1. Load image and convert to grayscale ===
image = cv2.imread('/workspaces/AQA/images/Edge-Test-2.jpg', cv2.IMREAD_GRAYSCALE)
assert image is not None, "Image not found"

# === 2. Define 8-direction 5x5 Sobel kernels ===
def get_8dir_sobel_kernels():
    # Kernels defined as per the paper
    G0 = np.array([
        [0, 0, 0, 0, 0],
        [-1, -2, -4, -2, -1],
        [0, 0, 0, 0, 0],
        [1, 2, 4, 2, 1],
        [0, 0, 0, 0, 0]
    ])
    G22_5 = np.array([
        [0, 0, 0, 0, 0],
        [0, -2, -4, -2, 0],
        [-1, -4, 0, 4, 1],
        [0, 2, 4, 2, 0],
        [0, 0, 0, 0, 0]
    ])
    G45 = np.array([
        [0, 0, 0, -1, 0],
        [0, -2, -4, 0, 1],
        [0, -4, 0, 4, 0],
        [-1, 0, 4, 2, 0],
        [0, 1, 0, 0, 0]
    ])
    G67_5 = np.array([
        [0, 0, -1, 0, 0],
        [0, -2, -4, 2, 0],
        [0, -4, 0, 4, 0],
        [0, -2, 4, 2, 0],
        [0, 0, 1, 0, 0]
    ])
    G90 = np.array([
        [0, -1, 0, 1, 0],
        [0, -2, 0, 2, 0],
        [0, -4, 0, 4, 0],
        [0, -2, 0, 2, 0],
        [0, -1, 0, 1, 0]
    ])
    G112_5 = np.array([
        [0, 0, 1, 0, 0],
        [0, -2, 4, 2, 0],
        [0, -4, 0, 4, 0],
        [0, -2, -4, 2, 0],
        [0, 0, -1, 0, 0]
    ])
    G135 = np.array([
        [0, 1, 0, 0, 0],
        [-1, 0, 4, 2, 0],
        [0, -4, 0, 4, 0],
        [0, -2, -4, 0, 1],
        [0, 0, 0, -1, 0]
    ])
    G157_5 = np.array([
        [0, 0, 0, 0, 0],
        [0, 2, 4, 2, 0],
        [-1, -4, 0, 4, 1],
        [0, -2, -4, -2, 0],
        [0, 0, 0, 0, 0]
    ])
    return [G0, G22_5, G45, G67_5, G90, G112_5, G135, G157_5]

# === 3. Apply all 8 filters and compute max magnitude ===
def eight_dir_sobel(image):
    kernels = get_8dir_sobel_kernels()
    gradients = [cv2.filter2D(image.astype(np.float64), -1, k) for k in kernels]
    magnitude = np.max(np.abs(gradients), axis=0)
    return np.uint8(np.clip(magnitude, 0, 255))

# === 4. Process image ===
edges = eight_dir_sobel(image)

# === 5. Visualize ===
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("8-dir Q-Sobel Edges")
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
