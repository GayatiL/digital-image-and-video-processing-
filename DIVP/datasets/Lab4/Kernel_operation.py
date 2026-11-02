import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. KERNELS GIVEN BY PROFESSOR
# ============================================================

# Second derivative - Laplacian
kernel_a = np.array([
    [1,  1,  1],
    [1, -8,  1],
    [1,  1,  1]
])

# First derivative - horizontal
kernel_b = np.array([
    [-1, -1, -1],
    [ 2,  2,  2],
    [-1, -1, -1]
])

# First derivative - vertical
kernel_c = np.array([
    [-1,  2, -1],
    [-1,  2, -1],
    [-1,  2, -1]
])

# Second derivative - diagonal
kernel_d = np.array([
    [ 2, -1, -1],
    [-1,  2, -1],
    [-1, -1,  2]
])


# ============================================================
# 2. FUNCTION FOR CONVOLUTION
# ============================================================

def apply_kernel(image, kernel):

    m, n = image.shape
    km, kn = kernel.shape

    # Padding
    pad_m = km // 2
    pad_n = kn // 2

    padded = np.pad(
        image,
        ((pad_m, pad_m), (pad_n, pad_n)),
        mode='constant',
        constant_values=0
    )

    output = np.zeros_like(image, dtype=float)

    for i in range(m):
        for j in range(n):

            region = padded[
                i:i+km,
                j:j+kn
            ]

            output[i, j] = np.sum(region * kernel)

    return output


# ============================================================
# 3. CREATE TEST IMAGE - 1 PIXEL WIDE
# ============================================================

image_1pixel = np.zeros((100, 100), dtype=float)

# Diagonal lines similar to professor's drawing
for i in range(20, 45):
    image_1pixel[i, i] = 255

for i in range(40, 75):
    image_1pixel[i, i+15] = 255

for i in range(60, 90):
    image_1pixel[i, i-10] = 255


# ============================================================
# 4. CREATE TEST IMAGE - 3 PIXEL WIDE
# ============================================================

image_3pixel = np.zeros((100, 100), dtype=float)

# Vertical 3-pixel-wide lines
image_3pixel[20:80, 25:28] = 255
image_3pixel[20:80, 50:53] = 255
image_3pixel[20:80, 75:78] = 255


# ============================================================
# 5. APPLY KERNELS TO 1-PIXEL IMAGE
# ============================================================

result_1_a = apply_kernel(image_1pixel, kernel_a)
result_1_b = apply_kernel(image_1pixel, kernel_b)
result_1_c = apply_kernel(image_1pixel, kernel_c)
result_1_d = apply_kernel(image_1pixel, kernel_d)


# ============================================================
# 6. APPLY KERNELS TO 3-PIXEL IMAGE
# ============================================================

result_3_a = apply_kernel(image_3pixel, kernel_a)
result_3_b = apply_kernel(image_3pixel, kernel_b)
result_3_c = apply_kernel(image_3pixel, kernel_c)
result_3_d = apply_kernel(image_3pixel, kernel_d)


# ============================================================
# 7. DISPLAY RESULTS - 1 PIXEL IMAGE
# ============================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_1pixel, cmap='gray')
plt.title("Original - 1 Pixel")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(result_1_a, cmap='gray')
plt.title("Kernel A - 2nd Derivative")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(result_1_b, cmap='gray')
plt.title("Kernel B - 1st Derivative")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(result_1_c, cmap='gray')
plt.title("Kernel C - 1st Derivative")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(result_1_d, cmap='gray')
plt.title("Kernel D - 2nd Derivative")
plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# 8. DISPLAY RESULTS - 3 PIXEL IMAGE
# ============================================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(image_3pixel, cmap='gray')
plt.title("Original - 3 Pixel Wide")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(result_3_a, cmap='gray')
plt.title("Kernel A - 2nd Derivative")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(result_3_b, cmap='gray')
plt.title("Kernel B - 1st Derivative")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(result_3_c, cmap='gray')
plt.title("Kernel C - 1st Derivative")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(result_3_d, cmap='gray')
plt.title("Kernel D - 2nd Derivative")
plt.axis('off')

plt.tight_layout()
plt.show()