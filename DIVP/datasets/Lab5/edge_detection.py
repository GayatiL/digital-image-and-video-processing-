import cv2
import numpy as np
import matplotlib

# Use non-interactive backend to avoid Tkinter errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os

# =========================================================
# 1. LOAD INPUT IMAGE
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_PATH = os.path.join(BASE_DIR, "house.png")

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(
        "house.png not found inside Lab 5 folder."
    )

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Unable to load the input image.")

# Normalize image intensity
img = img.astype(np.float64) / 255.0

print("=" * 60)
print("       MARR-HILDRETH EDGE DETECTION")
print("=" * 60)

print("Image size:", img.shape)
print("Intensity range:", img.min(), "to", img.max())


# =========================================================
# 2. CREATE OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 3. ZERO CROSSING DETECTION FUNCTION
# =========================================================

def find_zero_crossings(L, threshold):

    rows, cols = L.shape

    edges = np.zeros((rows, cols), dtype=np.uint8)

    # Horizontal, vertical and diagonal directions
    directions = [
        (0, 1),
        (1, 0),
        (1, 1),
        (1, -1)
    ]

    for dy, dx in directions:

        y1 = max(0, -dy)
        y2 = min(rows, rows - dy)

        x1 = max(0, -dx)
        x2 = min(cols, cols - dx)

        A = L[y1:y2, x1:x2]
        B = L[y1+dy:y2+dy, x1+dx:x2+dx]

        # Detect sign changes
        crossing = (
            ((A < 0) & (B >= 0)) |
            ((A >= 0) & (B < 0))
        )

        # Apply threshold
        crossing &= (np.abs(A - B) > threshold)

        # Mark both pixels
        region1 = edges[y1:y2, x1:x2]
        region2 = edges[y1+dy:y2+dy, x1+dx:x2+dx]

        region1[crossing] = 255
        region2[crossing] = 255

    return edges


# =========================================================
# 4. SIGMA AND KERNEL PARAMETERS
# =========================================================

parameters = [
    (4, 25),
    (7, 35),
    (9, 45)
]


# =========================================================
# 5. PROCESS ALL THREE SIGMA VALUES
# =========================================================

for sigma, kernel_size in parameters:

    print("\n" + "-" * 60)
    print("Processing Sigma =", sigma)
    print("Gaussian Kernel =", kernel_size, "x", kernel_size)

    # -----------------------------------------------------
    # A. GAUSSIAN SMOOTHING
    # -----------------------------------------------------

    smoothed = cv2.GaussianBlur(
        img,
        (kernel_size, kernel_size),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT
    )

    # -----------------------------------------------------
    # B. LAPLACIAN OPERATOR
    # -----------------------------------------------------

    laplacian = cv2.Laplacian(
        smoothed,
        cv2.CV_64F,
        ksize=3
    )

    laplacian_magnitude = np.abs(laplacian)

    # -----------------------------------------------------
    # C. ZERO CROSSINGS T = 0
    # -----------------------------------------------------

    edges_T0 = find_zero_crossings(
        laplacian,
        threshold=0
    )

    # -----------------------------------------------------
    # D. ZERO CROSSINGS T = 4% MAX
    # -----------------------------------------------------

    max_response = np.max(np.abs(laplacian))

    T4 = 0.04 * max_response

    edges_T4 = find_zero_crossings(
        laplacian,
        threshold=T4
    )

    print("Maximum Laplacian response:", max_response)
    print("4% Threshold:", T4)

    # -----------------------------------------------------
    # E. CREATE 2 x 3 OUTPUT FIGURE
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 10)
    )

    # (a) Original Image
    axes[0, 0].imshow(
        img,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 0].set_title("(a) Original Image", pad=8)

    # (b) Smoothed Image
    axes[0, 1].imshow(
        smoothed,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, 1].set_title(
        f"(b) Smoothed Image (sigma={sigma})",
        pad=8
    )

    # (c) Laplacian Magnitude
    axes[0, 2].imshow(
        laplacian_magnitude,
        cmap="gray"
    )
    axes[0, 2].set_title(
        "(c) Laplacian Magnitude",
        pad=8
    )

    # (d) Signed Laplacian
    limit = np.max(np.abs(laplacian))

    axes[1, 0].imshow(
        laplacian,
        cmap="gray",
        vmin=-limit,
        vmax=limit
    )
    axes[1, 0].set_title(
        "(d) Gaussian + Laplacian",
        pad=8
    )

    # (e) Zero Crossings T = 0
    axes[1, 1].imshow(
        edges_T0,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[1, 1].set_title(
        "(e) Zero Crossings, T = 0",
        pad=8
    )

    # (f) Zero Crossings T = 4% Max
    axes[1, 2].imshow(
        edges_T4,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[1, 2].set_title(
        "(f) Zero Crossings, T = 4% Max",
        pad=8
    )

    # Remove axis markings
    for ax in axes.flat:
        ax.axis("off")

    # Main title
    fig.suptitle(
        f"Marr-Hildreth Edge Detection "
        f"(Sigma = {sigma}, Kernel = {kernel_size}x{kernel_size})",
        fontsize=15,
        fontweight="bold"
    )

    # Adjust spacing to prevent overlapping titles
    fig.subplots_adjust(
        top=0.90,
        bottom=0.05,
        left=0.03,
        right=0.97,
        hspace=0.22,
        wspace=0.08
    )

    # -----------------------------------------------------
    # F. SAVE OUTPUT FIGURE
    # -----------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"Marr_Hildreth_Sigma_{sigma}.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    print("Output saved:", output_path)

    # Close figure to release memory
    plt.close(fig)


# =========================================================
# 6. FINAL COMPLETION
# =========================================================

print("\n" + "=" * 60)
print("ALL THREE SIGMA RESULTS GENERATED SUCCESSFULLY!")
print("=" * 60)

print("\nCheck the output folder for:")

for sigma in [4, 7, 9]:
    print(f"Marr_Hildreth_Sigma_{sigma}.png")

print("\nProgram completed successfully.")