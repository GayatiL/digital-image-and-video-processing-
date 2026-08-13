# ================================================================
# PCC-03: Digital Image and Video Processing
# Lab: Spatial Filtering
# ================================================================


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


# ================================================================
# SETTINGS
# ================================================================

#INPUT_IMAGE = "agv_clean.png"
#OUTPUT_DIR = "outputs"



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_IMAGE = os.path.join(
    BASE_DIR,
    "agv_clean.png"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

# Motion blur length required by the problem statement:
# 7-9 pixels. We choose 9 pixels.
MOTION_BLUR_LENGTH = 9

# Averaging filter sizes
AVERAGING_SIZES = [3, 5, 9]

# Noise levels
NOISE_LEVELS = [10, 25]

# High boost values
K_VALUES = [1, 1.5, 2, 3]

# Maximum image dimension.
# This keeps the from-scratch loops reasonably fast in VS Code.
# Set to None if you want to process the original resolution.
MAX_DIMENSION = 800


# ================================================================
# CREATE OUTPUT DIRECTORY
# ================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# IMAGE LOADING
# ================================================================

def load_grayscale_image(filename):
    """
    Load an image and convert it to grayscale.

    Returns:
        image as floating-point NumPy array in range 0-255.
    """

    image = Image.open(filename).convert("L")

    # Resize only to keep from-scratch filtering computationally
    # manageable. This does NOT perform the required filtering.
    if MAX_DIMENSION is not None:

        width, height = image.size

        largest_dimension = max(width, height)

        if largest_dimension > MAX_DIMENSION:

            scale = MAX_DIMENSION / largest_dimension

            new_width = int(width * scale)
            new_height = int(height * scale)

            image = image.resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS
            )

    image = np.array(image, dtype=np.float64)

    return image


# ================================================================
# SAVE IMAGE
# ================================================================

def save_image(image, filename):
    """
    Save a floating-point image safely as an 8-bit PNG.
    """

    image = np.clip(image, 0, 255)

    image_uint8 = image.astype(np.uint8)

    Image.fromarray(image_uint8).save(filename)


# ================================================================
# PADDING
# ================================================================

def pad_image(image, pad):
    """
    Zero-pad the image.

    This is used by our from-scratch correlation function.
    """

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="edge"
    )

    return padded


# ================================================================
# FROM-SCRATCH 2D CORRELATION
# ================================================================

def correlate2d(image, kernel):
    """
    Perform 2D spatial correlation from scratch.

    IMPORTANT:
    This is correlation, NOT convolution.

    In correlation, the kernel is NOT flipped.

    Parameters:
        image  : 2D grayscale image
        kernel : 2D filter kernel

    Returns:
        filtered image
    """

    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    # This implementation assumes odd-sized kernels.
    pad_y = kernel_height // 2
    pad_x = kernel_width // 2

    padded = pad_image(image, max(pad_y, pad_x))

    output = np.zeros_like(image, dtype=np.float64)

    # ------------------------------------------------------------
    # From-scratch nested loops
    # ------------------------------------------------------------

    for i in range(image_height):

        for j in range(image_width):

            total = 0.0

            for ki in range(kernel_height):

                for kj in range(kernel_width):

                    total += (
                        padded[i + ki, j + kj]
                        * kernel[ki, kj]
                    )

            output[i, j] = total

    return output


# ================================================================
# CREATE AVERAGING / BOX FILTER
# ================================================================

def create_average_kernel(size):
    """
    Create a size x size averaging kernel.

    Example for 3x3:

    1/9 * [[1,1,1],
           [1,1,1],
           [1,1,1]]
    """

    kernel = np.ones((size, size), dtype=np.float64)

    kernel = kernel / (size * size)

    return kernel


# ================================================================
# APPLY AVERAGING FILTER
# ================================================================

def averaging_filter(image, size):

    kernel = create_average_kernel(size)

    output = correlate2d(image, kernel)

    return np.clip(output, 0, 255)


# ================================================================
# GENERATE GAUSSIAN NOISE
# ================================================================

def add_gaussian_noise(image, sigma):

    """
    Add Gaussian noise.

    Noise:
        N(0, sigma^2)

    sigma = 10 and 25 are required by the assignment.
    """

    noise = np.random.normal(
        loc=0,
        scale=sigma,
        size=image.shape
    )

    noisy_image = image + noise

    noisy_image = np.clip(
        noisy_image,
        0,
        255
    )

    return noisy_image


# ================================================================
# CREATE MOTION BLUR KERNEL
# ================================================================

def create_motion_blur_kernel(length=9):
    """
    Create a horizontal linear motion-blur kernel.

    Example for length = 9:

    [1 1 1 1 1 1 1 1 1] / 9
    """

    kernel = np.zeros((length, length), dtype=np.float64)

    center = length // 2

    for i in range(length):

        kernel[center, i] = 1.0

    kernel = kernel / length

    return kernel


# ================================================================
# APPLY MOTION BLUR
# ================================================================

def apply_motion_blur(image, length=9):

    kernel = create_motion_blur_kernel(length)

    blurred = correlate2d(image, kernel)

    return np.clip(blurred, 0, 255)


# ================================================================
# LAPLACIAN KERNELS
# ================================================================

def get_laplacian_4():

    """
    4-neighbor Laplacian:

        [ 0 -1  0 ]
        [-1  4 -1 ]
        [ 0 -1  0 ]
    """

    kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float64)

    return kernel


def get_laplacian_8():

    """
    8-neighbor Laplacian:

        [-1 -1 -1]
        [-1  8 -1]
        [-1 -1 -1]
    """

    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float64)

    return kernel


# ================================================================
# LAPLACIAN SHARPENING
# ================================================================

def laplacian_sharpen(image, kernel):

    """
    Laplacian sharpening.

    First calculate:

        L(x,y) = f(x,y) * Laplacian_kernel

    Then:

        g(x,y) = f(x,y) - L(x,y)

    """

    laplacian_response = correlate2d(
        image,
        kernel
    )

    sharpened = image - laplacian_response

    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened, laplacian_response


# ================================================================
# UNSHARP MASKING / HIGH-BOOST FILTERING
# ================================================================

def high_boost_filter(image, k):

    """
    High-boost filtering.

    First calculate a blurred image:

        f_blur = averaging_filter(f)

    Mask:

        mask = f - f_blur

    High boost:

        g = f + k * mask

    When k = 1:
        unsharp masking

    k > 1:
        high-boost filtering
    """

    # 3x3 smoothing is used for the high-boost operation.
    blurred = averaging_filter(image, 3)

    mask = image - blurred

    sharpened = image + k * mask

    sharpened = np.clip(
        sharpened,
        0,
        255
    )

    return sharpened


# ================================================================
# PSNR
# ================================================================

def calculate_psnr(original, processed):

    """
    Calculate Peak Signal-to-Noise Ratio.

    MSE:

        MSE = mean((original - processed)^2)

    PSNR:

        PSNR = 10 log10(255^2 / MSE)

    Higher PSNR generally means the processed image
    is numerically closer to the clean ground truth.
    """

    mse = np.mean(
        (original - processed) ** 2
    )

    if mse == 0:

        return float("inf")

    psnr = 10 * np.log10(
        (255 ** 2) / mse
    )

    return psnr


# ================================================================
# SHARPNESS METRIC
# ================================================================

def calculate_sharpness(image):

    """
    Sharpness metric:

        Variance of Laplacian

    A higher value generally indicates stronger
    high-frequency/edge information.
    """

    laplacian_kernel = get_laplacian_4()

    laplacian = correlate2d(
        image,
        laplacian_kernel
    )

    sharpness = np.var(laplacian)

    return sharpness


# ================================================================
# DISPLAY IMAGE GRID
# ================================================================

def create_grid(images, titles, filename, main_title):

    number_of_images = len(images)

    columns = 3

    rows = int(np.ceil(
        number_of_images / columns
    ))

    plt.figure(
        figsize=(15, 5 * rows)
    )

    for index in range(number_of_images):

        plt.subplot(
            rows,
            columns,
            index + 1
        )

        plt.imshow(
            images[index],
            cmap="gray",
            vmin=0,
            vmax=255
        )

        plt.title(titles[index])

        plt.axis("off")

    plt.suptitle(
        main_title,
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ================================================================
# MAIN PROGRAM
# ================================================================

def main():

    print("\n")
    print("=" * 70)
    print("AGV SPATIAL FILTERING LAB")
    print("=" * 70)


    # ------------------------------------------------------------
    # STEP 1: LOAD CLEAN IMAGE
    # ------------------------------------------------------------

    print("\n[1] Loading clean image...")

    clean = load_grayscale_image(
        INPUT_IMAGE
    )

    print(
        "Image size:",
        clean.shape
    )

    save_image(
        clean,
        os.path.join(
            OUTPUT_DIR,
            "01_clean_ground_truth.png"
        )
    )


    # ------------------------------------------------------------
    # STEP 2: GENERATE GAUSSIAN NOISE
    # ------------------------------------------------------------

    print("\n[2] Generating Gaussian noise...")

    noisy_images = {}

    for sigma in NOISE_LEVELS:

        noisy = add_gaussian_noise(
            clean,
            sigma
        )

        noisy_images[sigma] = noisy

        save_image(
            noisy,
            os.path.join(
                OUTPUT_DIR,
                f"02_noisy_sigma_{sigma}.png"
            )
        )

        print(
            f"Generated noisy image: sigma = {sigma}"
        )


    # ------------------------------------------------------------
    # STEP 3: GENERATE MOTION BLUR
    # ------------------------------------------------------------

    print("\n[3] Generating motion blur...")

    motion_blurred = apply_motion_blur(
        clean,
        MOTION_BLUR_LENGTH
    )

    save_image(
        motion_blurred,
        os.path.join(
            OUTPUT_DIR,
            "03_motion_blurred.png"
        )
    )

    print(
        "Motion blur length:",
        MOTION_BLUR_LENGTH
    )


    # ============================================================
    # TASK 1
    # NOISE SUPPRESSION VIA AVERAGING
    # ============================================================

    print("\n")
    print("=" * 70)
    print("TASK 1 - AVERAGING FILTER")
    print("=" * 70)

    task1_images = []
    task1_titles = []

    results = []

    for sigma in NOISE_LEVELS:

        noisy = noisy_images[sigma]

        for size in AVERAGING_SIZES:

            print(
                f"Filtering sigma={sigma}, "
                f"kernel={size}x{size}"
            )

            filtered = averaging_filter(
                noisy,
                size
            )

            # Save output
            filename = os.path.join(
                OUTPUT_DIR,
                f"task1_sigma_{sigma}_{size}x{size}.png"
            )

            save_image(
                filtered,
                filename
            )

            # Add to grid
            task1_images.append(
                filtered
            )

            task1_titles.append(
                f"σ={sigma}, {size}×{size}"
            )

            # Metrics
            psnr = calculate_psnr(
                clean,
                filtered
            )

            sharpness = calculate_sharpness(
                filtered
            )

            results.append({
                "Task": "Task 1",
                "Configuration":
                    f"Average {size}x{size}, sigma={sigma}",
                "Noise Sigma": sigma,
                "Kernel Size": size,
                "k": None,
                "Laplacian": None,
                "PSNR (dB)": psnr,
                "Sharpness": sharpness
            })


    create_grid(
        task1_images,
        task1_titles,
        os.path.join(
            OUTPUT_DIR,
            "TASK1_averaging_grid.png"
        ),
        "Task 1 - Averaging Filter Results"
    )


    # ============================================================
    # TASK 2
    # LAPLACIAN SHARPENING
    # ============================================================

    print("\n")
    print("=" * 70)
    print("TASK 2 - LAPLACIAN SHARPENING")
    print("=" * 70)


    laplacian4 = get_laplacian_4()

    laplacian8 = get_laplacian_8()


    # 4-neighbor
    sharpened4, response4 = laplacian_sharpen(
        motion_blurred,
        laplacian4
    )

    # 8-neighbor
    sharpened8, response8 = laplacian_sharpen(
        motion_blurred,
        laplacian8
    )


    save_image(
        sharpened4,
        os.path.join(
            OUTPUT_DIR,
            "task2_laplacian_4_neighbor_sharpened.png"
        )
    )

    save_image(
        sharpened8,
        os.path.join(
            OUTPUT_DIR,
            "task2_laplacian_8_neighbor_sharpened.png"
        )
    )


    # Response maps need normalization only for visualization.
    def normalize_for_display(image):

        minimum = image.min()

        maximum = image.max()

        if maximum - minimum == 0:

            return np.zeros_like(image)

        normalized = (
            (image - minimum)
            / (maximum - minimum)
            * 255
        )

        return normalized


    response4_display = normalize_for_display(
        response4
    )

    response8_display = normalize_for_display(
        response8
    )


    save_image(
        response4_display,
        os.path.join(
            OUTPUT_DIR,
            "task2_4_neighbor_response.png"
        )
    )

    save_image(
        response8_display,
        os.path.join(
            OUTPUT_DIR,
            "task2_8_neighbor_response.png"
        )
    )


    # Display Task 2
    create_grid(
        [
            motion_blurred,
            response4_display,
            response8_display,
            sharpened4,
            sharpened8
        ],
        [
            "Motion Blurred",
            "4-Neighbour Laplacian Response",
            "8-Neighbour Laplacian Response",
            "4-Neighbour Sharpened",
            "8-Neighbour Sharpened"
        ],
        os.path.join(
            OUTPUT_DIR,
            "TASK2_laplacian_grid.png"
        ),
        "Task 2 - Laplacian Sharpening"
    )


    # Metrics for Task 2
    psnr4 = calculate_psnr(
        clean,
        sharpened4
    )

    sharpness4 = calculate_sharpness(
        sharpened4
    )

    psnr8 = calculate_psnr(
        clean,
        sharpened8
    )

    sharpness8 = calculate_sharpness(
        sharpened8
    )


    results.append({
        "Task": "Task 2",
        "Configuration": "Laplacian 4-neighbor",
        "Noise Sigma": None,
        "Kernel Size": None,
        "k": None,
        "Laplacian": "4-neighbor",
        "PSNR (dB)": psnr4,
        "Sharpness": sharpness4
    })


    results.append({
        "Task": "Task 2",
        "Configuration": "Laplacian 8-neighbor",
        "Noise Sigma": None,
        "Kernel Size": None,
        "k": None,
        "Laplacian": "8-neighbor",
        "PSNR (dB)": psnr8,
        "Sharpness": sharpness8
    })


    # ============================================================
    # TASK 3
    # UNSHARP MASKING / HIGH-BOOST
    # ============================================================

    print("\n")
    print("=" * 70)
    print("TASK 3 - UNSHARP MASKING / HIGH-BOOST")
    print("=" * 70)


    sigma10_image = noisy_images[10]

    task3_images = []
    task3_titles = []

    k_sharpness = []


    for k in K_VALUES:

        print(
            f"Applying high-boost filtering: k={k}"
        )

        output = high_boost_filter(
            sigma10_image,
            k
        )

        save_image(
            output,
            os.path.join(
                OUTPUT_DIR,
                f"task3_highboost_k_{k}.png"
            )
        )

        task3_images.append(
            output
        )

        if k == 1:

            task3_titles.append(
                "Unsharp Masking (k=1)"
            )

        else:

            task3_titles.append(
                f"High-Boost (k={k})"
            )


        psnr = calculate_psnr(
            clean,
            output
        )

        sharpness = calculate_sharpness(
            output
        )

        k_sharpness.append(
            sharpness
        )

        results.append({
            "Task": "Task 3",
            "Configuration": f"High-Boost k={k}",
            "Noise Sigma": 10,
            "Kernel Size": 3,
            "k": k,
            "Laplacian": None,
            "PSNR (dB)": psnr,
            "Sharpness": sharpness
        })


    create_grid(
        task3_images,
        task3_titles,
        os.path.join(
            OUTPUT_DIR,
            "TASK3_highboost_grid.png"
        ),
        "Task 3 - Unsharp Masking / High-Boost"
    )


    # ------------------------------------------------------------
    # SHARPNESS vs k GRAPH
    # ------------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        K_VALUES,
        k_sharpness,
        marker="o"
    )

    plt.xlabel(
        "Boost Factor k"
    )

    plt.ylabel(
        "Sharpness - Variance of Laplacian"
    )

    plt.title(
        "Sharpness vs High-Boost Factor"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "TASK3_sharpness_vs_k.png"
        ),
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # ============================================================
    # REFLECTION QUESTION 1
    #
    # Compare:
    #       Denoise -> Sharpen
    #
    # versus:
    #       Sharpen -> Denoise
    # ============================================================

    print("\n")
    print("=" * 70)
    print("REFLECTION QUESTION 1")
    print("DENOISE -> SHARPEN VS SHARPEN -> DENOISE")
    print("=" * 70)


    # -------------------------------
    # DENOISE -> SHARPEN
    # -------------------------------

    denoised = averaging_filter(
        sigma10_image,
        5
    )

    pipeline_A, _ = laplacian_sharpen(
        denoised,
        laplacian4
    )


    # -------------------------------
    # SHARPEN -> DENOISE
    # -------------------------------

    sharpen_first, _ = laplacian_sharpen(
        sigma10_image,
        laplacian4
    )

    pipeline_B = averaging_filter(
        sharpen_first,
        5
    )


    psnr_A = calculate_psnr(
        clean,
        pipeline_A
    )

    sharpness_A = calculate_sharpness(
        pipeline_A
    )

    psnr_B = calculate_psnr(
        clean,
        pipeline_B
    )

    sharpness_B = calculate_sharpness(
        pipeline_B
    )


    save_image(
        pipeline_A,
        os.path.join(
            OUTPUT_DIR,
            "reflection_denoise_then_sharpen.png"
        )
    )

    save_image(
        pipeline_B,
        os.path.join(
            OUTPUT_DIR,
            "reflection_sharpen_then_denoise.png"
        )
    )


    print("\nDenoise -> Sharpen")
    print(
        "PSNR:",
        round(psnr_A, 3),
        "dB"
    )

    print(
        "Sharpness:",
        round(sharpness_A, 3)
    )


    print("\nSharpen -> Denoise")
    print(
        "PSNR:",
        round(psnr_B, 3),
        "dB"
    )

    print(
        "Sharpness:",
        round(sharpness_B, 3)
    )


    # ============================================================
    # TASK 4
    # SAVE RESULTS TABLE
    # ============================================================

    print("\n")
    print("=" * 70)
    print("TASK 4 - OBJECTIVE EVALUATION")
    print("=" * 70)


    results_df = pd.DataFrame(
        results
    )


    # Round numerical values
    results_df["PSNR (dB)"] = results_df[
        "PSNR (dB)"
    ].round(3)

    results_df["Sharpness"] = results_df[
        "Sharpness"
    ].round(3)


    csv_file = os.path.join(
        OUTPUT_DIR,
        "TASK4_metrics.csv"
    )

    results_df.to_csv(
        csv_file,
        index=False
    )


    print("\nComplete metrics table:")
    print(
        results_df.to_string(
            index=False
        )
    )


    # ============================================================
    # FIND BEST DENOISING CONFIGURATION
    # ============================================================

    task1_results = results_df[
        results_df["Task"] == "Task 1"
    ]


    best_denoise = task1_results.loc[
        task1_results["PSNR (dB)"].idxmax()
    ]


    print("\n")
    print("-" * 70)
    print("BEST DENOISING CONFIGURATION BASED ON PSNR")
    print("-" * 70)


    print(
        best_denoise[
            "Configuration"
        ]
    )

    print(
        "PSNR:",
        best_denoise[
            "PSNR (dB)"
        ],
        "dB"
    )

    print(
        "Sharpness:",
        best_denoise[
            "Sharpness"
        ]
    )


    # ============================================================
    # FIND BEST LAPLACIAN CONFIGURATION
    # ============================================================

    task2_results = results_df[
        results_df["Task"] == "Task 2"
    ]


    best_laplacian = task2_results.loc[
        task2_results["PSNR (dB)"].idxmax()
    ]


    print("\n")
    print("-" * 70)
    print("BEST LAPLACIAN CONFIGURATION BASED ON PSNR")
    print("-" * 70)


    print(
        best_laplacian[
            "Configuration"
        ]
    )

    print(
        "PSNR:",
        best_laplacian[
            "PSNR (dB)"
        ],
        "dB"
    )

    print(
        "Sharpness:",
        best_laplacian[
            "Sharpness"
        ]
    )


    # ============================================================
    # FIND BEST HIGH-BOOST CONFIGURATION
    # ============================================================

    task3_results = results_df[
        results_df["Task"] == "Task 3"
    ]


    best_highboost = task3_results.loc[
        task3_results["PSNR (dB)"].idxmax()
    ]


    print("\n")
    print("-" * 70)
    print("BEST HIGH-BOOST CONFIGURATION BASED ON PSNR")
    print("-" * 70)


    print(
        best_highboost[
            "Configuration"
        ]
    )

    print(
        "PSNR:",
        best_highboost[
            "PSNR (dB)"
        ],
        "dB"
    )

    print(
        "Sharpness:",
        best_highboost[
            "Sharpness"
        ]
    )


    # ============================================================
    # FINAL PIPELINE
    # ============================================================

    print("\n")
    print("=" * 70)
    print("TASK 5 - FINAL PIPELINE")
    print("=" * 70)


    # We use the best denoising kernel discovered from Task 1.
    best_kernel = int(
        best_denoise["Kernel Size"]
    )


    # Denoising
    final_denoised = averaging_filter(
        sigma10_image,
        best_kernel
    )


    # Sharpen using 4-neighbor Laplacian.
    final_output, final_laplacian = laplacian_sharpen(
        final_denoised,
        laplacian4
    )


    # Metrics
    final_psnr = calculate_psnr(
        clean,
        final_output
    )

    final_sharpness = calculate_sharpness(
        final_output
    )


    save_image(
        final_denoised,
        os.path.join(
            OUTPUT_DIR,
            "FINAL_denoised.png"
        )
    )

    save_image(
        final_output,
        os.path.join(
            OUTPUT_DIR,
            "FINAL_AGV_pipeline_output.png"
        )
    )


    # Final comparison image
    create_grid(
        [
            clean,
            sigma10_image,
            final_denoised,
            final_output
        ],
        [
            "Clean Ground Truth",
            "Noisy σ=10",
            f"Denoised {best_kernel}×{best_kernel}",
            "Final Denoise → Sharpen"
        ],
        os.path.join(
            OUTPUT_DIR,
            "FINAL_pipeline_comparison.png"
        ),
        "Final Recommended AGV Preprocessing Pipeline"
    )


    # ============================================================
    # FINAL REPORT SUMMARY
    # ============================================================

    print("\n")
    print("=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)


    print(
        f"""
Recommended pipeline:

    Input AGV frame
          ↓
    Averaging filter
    Kernel = {best_kernel} × {best_kernel}
          ↓
    4-neighbor Laplacian sharpening
          ↓
    Final processed frame


Final PSNR:
    {final_psnr:.3f} dB

Final Sharpness:
    {final_sharpness:.3f}
"""
    )


    # ============================================================
    # SAVE FINAL SUMMARY
    # ============================================================

    summary_file = os.path.join(
        OUTPUT_DIR,
        "FINAL_summary.txt"
    )


    with open(
        summary_file,
        "w"
    ) as file:

        file.write(
            "AGV SPATIAL FILTERING LAB - FINAL SUMMARY\n"
        )

        file.write(
            "===========================================\n\n"
        )

        file.write(
            f"Recommended averaging kernel: "
            f"{best_kernel}x{best_kernel}\n"
        )

        file.write(
            "Recommended sharpening: "
            "4-neighbor Laplacian\n"
        )

        file.write(
            f"Final PSNR: {final_psnr:.3f} dB\n"
        )

        file.write(
            f"Final Sharpness: {final_sharpness:.3f}\n"
        )

        file.write(
            "\nPipeline:\n"
        )

        file.write(
            "Noisy Image -> Denoising -> "
            "Laplacian Sharpening -> Final Image\n"
        )


    print("\n")
    print("=" * 70)
    print("PROGRAM COMPLETED")
    print("=" * 70)

    print(
        "\nAll images, graphs and metrics are available in:"
    )

    print(
        os.path.abspath(OUTPUT_DIR)
    )

    print("\n")


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()