# Lab 5 – Marr-Hildreth Edge Detection

## Aim

To implement the **Marr-Hildreth edge detection algorithm** on a grayscale house image using Gaussian smoothing, the Laplacian operator, and zero-crossing detection. The experiment studies how different Gaussian standard deviation (`sigma`) values and zero-crossing thresholds affect the detected edges.

## Objectives

- Understand the Marr-Hildreth edge detection method.
- Load and convert an input image to grayscale.
- Normalize image intensity values.
- Reduce noise and fine details using Gaussian smoothing.
- Apply the Laplacian operator to the smoothed image.
- Calculate the magnitude of the Laplacian response.
- Detect zero crossings in horizontal, vertical, and diagonal directions.
- Compare zero-crossing detection with no threshold and with a threshold of 4% of the maximum Laplacian response.
- Study the effect of different Gaussian `sigma` and kernel-size values.

## Theory

### 1. Marr-Hildreth Edge Detection

Marr-Hildreth is an edge detection technique based on the combination of **Gaussian smoothing** and the **Laplacian operator**. It is also commonly described as the Laplacian of Gaussian (LoG) approach.

The main idea is to smooth the image first to reduce noise, then identify locations where the second derivative changes sign. These sign changes are called **zero crossings** and are used to locate potential edges.

The main stages in this experiment are:

1. Convert the input image to grayscale and normalize it.
2. Apply Gaussian smoothing.
3. Apply the Laplacian operator to the smoothed image.
4. Find zero crossings in the Laplacian response.
5. Apply a threshold to reduce weak zero-crossing responses.

### 2. Gaussian Smoothing

Gaussian smoothing reduces noise and small intensity variations before edge detection. It uses a Gaussian-shaped kernel to calculate a weighted average of nearby pixels.

The 2D Gaussian function is:

\[
G(x,y)=\frac{1}{2\pi\sigma^2}
\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
\]

where `sigma` controls the amount of smoothing. A larger `sigma` produces stronger smoothing and removes more fine detail.

The program uses OpenCV's `cv2.GaussianBlur()` function. The tested parameter pairs are:

| Sigma | Gaussian kernel size |
|---:|---:|
| 4 | 25 × 25 |
| 7 | 35 × 35 |
| 9 | 45 × 45 |

### 3. Laplacian Operator

The Laplacian is a second-order derivative operator. It measures changes in the image gradient and produces strong responses around rapid intensity transitions.

For a continuous image \(f(x,y)\), the Laplacian is:

\[
\nabla^2 f =
\frac{\partial^2 f}{\partial x^2}
+
\frac{\partial^2 f}{\partial y^2}
\]

In this experiment, the Laplacian is applied to the Gaussian-smoothed image using:

```python
laplacian = cv2.Laplacian(
    smoothed,
    cv2.CV_64F,
    ksize=3
)
```

`cv2.CV_64F` is used so that both positive and negative derivative responses can be represented.

The magnitude image is calculated as:

```python
laplacian_magnitude = np.abs(laplacian)
```

This displays the strength of the Laplacian response without showing its sign.

### 4. Zero-Crossing Detection

A zero crossing occurs when two neighboring Laplacian values have opposite signs, or when one value is zero and the other has the opposite sign.

For two neighboring values `A` and `B`, a sign change is detected when:

```python
((A < 0) & (B >= 0)) | ((A >= 0) & (B < 0))
```

The program checks four directions:

- Horizontal: `(0, 1)`
- Vertical: `(1, 0)`
- Diagonal: `(1, 1)`
- Opposite diagonal: `(1, -1)`

When a sign change is detected and the response difference satisfies the threshold condition, both pixels are marked as edge pixels with intensity 255.

### 5. Zero-Crossing Threshold

A threshold is used to reject weak zero crossings that may be caused by noise or small intensity variations.

**Case 1 – T = 0**

No minimum response-difference threshold is applied. Every detected sign change is accepted.

**Case 2 – T = 4% of maximum Laplacian response**

The threshold is calculated as:

\[
T=0.04\times\max(|L|)
\]

where `L` is the Laplacian response.

Only sign changes for which the absolute difference between neighboring Laplacian values is greater than this threshold are accepted:

\[
|A-B|>T
\]

This thresholding reduces weak edge responses and helps suppress unwanted detections.

## Implementation

### 1. Libraries Used

```python
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
```

- **OpenCV (`cv2`)**: reads the image, performs Gaussian smoothing, and calculates the Laplacian.
- **NumPy**: handles arrays, intensity normalization, sign comparisons, and threshold calculations.
- **Matplotlib**: creates and saves the result figures.
- **os**: constructs file paths and creates the output directory.

The Matplotlib `Agg` backend is selected so that figures can be saved without requiring a graphical display.

### 2. Loading the Input Image

The program reads `house.png` from the same folder as the Python script:

```python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "house.png")

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
```

The image is loaded in grayscale because the edge detection operation is performed on intensity values rather than color channels.

The image is converted to floating-point values and normalized to the range 0 to 1:

```python
img = img.astype(np.float64) / 255.0
```

The program checks whether the image file exists and whether OpenCV successfully loads it.

### 3. Creating the Output Directory

An `output` folder is created automatically inside the Lab 5 folder:

```python
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

The generated figures are saved in this folder.

### 4. Finding Zero Crossings

The function `find_zero_crossings(L, threshold)`:

1. Creates a blank output edge image.
2. Checks neighboring pixels in four directions.
3. Detects sign changes in the Laplacian response.
4. Applies the specified threshold to the response difference.
5. Marks both pixels around each accepted zero crossing as white (255).
6. Returns the binary edge image.

The function is implemented using NumPy array operations and loops over the four specified directions.

### 5. Processing Different Sigma Values

The program processes these parameter combinations:

```python
parameters = [
    (4, 25),
    (7, 35),
    (9, 45)
]
```

For each pair, it performs Gaussian smoothing, Laplacian filtering, zero-crossing detection with `T = 0`, and zero-crossing detection with `T = 4%` of the maximum absolute Laplacian response.

### 6. Creating and Saving the Result Figure

For each sigma value, the program creates a 2 × 3 figure containing:

| Panel | Description |
|---|---|
| (a) | Original grayscale image |
| (b) | Gaussian-smoothed image |
| (c) | Absolute Laplacian magnitude |
| (d) | Signed Laplacian response |
| (e) | Zero crossings with `T = 0` |
| (f) | Zero crossings with `T = 4%` of maximum response |

The figure is saved using:

```python
output_path = os.path.join(
    OUTPUT_DIR,
    f"Marr_Hildreth_Sigma_{sigma}.png"
)

fig.savefig(output_path, dpi=300, bbox_inches="tight")
```

The program closes each figure after saving it to release memory.

## Algorithm / Procedure

1. Start the program.
2. Import OpenCV, NumPy, Matplotlib, and `os`.
3. Set the Matplotlib backend to `Agg`.
4. Load `house.png` in grayscale.
5. Check that the input image was loaded successfully.
6. Convert the image to `float64` and normalize its intensity values to the range 0 to 1.
7. Create the output directory if it does not already exist.
8. Define the zero-crossing detection function.
9. Define the parameter pairs `(4, 25)`, `(7, 35)`, and `(9, 45)`.
10. For each parameter pair, apply Gaussian smoothing.
11. Apply the 3 × 3 Laplacian operator to the smoothed image.
12. Calculate the absolute Laplacian magnitude.
13. Detect zero crossings using threshold `T = 0`.
14. Calculate the maximum absolute Laplacian response.
15. Calculate the threshold as 4% of that maximum response.
16. Detect zero crossings using the calculated threshold.
17. Display the six processing stages in a 2 × 3 figure.
18. Save the figure in the output folder.
19. Repeat the process for all three sigma values.
20. End the program after all result figures have been generated.

## Results and Observations

### 1. Effect of Gaussian Smoothing

Gaussian smoothing reduces small intensity variations and noise before the Laplacian operation.

- A smaller sigma preserves more fine details but may also retain more noise.
- A larger sigma produces stronger smoothing.
- Stronger smoothing can suppress unwanted small edge responses, but it can also remove fine details and cause nearby edges to merge.

### 2. Laplacian Response

The Laplacian magnitude highlights areas with strong second-order intensity changes.

The signed Laplacian contains positive and negative values. These sign changes are important because zero-crossing detection uses them to locate potential edges.

### 3. Zero Crossings with T = 0

When the threshold is zero, all detected sign changes are accepted. This can produce many edge responses, including responses caused by small image variations and noise.

### 4. Zero Crossings with T = 4% of Maximum

Applying the 4% threshold rejects sign changes whose neighboring Laplacian values differ only weakly.

Compared with `T = 0`, the thresholded result generally contains fewer weak responses and emphasizes stronger edges. However, a threshold that is too high can also remove genuine weak edges.

### 5. Comparison of Sigma Values

The program generates results for sigma values 4, 7, and 9.

As sigma increases, Gaussian smoothing becomes stronger. This changes the amount of fine detail retained in the smoothed image and consequently changes the Laplacian and zero-crossing results.

The actual output images should be compared to determine which sigma gives the most useful balance between noise suppression and preservation of important house boundaries.

## Output

The program generates three figures automatically:

```text
output/
├── Marr_Hildreth_Sigma_4.png
├── Marr_Hildreth_Sigma_7.png
└── Marr_Hildreth_Sigma_9.png
```

Each figure contains six panels:

- Original image
- Gaussian-smoothed image
- Laplacian magnitude
- Signed Laplacian response
- Zero crossings with `T = 0`
- Zero crossings with `T = 4%` of maximum Laplacian response

The supplied output image shows the result for **sigma = 4 and kernel size = 25 × 25**.

## Software and Libraries Used

- Python 3
- OpenCV
- NumPy
- Matplotlib

Install the required libraries using:

```bash
pip install opencv-python numpy matplotlib
```

## Folder Structure

Keep the Python script and input image in the same Lab 5 folder:

```text
Lab5/
├── README.md
├── edge_detection.py
├── house.png
└── output/
    ├── Marr_Hildreth_Sigma_4.png
    ├── Marr_Hildreth_Sigma_7.png
    └── Marr_Hildreth_Sigma_9.png
```

The `output` folder is created automatically when the program runs.

## How to Run

1. Make sure Python 3 is installed.
2. Open a terminal in the Lab 5 folder.
3. Install the required libraries:

```bash
pip install opencv-python numpy matplotlib
```

4. Run the Python file:

```bash
python edge_detection.py
```

5. After execution, check the `output` folder for the three generated figures.

## Conclusion

In this experiment, the **Marr-Hildreth edge detection method** was implemented on a grayscale house image. The method combines Gaussian smoothing with the Laplacian operator and identifies edges using zero crossings in the signed Laplacian response.

Three Gaussian smoothing settings were tested: sigma values of 4, 7, and 9, with corresponding kernel sizes of 25 × 25, 35 × 35, and 45 × 45. For each setting, zero crossings were detected both without a threshold and with a threshold equal to 4% of the maximum absolute Laplacian response.

The results demonstrate that Gaussian smoothing affects the amount of detail and noise present before edge detection, while thresholding controls which zero-crossing responses are retained. Thus, the experiment illustrates how smoothing strength and zero-crossing threshold influence edge detection results.
