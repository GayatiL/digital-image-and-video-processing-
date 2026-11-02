# DIVP Experiment – First- and Second-Order Derivative Filtering

## Aim

To study the response of first- and second-order derivative spatial filters on synthetic grayscale images containing thin diagonal lines and wider vertical lines. The experiment implements 2D spatial filtering manually using NumPy and compares the output produced by four different 3 × 3 kernels.

## Objectives

- Understand the role of spatial derivative filters in Digital Image Processing.
- Implement 2D kernel filtering using nested loops.
- Study a Laplacian (second-derivative) kernel.
- Study directional first-derivative kernels.
- Examine the response of a diagonal second-derivative kernel.
- Compare filter responses for a 1-pixel-wide line image and a 3-pixel-wide line image.
- Observe how kernel direction and line thickness affect the filtered output.

## Theory

### 1. Spatial Filtering

Spatial filtering operates directly on image pixels. A small matrix, called a **kernel** or **mask**, is moved over the image. At each pixel location, the image neighborhood is multiplied element-by-element by the kernel, and the products are added to obtain the output pixel.

For an image \(f(x,y)\) and kernel \(h(i,j)\), the operation used in this experiment is:

\[
g(x,y)=\sum_i\sum_j f(x+i,y+j)h(i,j)
\]

Here:

- \(f(x,y)\): input image
- \(h(i,j)\): filter kernel
- \(g(x,y)\): filtered output

The code applies the kernel directly without flipping it. Strictly speaking, this is **2D spatial correlation** rather than mathematical convolution. For the symmetric kernels, correlation and convolution give the same result; for directional kernels, flipping the kernel would reverse the response polarity.

### 2. First-Order Derivative Filters

First-order derivatives measure the rate of change of image intensity. They respond strongly where intensity changes in a particular direction and are commonly used for edge detection.

This experiment uses two directional kernels:

**Kernel B**

```text
-1  -1  -1
 2   2   2
-1  -1  -1
```

This kernel is antisymmetric in the vertical direction and smooths across the horizontal neighborhood. It responds to intensity changes from one row to another.

**Kernel C**

```text
-1   2  -1
-1   2  -1
-1   2  -1
```

This kernel is antisymmetric in the horizontal direction and smooths across the vertical neighborhood. It responds to intensity changes from one column to another.

The signs of the output indicate the polarity of the intensity transition. The displayed grayscale image may show both positive and negative responses, depending on Matplotlib's automatic display scaling.

### 3. Second-Order Derivative Filters

Second-order derivatives measure how quickly the intensity gradient changes. They are useful for highlighting rapid intensity transitions and line boundaries.

**Kernel A – 8-neighbour Laplacian**

```text
 1   1   1
 1  -8   1
 1   1   1
```

This is an 8-neighbour Laplacian kernel. It compares the centre pixel with all eight surrounding pixels and responds to local intensity changes in multiple directions.

**Kernel D – diagonal second-order kernel**

```text
 2  -1  -1
-1   2  -1
-1  -1   2
```

This kernel has positive coefficients along its main diagonal and negative coefficients at the other positions. It is used here to examine a diagonal-oriented second-order response. It is not the standard 4-neighbour or 8-neighbour Laplacian kernel.

## Kernels Used

| Kernel | Type / description | Matrix |
|---|---|---|
| A | 8-neighbour Laplacian, second derivative | `[[1,1,1],[1,-8,1],[1,1,1]]` |
| B | Directional first derivative (vertical intensity change) | `[[-1,-1,-1],[2,2,2],[-1,-1,-1]]` |
| C | Directional first derivative (horizontal intensity change) | `[[-1,2,-1],[-1,2,-1],[-1,2,-1]]` |
| D | Diagonal-oriented second-order kernel | `[[2,-1,-1],[-1,2,-1],[-1,-1,2]]` |

## Implementation

### 1. Importing Libraries

The program uses:

```python
import numpy as np
import matplotlib.pyplot as plt
```

- **NumPy** is used to create images, define kernels, pad arrays, and perform element-wise multiplication and summation.
- **Matplotlib** is used to display the original images and filtered outputs.

### 2. Implementing the Filtering Function

The `apply_kernel()` function performs filtering without using a predefined filtering function such as `cv2.filter2D()` or `scipy.signal.convolve2d()`.

Main steps:

1. Read the image and kernel dimensions.
2. Calculate the padding required around the image.
3. Add zero-padding to the image.
4. Move the kernel over every pixel location using nested loops.
5. Extract the local image region.
6. Multiply the region by the kernel element-by-element.
7. Sum the products and store the result in the output image.

The central operation is:

```python
output[i, j] = np.sum(region * kernel)
```

Zero-padding is used so that the output has the same dimensions as the input image. Pixels near the image boundary are therefore filtered using neighbourhoods that include padded zero values.

### 3. Creating the 1-Pixel-Wide Test Image

A 100 × 100 black image is created. Three separate diagonal line segments are drawn by assigning an intensity of 255 to selected pixels:

- First segment: `image_1pixel[i, i]`
- Second segment: `image_1pixel[i, i+15]`
- Third segment: `image_1pixel[i, i-10]`

The line segments are one pixel wide and have a white intensity of 255 against a black background of intensity 0.

### 4. Creating the 3-Pixel-Wide Test Image

A second 100 × 100 black image is created. Three vertical white strips, each three pixels wide, are drawn:

```python
image_3pixel[20:80, 25:28] = 255
image_3pixel[20:80, 50:53] = 255
image_3pixel[20:80, 75:78] = 255
```

This image is used to observe how the same filters respond to wider lines.

### 5. Applying the Kernels

All four kernels (A, B, C and D) are applied independently to both test images. The program therefore generates eight filtered outputs:

- Four outputs for the 1-pixel-wide diagonal-line image.
- Four outputs for the 3-pixel-wide vertical-line image.

The original image and the four filtered outputs are displayed using `plt.imshow(..., cmap='gray')`.

## Algorithm / Procedure

1. Start the program.
2. Import NumPy and Matplotlib.
3. Define the four 3 × 3 filter kernels.
4. Define the `apply_kernel()` function.
5. Create a 100 × 100 black image.
6. Draw three 1-pixel-wide diagonal white line segments.
7. Create another 100 × 100 black image.
8. Draw three 3-pixel-wide vertical white strips.
9. Apply Kernel A to both images.
10. Apply Kernel B to both images.
11. Apply Kernel C to both images.
12. Apply Kernel D to both images.
13. Display the original 1-pixel image and its four filtered outputs.
14. Display the original 3-pixel image and its four filtered outputs.
15. Compare the filter responses and observe the effect of line direction and thickness.
16. End the program.

## Results and Observations

### 1-Pixel-Wide Diagonal-Line Image

The original image contains three thin diagonal white line segments on a black background.

- **Kernel A (8-neighbour Laplacian):** produces a second-order response around the line pixels and their boundaries. Positive and negative responses appear around intensity transitions.
- **Kernel B (directional first derivative):** responds to intensity changes in the vertical direction. Its response depends on the orientation of the diagonal line segments.
- **Kernel C (directional first derivative):** responds to intensity changes in the horizontal direction. Its response differs from Kernel B because the direction of differentiation is different.
- **Kernel D (diagonal-oriented second-order kernel):** produces a second-order response influenced by the diagonal arrangement of its coefficients.

### 3-Pixel-Wide Vertical-Line Image

The second image contains three vertical strips, each three pixels wide. Applying the same kernels makes it possible to compare their responses with those from the thin diagonal lines.

The wider strips contain larger uniform-intensity regions than the 1-pixel lines. Derivative filters primarily respond at intensity transitions, so the output is influenced by the strip boundaries and by the direction represented by each kernel.

### Display Note

Matplotlib automatically scales each displayed result to its own minimum and maximum when `vmin` and `vmax` are not specified. Therefore, the grey background and apparent contrast can differ between output panels. The displayed brightness should not be interpreted as a direct comparison of absolute filter-response magnitudes across panels.

## Output

The program displays two result figures:

**Figure 1 – 1-Pixel-Wide Diagonal Lines**

- Original 1-pixel image
- Kernel A – second derivative
- Kernel B – first derivative
- Kernel C – first derivative
- Kernel D – second derivative

**Figure 2 – 3-Pixel-Wide Vertical Lines**

- Original 3-pixel-wide image
- Kernel A – second derivative
- Kernel B – first derivative
- Kernel C – first derivative
- Kernel D – second derivative

If you save the figures from the program, you can add them to the repository and embed them here. For example:

```markdown
![1-pixel line filtering results](Figure_1.png)
![3-pixel line filtering results](Figure_2.png)
```

## Software and Libraries Used

- Python 3
- NumPy
- Matplotlib

Install the required libraries with:

```bash
pip install numpy matplotlib
```

## How to Run

1. Clone or download this repository.
2. Make sure Python 3 is installed.
3. Install the required libraries.
4. Run the Python experiment file:

```bash
python your_experiment_filename.py
```

The program opens two Matplotlib figures showing the original images and the corresponding filtered outputs.

## Conclusion

In this experiment, first- and second-order derivative spatial filters were implemented and applied to synthetic images containing 1-pixel-wide diagonal lines and 3-pixel-wide vertical strips. A custom 2D filtering function was developed using zero-padding and nested loops, without relying on built-in image filtering functions.

The four kernels produced different responses because they represent different derivative directions and neighbourhood weightings. Comparing the two test images demonstrates that filter output depends on both the orientation and thickness of image structures. The experiment provides a practical understanding of how spatial derivative kernels respond to intensity transitions and how they can be used for edge and line analysis in Digital Image Processing.
