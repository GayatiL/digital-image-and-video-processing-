# Lab 3 – AGV Spatial Filtering and Image Restoration

## Aim

To implement and analyze **spatial filtering techniques for restoration and enhancement of an AGV (Automated Guided Vehicle) camera image**, including noise suppression using averaging filters, boundary recovery using Laplacian sharpening, and image enhancement using unsharp masking and high-boost filtering.

The practical also aims to implement 2D spatial correlation **from scratch** and evaluate the performance of different filters using **Peak Signal-to-Noise Ratio (PSNR)** and **image sharpness**.

---

# Objectives

The objectives of this practical are:

* To understand spatial filtering in digital image processing.
* To implement 2D spatial correlation from first principles.
* To avoid using predefined filtering functions such as `cv2.filter2D()` or `scipy.signal`.
* To simulate image degradation caused by Gaussian noise.
* To simulate motion blur caused by vehicle movement or vibration.
* To suppress Gaussian noise using averaging filters.
* To study the effect of different averaging filter sizes.
* To recover image boundaries using Laplacian sharpening.
* To compare 4-neighbor and 8-neighbor Laplacian operators.
* To implement unsharp masking and high-boost filtering.
* To study the effect of different high-boost parameters.
* To calculate PSNR to quantitatively evaluate image restoration.
* To calculate image sharpness using the variance of the Laplacian.
* To compare the visual and quantitative performance of different filtering techniques.

---


# Theory

## 1. Spatial Filtering

Spatial filtering is an image processing technique in which a filter or kernel is applied directly to the pixels of an image.

A small matrix called a **kernel**, **mask**, or **filter** is moved across the image.

For each position, the pixel values in the local neighborhood are combined with the corresponding kernel values to generate the output pixel.

The general spatial filtering operation can be represented as:

**g(x,y) = ΣΣ f(x+i,y+j) h(i,j)**

where:

* `f(x,y)` = input image
* `h(i,j)` = filter kernel
* `g(x,y)` = output image

Spatial filtering can be used for:

* Noise reduction
* Image smoothing
* Edge detection
* Sharpening
* Feature enhancement

---

# 2. 2D Spatial Correlation

In this practical, 2D spatial correlation is implemented **from scratch**.

The implementation does not use:

```text
cv2.filter2D()
```

or:

```text
scipy.signal
```

Instead, the correlation operation is performed using nested loops.

The image is padded with zeros around its boundary so that the filter can be applied to pixels near the edges.

The implementation first calculates the required padding:

```python id="x7m2yl"
pad_h, pad_w = k_h // 2, k_w // 2
```

The image is then padded using:

```python id="1p2j9v"
np.pad(
    image,
    ((pad_h, pad_h), (pad_w, pad_w)),
    mode='constant',
    constant_values=0
)
```

For every pixel location, a region of the image corresponding to the kernel size is extracted.

The region and kernel are multiplied element-by-element and summed:

```python id="u0q7qm"
output[i, j] = np.sum(region * kernel)
```

This demonstrates the basic working principle of spatial filtering without relying on a built-in filtering function.

---

# 3. Image Degradation

To evaluate restoration techniques, the clean image is intentionally degraded.

Two types of degradation are generated:

1. Gaussian noise
2. Motion blur

---

## 3.1 Gaussian Noise

Gaussian noise is random noise following a Gaussian or normal probability distribution.

It is represented by:

**N ~ N(0, σ²)**

where:

* Mean = `0`
* Standard deviation = `σ`

The noisy image can be represented as:

**g(x,y) = f(x,y) + n(x,y)**

where:

* `f(x,y)` = original image
* `n(x,y)` = Gaussian noise
* `g(x,y)` = noisy image

In this practical, two noise levels are generated:

```text
Sigma = 10
Sigma = 25
```

A larger sigma value represents stronger noise.

The implementation uses:

```python id="b9kq52"
noise = np.random.normal(0, sigma, image.shape)
```

The generated noise is added to the clean image.

---

# 4. Motion Blur

Motion blur occurs when an object or camera moves during image acquisition.

For an AGV, motion blur can occur because of:

* Vehicle movement
* Camera vibration
* Sudden changes in vehicle speed
* Camera exposure during motion

In this practical, horizontal motion blur is simulated using a linear kernel.

For a blur length of 9:

```text id="l3w0uc"
9 × 9 motion blur kernel
```

The center row of the kernel is assigned equal values:

**1 / 9**

This produces horizontal blurring.

The blurred image is generated using the custom spatial correlation function.

---

# 5. Averaging Filter

An averaging filter is a smoothing filter used primarily for reducing noise.

For an `N × N` averaging filter, every element of the kernel has the same value:

**h(i,j) = 1 / N²**

For example, a 3 × 3 averaging filter is:

```text
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

The filter replaces each pixel with the average of the neighboring pixels.

This reduces random noise but can also blur edges.

---

# 6. Averaging Filter Trade-Off

Different kernel sizes are tested:

```text
3 × 3
5 × 5
9 × 9
```

The filtering operation is performed on images with:

```text
Sigma = 10
Sigma = 25
```

This allows the effect of:

* Noise level
* Filter size
* Image smoothing
* Edge preservation

to be compared.

### General observation

A smaller filter generally provides moderate smoothing while preserving more details.

A larger filter provides stronger smoothing and can reduce more noise, but it can also remove fine details and reduce image sharpness.

---

# 7. Laplacian Sharpening

The Laplacian operator is a second-order derivative operator used to detect rapid changes in image intensity.

It is commonly used for:

* Edge detection
* Boundary detection
* Image sharpening

In this practical, two Laplacian kernels are used.

---

## 7.1 4-Neighbor Laplacian

The 4-neighbor Laplacian kernel is:

```text
 0   1   0
 1  -4   1
 0   1   0
```

This considers the four directly connected neighboring pixels.

The Laplacian response is calculated using the custom correlation function.

---

## 7.2 8-Neighbor Laplacian

The 8-neighbor Laplacian kernel is:

```text
 1   1   1
 1  -8   1
 1   1   1
```

This considers all eight neighboring pixels surrounding the center pixel.

The 8-neighbor operator generally provides stronger sensitivity to changes in multiple directions.

---

# 8. Laplacian Sharpening Equation

The sharpening operation implemented in this practical is:

**g(x,y) = f(x,y) - ∇²f(x,y)**

Since the selected Laplacian kernels have a negative center coefficient, subtracting the Laplacian response produces a sharpened image.

The implementation calculates:

```python id="e3kwqj"
sharp_4 = img_blurred.astype(np.float64) - resp_4

sharp_8 = img_blurred.astype(np.float64) - resp_8
```

The resulting values are clipped to the valid grayscale range:

```text
0 to 255
```

---

# 9. Unsharp Masking

Unsharp masking is an image sharpening technique.

The basic procedure is:

1. Blur the image.
2. Subtract the blurred image from the original image.
3. Obtain a high-frequency detail mask.
4. Add the mask back to the original image.

The mask is calculated as:

**mask = original - blurred**

The sharpened image can then be represented as:

**g = original + k × mask**

where `k` controls the strength of sharpening.

---

# 10. High-Boost Filtering

High-boost filtering is an extension of unsharp masking.

The practical uses the following values:

```text
k = 1
k = 1.5
k = 2
k = 3
```

The implementation calculates:

```python id="8xg45d"
boosted = original + (k * mask)
```

### Effect of k

* `k = 1` → standard unsharp masking behavior.
* `k > 1` → stronger enhancement of high-frequency details.
* Larger `k` → stronger sharpening.
* Excessively large `k` may amplify noise and introduce unwanted artifacts.

This experiment demonstrates the trade-off between sharpness and image quality.

---

# 11. Peak Signal-to-Noise Ratio (PSNR)

PSNR is used to quantitatively measure the quality of a processed image compared with a known original image.

It is calculated using the Mean Squared Error (MSE).

### Mean Squared Error

**MSE = (1/N) Σ (Original - Processed)²**

The PSNR is:

**PSNR = 20 × log₁₀(MAX / √MSE)**

For an 8-bit grayscale image:

**MAX = 255**

Therefore:

**PSNR = 20 × log₁₀(255 / √MSE)**

### Interpretation

A higher PSNR generally indicates that the processed image is closer to the original image in terms of pixel-wise error.

A lower PSNR indicates greater difference between the processed and original images.

If the MSE is zero, the PSNR becomes infinite because the two images are identical.

---

# 12. Image Sharpness

In addition to PSNR, the practical measures image sharpness using the:

**Variance of the Laplacian**

The Laplacian response is calculated using:

```text
0   1   0
1  -4   1
0   1   0
```

The variance of this response is then calculated.

A higher variance generally indicates stronger intensity changes and more pronounced edges.

The sharpness metric is useful for comparing the amount of edge detail preserved or enhanced by different filtering techniques.

---

# Software and Libraries Used

The following technologies and libraries are used:

* **Python**
* **OpenCV**
* **NumPy**
* **Matplotlib**

### Libraries

```python id="v0z6fz"
import numpy as np
import cv2
import matplotlib.pyplot as plt
```

---



# Algorithm / Procedure

1. Start the program.
2. Generate a clean synthetic AGV scene.
3. Treat the clean image as the ground-truth image.
4. Generate Gaussian noise with `σ = 10`.
5. Generate Gaussian noise with `σ = 25`.
6. Generate horizontal motion blur using a 9-pixel blur kernel.
7. Implement 2D spatial correlation from scratch.
8. Apply 3 × 3, 5 × 5, and 9 × 9 averaging filters to both noisy images.
9. Calculate PSNR and sharpness for each filtered image.
10. Apply the 4-neighbor Laplacian operator to the motion-blurred image.
11. Apply the 8-neighbor Laplacian operator to the motion-blurred image.
12. Generate sharpened images using Laplacian sharpening.
13. Calculate PSNR and sharpness for both sharpened images.
14. Blur the Sigma 10 noisy image using a 5 × 5 averaging filter.
15. Generate the high-frequency mask.
16. Apply high-boost filtering using `k = 1, 1.5, 2, 3`.
17. Calculate PSNR and sharpness for every high-boost result.
18. Display the degraded images.
19. Display the averaging filter results.
20. Display the Laplacian sharpening results.
21. Display the high-boost filtering results.
22. Compare the visual and quantitative results.
23. End the program.

---

# Results and Observations

## Averaging Filter

The averaging filter reduces Gaussian noise by smoothing neighboring pixel values.

The effect depends on both the noise level and filter size.

* The 3 × 3 filter provides moderate smoothing while retaining more image details.
* The 5 × 5 filter provides stronger smoothing.
* The 9 × 9 filter provides the strongest smoothing but can cause greater loss of fine details and edge sharpness.
* Higher noise levels, such as `σ = 25`, require stronger smoothing compared with `σ = 10`.

Therefore, there is a trade-off between **noise reduction and image sharpness**.

---

## Laplacian Sharpening

Laplacian sharpening improves boundaries and edges in the motion-blurred image.

The 4-neighbor and 8-neighbor operators produce different sharpening responses because they consider different neighborhoods.

* The 4-neighbor operator considers horizontal and vertical neighbors.
* The 8-neighbor operator considers all eight surrounding pixels.
* Sharpening increases edge strength, which is reflected in the sharpness metric.

---

## High-Boost Filtering

High-boost filtering enhances high-frequency image components.

As the value of `k` increases:

```text
1 → 1.5 → 2 → 3
```

the sharpening effect becomes stronger.

However, excessive sharpening can also amplify unwanted noise and create artifacts.

Therefore, an appropriate value of `k` must be selected based on the required balance between detail enhancement and image quality.

---

# Implementation

## Step 1 – Create Ground Truth Image

A synthetic AGV scene is generated with dimensions:

```text
200 × 200
```

This image represents the ideal image captured under clean conditions.

---

## Step 2 – Generate Degraded Images

Three degraded versions are generated.

### Gaussian Noise – Sigma 10

```python id="g7asqf"
img_noisy_10 = add_gaussian_noise(clean_img, sigma=10)
```

### Gaussian Noise – Sigma 25

```python id="1n4y5e"
img_noisy_25 = add_gaussian_noise(clean_img, sigma=25)
```

### Motion Blur

```python id="m5s8g7"
img_blurred = add_motion_blur(clean_img, length=9)
```

These images represent different possible degradation conditions experienced by an AGV camera.

---

# Task 1 – Noise Suppression Using Averaging Filters

Three averaging kernel sizes are tested:

```text
3 × 3
5 × 5
9 × 9
```

Each filter is applied to both noisy images:

* Sigma 10
* Sigma 25

The averaging kernel is generated as:

```python id="w4kkjj"
box_kernel = np.ones((size, size)) / (size * size)
```

The custom correlation function is then used to apply the filter.

The output is clipped to the valid intensity range:

```text
0 to 255
```

For every result, both PSNR and sharpness are calculated.

---

# Task 2 – Boundary Recovery Using Laplacian Sharpening

The motion-blurred image is processed using two Laplacian operators.

### 4-Neighbor Laplacian

```text
0   1   0
1  -4   1
0   1   0
```

### 8-Neighbor Laplacian

```text
1   1   1
1  -8   1
1   1   1
```

The Laplacian response is calculated for the motion-blurred image.

The sharpening equation is then applied:

**Sharpened Image = Blurred Image − Laplacian Response**

Both results are evaluated using PSNR and sharpness.

---

# Task 3 – Unsharp Masking / High-Boost Filtering

A 5 × 5 averaging filter is first used to blur the noisy image.

The blur kernel is:

```python id="7cqgrf"
blur_kernel = np.ones((5, 5)) / 25.0
```

The high-frequency mask is calculated as:

**Mask = Original − Blurred**

The mask is then multiplied by different values of `k`.

The following values are tested:

```text
k = 1
k = 1.5
k = 2
k = 3
```

The final enhanced image is calculated using:

**g = Original + k × Mask**

For every value of `k`, PSNR and sharpness are calculated.

---

# Task 4 – Quantitative Evaluation

The results of the filtering techniques are evaluated using two metrics:

### 1. PSNR

Measures similarity between the processed image and the clean ground-truth image.

### 2. Sharpness

Calculated using the variance of the Laplacian and used to measure edge strength.

The program prints results in the following form:

```text
Sigma 10, Kernel 3x3 -> PSNR: ... | Sharpness: ...
Sigma 25, Kernel 3x3 -> PSNR: ... | Sharpness: ...
```

Similar measurements are produced for:

* 5 × 5 averaging
* 9 × 9 averaging
* 4-neighbor Laplacian
* 8-neighbor Laplacian
* High-boost `k = 1`
* High-boost `k = 1.5`
* High-boost `k = 2`
* High-boost `k = 3`

---

# Result

The spatial filtering techniques were successfully implemented on the degraded AGV camera images. Gaussian noise was reduced using averaging filters, motion blur was processed using Laplacian sharpening, and image details were enhanced using unsharp masking and high-boost filtering. PSNR and sharpness values were calculated for the processed images.

# Observations

- Averaging filters reduce Gaussian noise, but larger kernels also smooth out more image details.
- The 3 × 3 filter provides less smoothing while preserving more edges compared with larger filters.
- The 9 × 9 filter gives stronger smoothing but can reduce image sharpness.
- Both 4-neighbor and 8-neighbor Laplacian operators enhance image boundaries, with the 8-neighbor operator considering a wider neighborhood.
- Increasing the high-boost value `k` produces stronger sharpening.
- Very high sharpening strength can also emphasize unwanted noise and artifacts.
- PSNR and variance of the Laplacian provide quantitative measures for comparing the processed images.

# Output

The practical produces the following major outputs:

### Degraded Images

* Clean Ground Truth
* Gaussian Noise with `σ = 10`
* Gaussian Noise with `σ = 25`
* Motion Blurred Image

### Averaging Filter Results

* Sigma 10 with 3 × 3 filter
* Sigma 10 with 5 × 5 filter
* Sigma 10 with 9 × 9 filter
* Sigma 25 with 3 × 3 filter
* Sigma 25 with 5 × 5 filter
* Sigma 25 with 9 × 9 filter

### Laplacian Sharpening Results

* 4-neighbor sharpened image
* 8-neighbor sharpened image

### High-Boost Results

* `k = 1`
* `k = 1.5`
* `k = 2`
* `k = 3`

### Quantitative Results

PSNR and sharpness values are calculated for every processed image.

---

# Conclusion

In this practical, spatial filtering techniques were studied and implemented for the restoration and enhancement of an **AGV camera image**.

A 2D spatial correlation function was implemented from first principles without using predefined filtering functions such as `cv2.filter2D()` or `scipy.signal`. This provided a clear understanding of how spatial filters operate at the pixel level.

Gaussian noise with two different noise levels and motion blur were introduced to simulate realistic degradation that may occur in an AGV camera system. Averaging filters of sizes **3 × 3, 5 × 5, and 9 × 9** were then applied to suppress Gaussian noise. The experiment demonstrated the trade-off between noise reduction and preservation of image details.

Laplacian sharpening using both **4-neighbor and 8-neighbor operators** was applied to the motion-blurred image to improve boundaries and edge information. Unsharp masking and high-boost filtering were also implemented using different values of `k` to study the effect of controlled sharpening.

Finally, **PSNR** and **variance of the Laplacian** were used as quantitative measures to evaluate image quality and sharpness.

Thus, the practical demonstrated how spatial filtering can be used for **noise suppression, boundary recovery, sharpening, and image restoration**, which are important preprocessing operations for AGV vision systems and other computer vision applications.