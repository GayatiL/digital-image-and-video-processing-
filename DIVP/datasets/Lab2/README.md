# Lab 1 – Image Intensity Transformations

## Aim

To study and implement basic image intensity transformation techniques, namely **Negative Transformation, Power-Law (Gamma) Transformation, and Logarithmic Transformation**, using Python and OpenCV.

---

## Objectives

The objectives of this practical are:

- To understand the concept of image intensity transformations.
- To implement negative transformation for a grayscale image.
- To implement Power-Law (Gamma) transformation.
- To implement logarithmic transformation.
- To understand how transformation functions modify pixel intensity values.
- To implement the transformations using pixel matrices.
- To implement the same transformations using Python libraries such as OpenCV and NumPy.

---

## Theory

Image intensity transformation is a basic technique in **Digital Image Processing** used to modify pixel intensity values to improve an image or highlight particular details.

An intensity transformation maps an input pixel intensity `r` to an output intensity `s`.

**s = T(r)**

where:

- `r` = input pixel intensity
- `s` = output pixel intensity
- `T` = transformation function

For an 8-bit grayscale image, pixel intensity values range from **0 to 255**. A value of `0` represents black, while `255` represents white.

Three intensity transformations are implemented in this practical.

---

## 1. Negative Transformation

Negative transformation reverses the intensity values of an image.

The transformation is given by:

**s = (L - 1) - r**

For an 8-bit grayscale image:

**s = 255 - r**

where `L = 256` is the number of possible intensity levels.

For example:

| Original Pixel | Negative Pixel |
|---:|---:|
| 0 | 255 |
| 50 | 205 |
| 100 | 155 |
| 150 | 105 |
| 255 | 0 |

This transformation converts dark regions into bright regions and bright regions into dark regions.

---

## 2. Power-Law (Gamma) Transformation

Power-Law transformation, also called **Gamma transformation**, is represented by:

**s = c × r^γ**

The input pixel values are first normalized to the range `[0, 1]`:

```python
r_norm = r / 255
```

The transformation is then applied:

```python
s_norm = c * (r_norm ** gamma)
```

The result is finally scaled back to the range `[0, 255]`.

### Effect of Gamma

The Gamma value controls the brightness of the output image:

- **γ < 1** → image becomes brighter.
- **γ = 1** → approximately preserves the original intensity.
- **γ > 1** → image becomes darker.

In this practical, a Gamma value of:

```text
γ = 0.5
```

is used.

A Lookup Table (LUT) can be used with OpenCV to efficiently apply the transformation to all possible 8-bit intensity values.

---

## 3. Logarithmic Transformation

The logarithmic transformation is represented by:

**s = c × log(1 + r)**

where:

- `r` = input pixel intensity
- `s` = output pixel intensity
- `c` = scaling constant

The scaling constant is calculated as:

**c = 255 / log(1 + r_max)**

where `r_max` is the maximum pixel intensity in the image.

Logarithmic transformation expands lower intensity values and compresses higher intensity values. It can therefore be useful for enhancing details present in darker regions.

---

# Software and Libraries Used

The following software and Python libraries are used:

- **Python**
- **OpenCV (`cv2`)**
- **NumPy**
- **Matplotlib**

### Python Libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

---

# Implementation

The practical is implemented in two stages:

1. **Manual implementation using a pixel matrix**
2. **Image processing implementation using OpenCV and NumPy**

---

## Part A – Manual Matrix Implementation

A user-defined image matrix is taken as input.

The program accepts:

- Number of rows
- Number of columns
- Pixel values between 0 and 255
- Gamma value for Power-Law transformation

The input matrix is then processed using the three transformation functions.

### 1. Manual Natural Logarithm

A custom `manual_ln()` function is implemented using the Taylor series.

The formula used is:

**ln(x) = 2 × [z + z³/3 + z⁵/5 + ...]**

where:

**z = (x - 1) / (x + 1)**

This allows the natural logarithm to be calculated without directly using Python's built-in logarithm function.

### 2. Negative Transformation

The negative transformation is implemented using:

```python
new_pixel = max_intensity - pixel
```

For an 8-bit image:

```text
new_pixel = 255 - pixel
```

The transformation is applied to every pixel of the input matrix.

### 3. Power-Law Transformation

The input pixel is first normalized:

```python
r_norm = pixel / max_intensity
```

The Power-Law transformation is then applied:

```python
s_norm = c * (r_norm ** gamma)
```

The result is converted back to the intensity range `[0, 255]`.

### 4. Logarithmic Transformation

A scaling constant is calculated using the maximum intensity value:

```python
c = max_intensity / manual_ln(1 + max_intensity)
```

The logarithmic transformation is then applied:

```python
s = c * manual_ln(1 + pixel)
```

The resulting values are restricted to the valid range of 0 to 255.

---

# Part B – Image Processing Using OpenCV

A grayscale image is loaded using OpenCV:

```python
img = cv2.imread("Lab1/image.jpg", cv2.IMREAD_GRAYSCALE)
```

The three transformations are then applied to the image.

---

## Negative Transformation Using OpenCV

The negative image is generated using:

```python
neg_img = cv2.bitwise_not(img)
```

This reverses the intensity of every pixel.

---

## Power-Law Transformation Using OpenCV

A Lookup Table containing the transformed values for all 256 possible grayscale intensities is generated.

The Gamma value used is:

```python
gamma = 0.5
```

The lookup table is created using:

```python
lut = np.array(
    [((i / 255.0) ** 0.5) * 255 for i in np.arange(0, 256)]
).astype("uint8")
```

The LUT is then applied using:

```python
gamma_img = cv2.LUT(img, lut)
```

Using a lookup table makes the transformation more efficient because the calculation does not have to be repeated independently for every pixel.

---

## Logarithmic Transformation Using OpenCV and NumPy

The maximum pixel value in the image is obtained using:

```python
max_pixel_val = float(np.max(img))
```

The scaling constant is calculated as:

```python
c = 255 / np.log(1 + max_pixel_val)
```

The logarithmic transformation is applied using NumPy:

```python
log_img = np.array(
    c * np.log1p(img.astype(np.float32)),
    dtype=np.uint8
)
```

`np.log1p()` is used to calculate `log(1 + x)` accurately.

---

# Algorithm / Procedure

1. Start the program.
2. Define the input grayscale image or pixel matrix.
3. Accept the required matrix dimensions and pixel values for the manual implementation.
4. Accept a Gamma value for the Power-Law transformation.
5. Apply Negative Transformation.
6. Apply Power-Law (Gamma) Transformation.
7. Apply Logarithmic Transformation.
8. Display the transformed pixel matrices.
9. Load the grayscale image using OpenCV.
10. Apply Negative Transformation using `cv2.bitwise_not()`.
11. Create a Lookup Table for Power-Law transformation.
12. Apply the LUT using `cv2.LUT()`.
13. Calculate the scaling constant for logarithmic transformation.
14. Apply the logarithmic transformation using NumPy.
15. Store/display the transformed images as required.
16. End the program.

---

# Input

The input for the practical consists of:

- A grayscale image stored as `image.jpg`.
- Pixel intensity values ranging from **0 to 255**.
- A user-defined Gamma value for the manual Power-Law transformation.

For the OpenCV implementation, the image is loaded from:

```text
Lab1/image.jpg
```

---

# Result

The **Negative, Power-Law (Gamma), and Logarithmic intensity transformations** were successfully implemented using both manual pixel-matrix operations and Python image-processing libraries.

The Negative Transformation successfully reversed the intensity values. The Power-Law transformation with `γ = 0.5` produced a brighter image, while the Logarithmic Transformation enhanced lower intensity values by expanding darker regions and compressing higher intensity values.

Thus, the practical successfully demonstrated the implementation of fundamental image intensity transformation techniques using **Python, OpenCV, and NumPy**.

---

# Conclusion

In this practical, three basic image intensity transformation techniques — **Negative, Power-Law (Gamma), and Logarithmic Transformation** — were studied and implemented.

The practical helped in understanding how pixel intensity values can be mathematically modified for image enhancement. Both manual pixel-level implementation and library-based implementation were performed, providing an understanding of the underlying operations as well as their practical implementation using Python.
