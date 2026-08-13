import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Get the folder where lab2.py is located
folder = os.path.dirname(os.path.abspath(__file__))

# Image paths
source1_path = os.path.join(folder, "source1.jpg")
source2_path = os.path.join(folder, "source2.jpg")
reference_path = os.path.join(folder, "reference.jpg")

# Read images
source1 = cv2.imread(source1_path, 0)
source2 = cv2.imread(source2_path, 0)
reference = cv2.imread(reference_path, 0)

# Check whether images loaded
print("Source1 loaded:", source1 is not None)
print("Source2 loaded:", source2 is not None)
print("Reference loaded:", reference is not None)

# ---------------------------------------------------
# Function to plot Histogram
# ---------------------------------------------------

def plot_hist(img, title):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist)
    plt.title(title)
    plt.xlim([0,256])

# ---------------------------------------------------
# Histogram Matching (From First Principles)
# ---------------------------------------------------

def histogram_matching(source, reference):

    src_hist = cv2.calcHist([source],[0],None,[256],[0,256]).ravel()
    ref_hist = cv2.calcHist([reference],[0],None,[256],[0,256]).ravel()

    # Normalize Histograms
    src_hist = src_hist / src_hist.sum()
    ref_hist = ref_hist / ref_hist.sum()

    # CDF
    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)

    # Lookup Table
    lookup = np.zeros(256)

    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        lookup[i] = j

    matched = lookup[source].astype(np.uint8)

    return matched

# ---------------------------------------------------
# Histogram Matching Results
# ---------------------------------------------------

matched1 = histogram_matching(source1, reference)
matched2 = histogram_matching(source2, reference)

# ---------------------------------------------------
# Analytical Target Histogram
# Dark / Moody Style
# ---------------------------------------------------

target_hist = np.zeros(256)

for i in range(256):
    target_hist[i] = np.exp(-i/50)

target_hist = target_hist/target_hist.sum()
target_cdf = np.cumsum(target_hist)

def histogram_match_custom(source, target_cdf):

    src_hist = cv2.calcHist([source],[0],None,[256],[0,256]).ravel()
    src_hist = src_hist/src_hist.sum()

    src_cdf = np.cumsum(src_hist)

    lookup = np.zeros(256)

    j=0
    for i in range(256):
        while j<255 and target_cdf[j] < src_cdf[i]:
            j+=1
        lookup[i]=j

    return lookup[source].astype(np.uint8)

stylized = histogram_match_custom(source1,target_cdf)

# ---------------------------------------------------
# CLAHE
# ---------------------------------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8,8)
)

clahe_img = clahe.apply(source1)

# ---------------------------------------------------
# Parameter Sweep
# ---------------------------------------------------

clip_limits = [1,2,4]
tile_sizes = [4,8,16]

plt.figure(figsize=(12,8))

count=1

for clip in clip_limits:
    for tile in tile_sizes:

        clahe=cv2.createCLAHE(
            clipLimit=clip,
            tileGridSize=(tile,tile)
        )

        result=clahe.apply(source1)

        plt.subplot(3,3,count)
        plt.imshow(result,cmap='gray')
        plt.title("Clip="+str(clip)+" Tile="+str(tile))
        plt.axis("off")

        count+=1

plt.show()

# ---------------------------------------------------
# Histogram Comparison
# ---------------------------------------------------

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(source1,cmap='gray')
plt.title("Source1")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(reference,cmap='gray')
plt.title("Reference")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(matched1,cmap='gray')
plt.title("Matched")
plt.axis("off")

plt.subplot(2,3,4)
plot_hist(source1,"Source Histogram")

plt.subplot(2,3,5)
plot_hist(reference,"Reference Histogram")

plt.subplot(2,3,6)
plot_hist(matched1,"Matched Histogram")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Show Images
# ---------------------------------------------------

cv2.imshow("Source1",source1)
cv2.imshow("Source2",source2)
cv2.imshow("Reference",reference)

cv2.imshow("Matched1",matched1)
cv2.imshow("Matched2",matched2)

cv2.imshow("Stylized",stylized)

cv2.imshow("CLAHE",clahe_img)

cv2.waitKey(0)
cv2.destroyAllWindows()