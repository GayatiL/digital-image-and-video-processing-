# Original image represented as a 2D array
image = [
    [10, 50, 100],
    [150, 200, 255],
    [30, 80, 170]
]

rows = len(image)
cols = len(image[0])

negative = []

for i in range(rows):
    row = []
    for j in range(cols):
        pixel = 255 - image[i][j]
        row.append(pixel)
    negative.append(row)

print("Original Image:")
for row in image:
    print(row)

print("\nNegative Image:")
for row in negative:
    print(row) 