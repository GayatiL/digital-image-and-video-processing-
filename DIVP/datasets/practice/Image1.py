import cv2
import numpy as np

img = cv2.imread("baboon.png")

if img is None: 
    print("imgae not found")
    exit()

while True:
    print("/n Baboon image processing menu")
    print("1. Display orignal image")
    print("2. Conert to gray scale")
    print("3. Increase the brightness")
    print("4. Decrease the brightness")
    print("5. Increase contrast")
    print("6. Decrease Contrast")
    print("7. Negative Image")
    print("8. Blur image ")
    print("9. Sharpen image")
    print("10.Resize image")
    print("11.Rotate image")
    print("12.Exit")
    
    choice = int(input("Enter your choice: "))

    if choice == 1:
        cv2.imshow("Original Image", img)

    elif choice == 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Grayscale", gray)

    elif choice == 3:
        bright = cv2.convertScaleAbs(img, alpha=1, beta=50)
        cv2.imshow("Brightness Increased", bright)

    elif choice == 4:
        dark = cv2.convertScaleAbs(img, alpha=1, beta=-50)
        cv2.imshow("Brightness Decreased", dark)

    elif choice == 5:
        contrast = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        cv2.imshow("Contrast Increased", contrast)

    elif choice == 6:
        contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
        cv2.imshow("Contrast Decreased", contrast)

    elif choice == 7:
        negative = 255 - img
        cv2.imshow("Negative Image", negative)

    elif choice == 8:
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        cv2.imshow("Blurred Image", blur)

    elif choice == 9:
        kernel = np.array([[0,-1,0],
                           [-1,5,-1],
                           [0,-1,0]])
        sharp = cv2.filter2D(img, -1, kernel)
        cv2.imshow("Sharpened Image", sharp)

    

    elif choice == 10:
        resized = cv2.resize(img, (300, 300))
        cv2.imshow("Resized Image", resized)

    elif choice == 11:
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows))
        cv2.imshow("Rotated Image", rotated)

 

    elif choice == 12:
        print("Exiting...")
        break

    else:
        print("Invalid Choice!")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



