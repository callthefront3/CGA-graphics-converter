import sys
import cv2
import numpy
import numba

# image load
origin_image = cv2.imread('youngk.jpg')
origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

# image dithering
def floyd_steinberg(image):
    image = image / 255
    height = image.shape[0]
    width = image.shape[1]

    for y in range(height) :
        for x in range(width) :
            rounded = round(image[y, x])
            err = image[y, x] - rounded
            image[y, x] = rounded
            
            # Floyd-Steinberg
            if x < width - 1 : image[y, x+1] = image[y, x+1] + (7/16) * err
            if y < height - 1 :
                image[y+1, x] = image[y+1, x] + (5/16) * err
                if x > 0 : image[y+1, x-1] = image[y+1, x-1] + (1/16) * err
                if x < width - 1 : image[y+1, x+1] = image[y+1, x+1] + (3/16) * err

    return image * 255

def atkinson(image):
    image = image / 255
    frac = 8
    height = image.shape[0]
    width = image.shape[1]

    for y in range(height) :
        for x in range(width) :
            rounded = round(image[y, x])
            err = image[y, x] - rounded
            image[y, x] = rounded
            
            # Atkinson
            if x < width - 1 : image[y, x+1] = image[y, x+1] + err / frac
            if x < width - 2 : image[y, x+2] = image[y, x+2] + err / frac
            if y < height - 1 :
                image[y+1, x] = image[y+1, x] + err / frac
                if x > 0 : image[y+1, x-1] = image[y+1, x-1] + err / frac
                if x < width - 1 : image[y+1, x+1] = image[y+1, x+1] + err / frac
            if y < height - 2 : image[y+2, x] = image[y+2, x] + err / frac

    return image * 255

# image show
converted_image = floyd_steinberg(origin_image)
# converted_image = atkinson(origin_image)

cv2.imshow("Image", converted_image)

# image save
cv2.imwrite("bnw_floyd.png", floyd_steinberg(origin_image))
cv2.imwrite("bnw_atkinson.png", atkinson(origin_image))

cv2.waitKey()
cv2.destroyAllWindows()

