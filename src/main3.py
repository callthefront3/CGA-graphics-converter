import math
import cv2
import numpy
import numba

# image load
origin_image = cv2.imread('youngk.jpg')
converted_image = origin_image

# image change color
rgb_b = [0, 0, 0]
rgb_m = [255, 85, 255]
rgb_c = [255, 255, 85]
rgb_w = [255, 255, 255]

# R=3 & G=3 || R=2 & G=2 & B=2 -> w
# R+G <= 3  || R=2 & G=2 & B=1 -> b
# R=3 -> m
# G=3 || R=2 & G=2 & B=3 -> c
def bgrToCGA(bgr) :
    r = math.floor(bgr[2] / 86) + 1
    g = math.floor(bgr[1] / 86) + 1
    b = math.floor(bgr[0] / 86) + 1
    
    if (r==3 and g==3) or (r==2 and g==2 and b==2):
        return rgb_w
    elif (r+g <= 3) or (r==2 and g==2 and b==1):
        return rgb_b
    elif (r == 3):
        return rgb_m
    elif (g == 3) or (r==2 and g==2 and b==3):
        return rgb_c
    
    print("예외 발생: " + str(bgr[2]) + "->" + str(r) + " " + str(bgr[1]) + "->" +  str(g) + " " + str(bgr[0]) + "->" +  str(b))
    return rgb_b

# image dithering
def floyd_steinberg(image):
    height = image.shape[0]
    width = image.shape[1]

    for y in range(height) :
        for x in range(width) :
            rounded = bgrToCGA(image[y][x])        
            err = image[y, x] - rounded
            image[y, x] = rounded
            
            # Floyd-Steinberg
            if x < width - 1 : image[y, x+1] = image[y, x+1] + (7/16) * err
            if y < height - 1 :
                image[y+1, x] = image[y+1, x] + (5/16) * err
                if x > 0 : image[y+1, x-1] = image[y+1, x-1] + (1/16) * err
                if x < width - 1 : image[y+1, x+1] = image[y+1, x+1] + (3/16) * err

    return image

def atkinson(image):
    height = image.shape[0]
    width = image.shape[1]
    frac = 8

    for y in range(height) :
        for x in range(width) :
            rounded = bgrToCGA(image[y][x])
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

    return image

# image convert
# converted_image = floyd_steinberg(converted_image)
converted_image = atkinson(converted_image)

# image show
cv2.imshow("Image", converted_image)

# image save
cv2.imwrite("rgbTorgb_atkinson_inside.png", converted_image)

# quit
cv2.waitKey()
cv2.destroyAllWindows()

