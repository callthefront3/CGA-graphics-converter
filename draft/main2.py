import math
import cv2
import numpy
import numba

# image load
origin_image = cv2.imread('youngk.jpg', cv2.IMREAD_UNCHANGED)
converted_image = origin_image

# image dithering
def floyd_steinberg(image):
    image = image / 255
    
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]

    for y in range(height) :
        for x in range(width) :
            for c in range(channel) :
                rounded = round(image[y][x][c])        
                err = image[y, x, c] - rounded
                image[y, x, c] = rounded
                
                # Floyd-Steinberg
                if x < width - 1 : image[y, x+1, c] = image[y, x+1, c] + (7/16) * err
                if y < height - 1 :
                    image[y+1, x, c] = image[y+1, x, c] + (5/16) * err
                    if x > 0 : image[y+1, x-1, c] = image[y+1, x-1, c] + (1/16) * err
                    if x < width - 1 : image[y+1, x+1, c] = image[y+1, x+1, c] + (3/16) * err

    return image * 255

def atkinson(image):
    image = image / 255
    
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]
    frac = 8

    for y in range(height) :
        for x in range(width) :
            for c in range(channel) :
                rounded = round(image[y][x][c])
                err = image[y, x, c] - rounded
                image[y, x, c] = rounded
                
                # Atkinson
                if x < width - 1 : image[y, x+1, c] = image[y, x+1, c] + err / frac
                if x < width - 2 : image[y, x+2, c] = image[y, x+2, c] + err / frac
                if y < height - 1 :
                    image[y+1, x, c] = image[y+1, x, c] + err / frac
                    if x > 0 : image[y+1, x-1, c] = image[y+1, x-1, c] + err / frac
                    if x < width - 1 : image[y+1, x+1, c] = image[y+1, x+1, c] + err / frac
                if y < height - 2 : image[y+2, x, c] = image[y+2, x, c] + err / frac

    return image * 255

# image convert
# converted_image = floyd_steinberg(converted_image)
converted_image = atkinson(converted_image)

# image change color
rgb_b = [0, 0, 0]
rgb_m = [255, 85, 255]
rgb_c = [255, 255, 85]
rgb_w = [255, 255, 255]

# R=3 & G=3 || R=2 & G=2 & B=2 -> w
# R+G <= 3  || R=2 & G=2 & B=1 -> b
# R=3 -> m
# G=3 || R=2 & G=2 & B=3 -> c
def rgbToIrgb(bgr) :
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
    
    print("예외 발생" + str(r) + str(g) + str(b))
    return rgb_b

for y in range(converted_image.shape[0]) :
    for x in range(converted_image.shape[1]) :
        converted_image[y, x] = rgbToIrgb(converted_image[y, x])

# image show
cv2.imshow("Image", numpy.uint8(converted_image))

# image save
cv2.imwrite("rgbTorgb_atkinson_later.png", converted_image)

# quit
cv2.waitKey()
cv2.destroyAllWindows()

