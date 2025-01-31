from cv2 import imread, imwrite, cvtColor, resize, COLOR_BGR2GRAY
from os import path, makedirs, listdir
from math import floor
from numpy import round
from numba import jit


"""

    functions :

"""
# image change color
rgb_b = [0, 0, 0]
rgb_m = [255, 85, 255]
rgb_c = [255, 255, 85]
rgb_w = [255, 255, 255]

# image dithering function : Floyd-Steinberg
@jit(nopython=True)
def floyd_steinberg(image):
    image = image / 255
    height = image.shape[0]
    width = image.shape[1]

    for y in range(height) :
        for x in range(width) :
            rounded = round(image[y][x])        
            err = image[y, x] - rounded
            image[y, x] = rounded
            
            if x < width - 1 : image[y, x+1] = image[y, x+1] + (7/16) * err
            if y < height - 1 :
                image[y+1, x] = image[y+1, x] + (5/16) * err
                if x > 0 : image[y+1, x-1] = image[y+1, x-1] + (1/16) * err
                if x < width - 1 : image[y+1, x+1] = image[y+1, x+1] + (3/16) * err

    return image * 255

# image dithering function : Atkinson
@jit(nopython=True)
def atkinson(image):
    image = image / 255
    height = image.shape[0]
    width = image.shape[1]
    frac = 8

    for y in range(height) :
        for x in range(width) :
            rounded = round(image[y][x])
            err = image[y, x] - rounded
            image[y, x] = rounded
            
            if x < width - 1 : image[y, x+1] = image[y, x+1] + err / frac
            if x < width - 2 : image[y, x+2] = image[y, x+2] + err / frac
            if y < height - 1 :
                image[y+1, x] = image[y+1, x] + err / frac
                if x > 0 : image[y+1, x-1] = image[y+1, x-1] + err / frac
                if x < width - 1 : image[y+1, x+1] = image[y+1, x+1] + err / frac
            if y < height - 2 : image[y+2, x] = image[y+2, x] + err / frac

    return image * 255

# BGR color to CGA color function
def rgbToIrgb(bgr) :
    r = floor(bgr[2] / 86) + 1
    g = floor(bgr[1] / 86) + 1
    b = floor(bgr[0] / 86) + 1
    
    if (r==3 and g==3) or (r==2 and g==2 and b==2):
        return rgb_w
    elif (r+g <= 3) or (r==2 and g==2 and b==1):
        return rgb_b
    elif (r == 3):
        return rgb_m
    elif (g == 3) or (r==2 and g==2 and b==3):
        return rgb_c
    
    print("Error: color out of range " + str(r) + " " + str(g) + " " + str(b))
    return rgb_b

# image color convert function
def cga_convert(image):
    for y in range(image.shape[0]) :
        for x in range(image.shape[1]) :
            image[y, x] = rgbToIrgb(image[y, x])
            
    return image

# create directory function
def createDirectory(directory):
    try:
        if not path.exists(directory):
            makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")



"""

    Main code :

"""
image_list = listdir('origin')
createDirectory('gray')
createDirectory('cga')

for image_name in image_list:
    print("Converting... " + image_name)
    
    # image load
    origin_image = imread('./origin/' + image_name)
    origin_image = resize(origin_image, (int(origin_image.shape[1] * 540 / origin_image.shape[0]), 540))

    # gray image
    gray_image = cvtColor(origin_image, COLOR_BGR2GRAY)
    gray_image = floyd_steinberg(gray_image)
    imwrite("./gray/" + image_name.split('.')[0] + ".png", gray_image)

    # cga image
    cga_image = atkinson(origin_image)
    cga_image = cga_convert(cga_image)
    imwrite("./cga/" + image_name.split('.')[0] + ".png", cga_image)


