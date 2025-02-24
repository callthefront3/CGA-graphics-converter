from cv2 import imread, imwrite, cvtColor, resize, convertScaleAbs, COLOR_BGR2GRAY, INTER_NEAREST
from os import path, makedirs, listdir
from math import floor
from numpy import round
from numba import jit


"""

    functions :

"""
# image change color : CGA
# cga_rgb_b = [0, 0, 0]
# cga_rgb_m = [255, 85, 255]
# cga_rgb_c = [255, 255, 85]
# cga_rgb_w = [255, 255, 255]
cga_rgb_b = [0, 0, 0]
cga_rgb_m = [100, 56, 255]
cga_rgb_c = [230, 226, 45]
cga_rgb_w = [255, 255, 255]

# image change color : Sefia
sefia_rgb_b = [0, 0, 0]
sefia_rgb_m = [16, 49, 132]
sefia_rgb_c = [16, 49, 132]
# sefia_rgb_c = [99, 173, 255]
sefia_rgb_w = [99, 173, 255]
# sefia_rgb_w = [255, 255, 255]

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
def rgbToCga(bgr) :
    r = floor(bgr[2] / 86) + 1
    g = floor(bgr[1] / 86) + 1
    b = floor(bgr[0] / 86) + 1
    
    if (r==3 and g==3) or (r==2 and g==2 and b==2):
        return cga_rgb_w
    elif (r+g <= 3) or (r==2 and g==2 and b==1):
        return cga_rgb_b
    elif (r == 3):
        return cga_rgb_m
    elif (g == 3) or (r==2 and g==2 and b==3):
        return cga_rgb_c
    
    print("Error: color out of range " + str(r) + " " + str(g) + " " + str(b))
    return cga_rgb_b

# BGR color to Sefia color function
def rgbToSefia(bgr) :
    r = floor(bgr[2] / 86) + 1
    g = floor(bgr[1] / 86) + 1
    b = floor(bgr[0] / 86) + 1
    
    if (r==3 and g==3) or (r==2 and g==2 and b==2):
        return sefia_rgb_w
    elif (r+g <= 3) or (r==2 and g==2 and b==1):
        return sefia_rgb_b
    elif (r == 3):
        return sefia_rgb_m
    elif (g == 3) or (r==2 and g==2 and b==3):
        return sefia_rgb_c
    
    print("Error: color out of range " + str(r) + " " + str(g) + " " + str(b))
    return sefia_rgb_b

# image CGA color convert function
def cga_convert(image):
    for y in range(image.shape[0]) :
        for x in range(image.shape[1]) :
            image[y, x] = rgbToCga(image[y, x])
            
    return image

# image Sefia color convert function
def sefia_convert(image):
    for y in range(image.shape[0]) :
        for x in range(image.shape[1]) :
            image[y, x] = rgbToSefia(image[y, x])
            
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
createDirectory('normal')
createDirectory('cga')
createDirectory('sefia')

for i in range(len(image_list)):
    image_name = image_list[i]
    print("Converting " + str(int(i / len(image_list) * 100)) + "%... " + image_name)
    
    # image load
    origin_image = imread('./origin/' + image_name)
    origin_image = convertScaleAbs(origin_image, alpha=1.5, beta=-50)
    origin_image_180 = resize(origin_image, (int(origin_image.shape[1] * 180 / origin_image.shape[0]), 180), interpolation = INTER_NEAREST)
    origin_image_360 = resize(origin_image, (int(origin_image.shape[1] * 360 / origin_image.shape[0]), 360), interpolation = INTER_NEAREST)
    origin_image_480 = resize(origin_image, (int(origin_image.shape[1] * 480 / origin_image.shape[0]), 480), interpolation = INTER_NEAREST)
    origin_image_540 = resize(origin_image, (int(origin_image.shape[1] * 540 / origin_image.shape[0]), 540), interpolation = INTER_NEAREST)

    # gray image
    gray_image = cvtColor(origin_image, COLOR_BGR2GRAY)
    gray_image = floyd_steinberg(gray_image)
    imwrite("./gray/" + image_name.split('.')[0] + ".png", gray_image)

    gray_image_180 = cvtColor(origin_image_180, COLOR_BGR2GRAY)
    gray_image_180 = floyd_steinberg(gray_image_180)
    gray_image_180 = resize(gray_image_180, (gray_image_180.shape[1] * 4, gray_image_180.shape[0] * 4), interpolation = INTER_NEAREST)
    imwrite("./gray/" + image_name.split('.')[0] + "_180.png", gray_image_180)

    gray_image_360 = cvtColor(origin_image_360, COLOR_BGR2GRAY)
    gray_image_360 = floyd_steinberg(gray_image_360)
    gray_image_360 = resize(gray_image_360, (gray_image_360.shape[1] * 2, gray_image_360.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./gray/" + image_name.split('.')[0] + "_360.png", gray_image_360)

    gray_image_480 = cvtColor(origin_image_480, COLOR_BGR2GRAY)
    gray_image_480 = floyd_steinberg(gray_image_480)
    gray_image_480 = resize(gray_image_480, (gray_image_480.shape[1] * 2, gray_image_480.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./gray/" + image_name.split('.')[0] + "_480.png", gray_image_480)

    gray_image_540 = cvtColor(origin_image_540, COLOR_BGR2GRAY)
    gray_image_540 = floyd_steinberg(gray_image_540)
    gray_image_540 = resize(gray_image_540, (gray_image_540.shape[1] * 2, gray_image_540.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./gray/" + image_name.split('.')[0] + "_540.png", gray_image_540)

    # Nomal Image
    nomal_image = atkinson(origin_image)
    imwrite("./normal/" + image_name.split('.')[0] + ".png", nomal_image)

    nomal_image_180 = atkinson(origin_image_180)
    nomal_image_180 = resize(nomal_image_180, (nomal_image_180.shape[1] * 4, nomal_image_180.shape[0] * 4), interpolation = INTER_NEAREST)
    imwrite("./normal/" + image_name.split('.')[0] + "_180.png", nomal_image_180)

    nomal_image_360 = atkinson(origin_image_360)
    nomal_image_360 = resize(nomal_image_360, (nomal_image_360.shape[1] * 2, nomal_image_360.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./normal/" + image_name.split('.')[0] + "_360.png", nomal_image_360)

    nomal_image_480 = atkinson(origin_image_480)
    nomal_image_480 = resize(nomal_image_480, (nomal_image_480.shape[1] * 2, nomal_image_480.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./normal/" + image_name.split('.')[0] + "_480.png", nomal_image_480)

    nomal_image_540 = atkinson(origin_image_540)
    nomal_image_540 = resize(nomal_image_540, (nomal_image_540.shape[1] * 2, nomal_image_540.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./normal/" + image_name.split('.')[0] + "_540.png", nomal_image_540)

    # CGA image
    cga_image = atkinson(origin_image)
    cga_image = cga_convert(cga_image)
    imwrite("./cga/" + image_name.split('.')[0] + ".png", cga_image)

    cga_image_180 = atkinson(origin_image_180)
    cga_image_180 = cga_convert(cga_image_180)
    cga_image_180 = resize(cga_image_180, (cga_image_180.shape[1] * 4, cga_image_180.shape[0] * 4), interpolation = INTER_NEAREST)
    imwrite("./cga/" + image_name.split('.')[0] + "_180.png", cga_image_180)

    cga_image_360 = atkinson(origin_image_360)
    cga_image_360 = cga_convert(cga_image_360)
    cga_image_360 = resize(cga_image_360, (cga_image_360.shape[1] * 2, cga_image_360.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./cga/" + image_name.split('.')[0] + "_360.png", cga_image_360)

    cga_image_480 = atkinson(origin_image_480)
    cga_image_480 = cga_convert(cga_image_480)
    cga_image_480 = resize(cga_image_480, (cga_image_480.shape[1] * 2, cga_image_480.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./cga/" + image_name.split('.')[0] + "_480.png", cga_image_480)

    cga_image_540 = atkinson(origin_image_540)
    cga_image_540 = cga_convert(cga_image_540)
    cga_image_540 = resize(cga_image_540, (cga_image_540.shape[1] * 2, cga_image_540.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./cga/" + image_name.split('.')[0] + "_540.png", cga_image_540)

    # Sefia image
    sefia_image = atkinson(origin_image)
    sefia_image = sefia_convert(sefia_image)
    imwrite("./sefia/" + image_name.split('.')[0] + ".png", sefia_image)

    sefia_image_180 = atkinson(origin_image_180)
    sefia_image_180 = sefia_convert(sefia_image_180)
    sefia_image_180 = resize(sefia_image_180, (sefia_image_180.shape[1] * 4, sefia_image_180.shape[0] * 4), interpolation = INTER_NEAREST)
    imwrite("./sefia/" + image_name.split('.')[0] + "_180.png", sefia_image_180)

    sefia_image_360 = atkinson(origin_image_360)
    sefia_image_360 = sefia_convert(sefia_image_360)
    sefia_image_360 = resize(sefia_image_360, (sefia_image_360.shape[1] * 2, sefia_image_360.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./sefia/" + image_name.split('.')[0] + "_360.png", sefia_image_360)

    sefia_image_480 = atkinson(origin_image_480)
    sefia_image_480 = sefia_convert(sefia_image_480)
    sefia_image_480 = resize(sefia_image_480, (sefia_image_480.shape[1] * 2, sefia_image_480.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./sefia/" + image_name.split('.')[0] + "_480.png", sefia_image_480)

    sefia_image_540 = atkinson(origin_image_540)
    sefia_image_540 = sefia_convert(sefia_image_540)
    sefia_image_540 = resize(sefia_image_540, (sefia_image_540.shape[1] * 2, sefia_image_540.shape[0] * 2), interpolation = INTER_NEAREST)
    imwrite("./sefia/" + image_name.split('.')[0] + "_540.png", sefia_image_540)
