import cv2
import numpy as np
from os import path, makedirs, listdir
from numba import jit

"""
    functions :
"""
palette_cga = [
    (0, 0, 0),
    (255, 56, 100),
    (45, 226, 230),
    (255, 255, 255),
]

palette_sefia = [
    (0, 0, 0),
    (132, 49, 16),
    (132, 49, 16),
    (255, 173, 99),
]

# https://lospec.com/palette-list/ty-murder-mystery-16
palette_ty_murder_mystery_16 = [
    (222, 206, 180),  # #deceb4
    (156, 171, 177),  # #9cabb1
    (128, 128, 120),  # #808078
    (83, 74, 59),     # #534a3b
    (224, 155, 77),   # #e09b4d
    (161, 125, 55),   # #a17d37
    (120, 92, 59),    # #785c3b
    (62, 43, 34),     # #3e2b22
    (187, 143, 107),  # #bb8f6b
    (122, 98, 96),    # #7a6260
    (75, 65, 88),     # #4b4158
    (187, 100, 58),   # #bb643a
    (141, 62, 41),    # #8d3e29
    (94, 41, 47),     # #5e292f
    (121, 146, 64),   # #799240
    (99, 96, 46),     # #63602e
]

# https://lospec.com/palette-list/retro-115
palette_retro_115 = [
    (4, 0, 38),      # #040026
    (0, 75, 192),    # #004bc0
    (0, 151, 236),   # #0097ec
    (0, 243, 252),   # #00f3fc
    (255, 188, 255), # #ffbcff
    (213, 105, 246), # #d569f6
    (108, 31, 211),  # #6c1fd3
]

# https://lospec.com/palette-list/lost-century
palette_lost_century = [
    (209, 177, 135),  # #d1b187
    (199, 123, 88),   # #c77b58
    (174, 93, 64),    # #ae5d40
    (121, 68, 74),    # #79444a
    (75, 61, 68),     # #4b3d44
    (186, 145, 88),   # #ba9158
    (146, 116, 65),   # #927441
    (77, 69, 57),     # #4d4539
    (119, 116, 59),   # #77743b
    (179, 165, 85),   # #b3a555
    (210, 201, 165),  # #d2c9a5
    (140, 171, 161),  # #8caba1
    (75, 114, 110),   # #4b726e
    (87, 72, 82),     # #574852
    (132, 120, 117),  # #847875
    (171, 155, 142),  # #ab9b8e
]


# image dithering function : Floyd-Steinberg
@jit(nopython=True)
def floyd_steinberg(image):
    image = image / 255
    height = image.shape[0]
    width = image.shape[1]

    for y in range(height) :
        for x in range(width) :
            rounded = np.round(image[y][x])        
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
            rounded = np.round(image[y][x])
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

def apply_palette(image, palette):
    image = image.astype(np.float32)
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)

    palette_np = np.array(palette, dtype=np.float32)
    palette_np = palette_np[:, [2, 1, 0]]

    # 브로드캐스팅을 활용한 유클리디안 거리 계산
    diff = pixels[:, None, :] - palette_np[None, :, :]   # (H*W, N, 3)
    dists = np.sum(diff ** 2, axis=2)                    # (H*W, N)
    indices = np.argmin(dists, axis=1)                   # (H*W,)
    matched = palette_np[indices]                        # (H*W, 3)

    converted_image = matched.reshape(h, w, 3).astype(np.uint8)
    return converted_image

def downsample_pick_median_brightness_color(image, target_w, target_h):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w, c = image.shape
    block_h, block_w = h // target_h, w // target_w

    result = np.zeros((target_h, target_w, c), dtype=image.dtype)

    for i in range(target_h):
        for j in range(target_w):
            block_bgr = image[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            block_v = hsv[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w, 2]

            median_val = np.median(block_v)
            # median에 가장 가까운 픽셀 인덱스 찾기
            abs_diff = np.abs(block_v - median_val)
            idx = np.unravel_index(np.argmin(abs_diff), abs_diff.shape)

            result[i, j] = block_bgr[idx]

    return result

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

for i in range(len(image_list)):
    image_name = image_list[i]
    print("Converting " + str(int(i / len(image_list) * 100)) + "%... " + image_name)
    
    # image load
    origin_image = cv2.imread('./origin/' + image_name)
    
    # # 1. 히스토그램 평탄화 (CLAHE)
    # lab = cv2.cvtColor(origin_image, cv2.COLOR_BGR2LAB)
    # l, a, b = cv2.split(lab)
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # cl = clahe.apply(l)
    # lab_clahe = cv2.merge((cl, a, b))
    # contrast_img = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # 2. 대비 조작
    contrast_img = cv2.convertScaleAbs(origin_image, alpha=1.0, beta=0)

    # 3. 이미지 크기 축소
    # contrast_img_small = cv2.resize(contrast_img, (int(contrast_img.shape[1] * 128 / contrast_img.shape[0]), 128), interpolation = cv2.INTER_NEAREST)
    contrast_img_small = downsample_pick_median_brightness_color(contrast_img, int(contrast_img.shape[1] * 128 / contrast_img.shape[0]), 128)
    
    # Atkinson 적용
    contrast_img_small = atkinson(contrast_img_small)
    
    # cga
    createDirectory('cga')
    new_image = apply_palette(contrast_img_small, palette_cga)
    new_image = cv2.resize(new_image, (new_image.shape[1] * 10, new_image.shape[0] * 10), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite("./cga/" + image_name.split('.')[0] + ".png", new_image)
    
    # # sefia
    createDirectory('sefia')
    new_image = apply_palette(contrast_img_small, palette_sefia)
    new_image = cv2.resize(new_image, (new_image.shape[1] * 10, new_image.shape[0] * 10), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite("./sefia/" + image_name.split('.')[0] + ".png", new_image)
    
    # # retro_115
    createDirectory('retro_115')
    new_image = apply_palette(contrast_img_small, palette_retro_115)
    new_image = cv2.resize(new_image, (new_image.shape[1] * 10, new_image.shape[0] * 10), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite("./retro_115/" + image_name.split('.')[0] + ".png", new_image)

    # # ty_murder_mystery_16
    createDirectory('ty_murder_mystery_16')
    new_image = apply_palette(contrast_img_small, palette_ty_murder_mystery_16)
    new_image = cv2.resize(new_image, (new_image.shape[1] * 10, new_image.shape[0] * 10), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite("./ty_murder_mystery_16/" + image_name.split('.')[0] + ".png", new_image)
    
    # resurrect_64
    createDirectory('lost_century')
    new_image = apply_palette(contrast_img_small, palette_lost_century)
    new_image = cv2.resize(new_image, (new_image.shape[1] * 10, new_image.shape[0] * 10), interpolation = cv2.INTER_NEAREST)
    cv2.imwrite("./lost_century/" + image_name.split('.')[0] + ".png", new_image)
    
    