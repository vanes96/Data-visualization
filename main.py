from PIL import Image
from sys import argv
import numpy as np
import math
import copy
import cv2
import os.path


def save_image(image, name):
    image_ = Image.fromarray(image)
    image_.save(name)


def show_image(file_name, title='Image'):
    image = cv2.imread(file_name)
    cv2.imshow(title, image)
    cv2.waitKey()


def create_pyramidal_decomposition(image):
    # definition of image size
    height, width = image.shape

    # definition of 3 pyramids: local minimums, maximums and average values
    mins = []
    avgs = []
    maxs = []

    # calculation of the pyramids levels number
    n_levels = math.ceil(math.log2(min(height, width)))

    # initialization of 3 pyramids by zeros
    for i in range(n_levels):
        mins.append(np.zeros((math.ceil(height / 2**i), math.ceil(width / 2**i)), np.uint8))
        avgs.append(np.zeros((math.ceil(height / 2**i), math.ceil(width / 2**i)), np.uint8))
        maxs.append(np.zeros((math.ceil(height / 2**i), math.ceil(width / 2**i)), np.uint8))

    # 0 level initialization of 3 pyramids by image
    mins[0] = image
    avgs[0] = image
    maxs[0] = image

    # filling of 3 pyramids
    for i in range(1, n_levels):
        height_l, width_l = maxs[i - 1].shape
        for row in range(math.ceil(height / 2**i)):
            for col in range(math.ceil(width / 2**i)):
                max_ = maxs[i - 1][row * 2][col * 2]
                min_ = mins[i - 1][row * 2][col * 2]
                sum_ = int(avgs[i - 1][row * 2][col * 2])
                count = 1
                if row * 2 + 1 < height_l:
                    max_ = max(maxs[i - 1][row * 2 + 1][col * 2], max_)
                    min_ = min(mins[i - 1][row * 2 + 1][col * 2], min_)
                    sum_ += avgs[i - 1][row * 2 + 1][col * 2]
                    count += 1
                if col * 2 + 1 < width_l:
                    max_ = max(maxs[i - 1][row * 2][col * 2 + 1], max_)
                    min_ = min(mins[i - 1][row * 2][col * 2 + 1], min_)
                    sum_ += avgs[i - 1][row * 2][col * 2 + 1]
                    count += 1
                if row * 2 + 1 < height_l and col * 2 + 1 < width_l:
                    max_ = max(maxs[i - 1][row * 2 + 1][col * 2 + 1], max_)
                    min_ = min(mins[i - 1][row * 2 + 1][col * 2 + 1], min_)
                    sum_ += avgs[i - 1][row * 2 + 1][col * 2 + 1]
                    count += 1
                maxs[i][row][col] = max_
                mins[i][row][col] = min_
                avgs[i][row][col] = sum_ // count

    return mins, avgs, maxs


def calculate_noise_threshold(image, is_normal_mode=True):
    height, width = image.shape
    box_size = 32
    n_min_boxes = 10
    limit_min, limit_max = 41, 150
    interval_size = (limit_max - limit_min + 1) // n_min_boxes

    boxes_indexes = []
    # boxes dispersions
    boxes_disps = [10000] * n_min_boxes
    # indexes of boxes with min dispersions
    boxes_min_indexes = [(-1, -1)] * n_min_boxes

    # searching for boxes with smallest dispersions in [n_min_boxes] intervals of brightness
    if is_normal_mode:
        # mode with not covered boxes
        for row in range(0, height - box_size, box_size):
            for col in range(0, width - box_size, box_size):
                avg, sum, disp, sum2 = 0, 0, 0, 0

                for i in range(box_size):
                    for j in range(box_size):
                        sum += image[row + i][col + j]

                avg = sum // box_size**2

                if limit_min <= avg <= limit_max:
                    k = (avg - limit_min) // interval_size
                    for i in range(box_size):
                        for j in range(box_size):
                            sum2 += (image[row + i][col + j] - avg)**2

                    disp = sum2 // box_size**2

                    if disp < boxes_disps[k]:
                        boxes_disps[k] = disp
                        boxes_min_indexes[k] = (row, col)
    else:
        # mode with covered boxes
        for row in range(height - box_size):
            for col in range(width - box_size):
                avg, sum, disp, sum2 = 0, 0, 0, 0

                for i in range(box_size):
                    for j in range(box_size):
                        sum += image[row + i][col + j]

                avg = sum // box_size**2

                if 31 <= avg <= 230:
                    k = (avg - 31) // 20
                    for i in range(box_size):
                        for j in range(box_size):
                            sum2 += (image[row + i][col + j] - avg)**2

                    disp = sum2 // box_size**2

                    if disp < boxes_disps[k]:
                        boxes_disps[k] = disp
                        boxes_min_indexes[k] = (row, col)

    # counting the number of active intervals
    sum = 0
    count_min_boxes = 0
    ind = 0
    for index in boxes_min_indexes:
        if boxes_disps[ind] != 10000:
            count_min_boxes += 1
            for i in range(0, box_size):
                for j in range(0, box_size):
                    sum += image[index[0] + i][index[1] + j]
        ind += 1

    # calculation of the result dispersion
    n_elems = count_min_boxes * box_size**2
    avg = sum // n_elems
    sum2 = 0
    for index in boxes_min_indexes:
        if index != (-1, -1):
            for i in range(box_size):
                for j in range(box_size):
                    sum2 += (image[index[0] + i][index[1] + j] - avg)**2

    disp_res = sum2 // n_elems

    #calculation of noise threshold
    noise_threshold = math.ceil(math.sqrt(disp_res))

    return noise_threshold


def create_threshold_map(mins, avgs, maxs, noise_threshold, is_normal_mode=True):
    n_levels = avgs.__len__()

    # definition of the pyramid levels sizes
    shapes = [avgs[n_levels - 1 - i].shape for i in range(n_levels)]

    # initialization of the threshold map by zeros
    threshold_map = [np.zeros((shapes[i][0], shapes[i][1]), np.uint8) for i in range(n_levels)]

    # ======================= Initialization if 0 level of threshold map =========================
    for row in range(shapes[0][0]):
        for col in range(shapes[0][1]):
            threshold_map[0][row][col] = (int(maxs[n_levels - 1][row][col]) + int(mins[n_levels - 1][row][col])) // 2
            #threshold_map[0][row][col] = int(avgs[n_levels - 1][row][col])
    # ============================================================================================

    # definition of the threshold map height (maximum level)
    k_level = n_levels

    for i in range(1, n_levels):
        level_min, level_avg, level_max = copy.deepcopy(mins[n_levels - 1 - i]), copy.deepcopy(avgs[n_levels - 1 - i]), copy.deepcopy(maxs[n_levels - 1 - i])
        # ======================= Increasing the threshold map by 2 times =========================
        for row in range(shapes[i - 1][0]):
            for col in range(shapes[i - 1][1]):
                if 2 * row < shapes[i][0] and 2 * col < shapes[i][1]:
                    threshold_map[i][2 * row][2 * col] = threshold_map[i - 1][row][col]
                if 2 * row < shapes[i][0] and 2 * col + 1 < shapes[i][1]:
                    threshold_map[i][2 * row][2 * col + 1] = threshold_map[i - 1][row][col]
                if 2 * row + 1 < shapes[i][0] and 2 * col < shapes[i][1]:
                    threshold_map[i][2 * row + 1][2 * col] = threshold_map[i - 1][row][col]
                if 2 * row + 1 < shapes[i][0] and 2 * col + 1 < shapes[i][1]:
                    threshold_map[i][2 * row + 1][2 * col + 1] = threshold_map[i - 1][row][col]
        # =====================================================================

        if is_normal_mode:
            # ======================= Convolution of i level of threshold map =========================
            level = copy.deepcopy(threshold_map[i])
            for row in range(1, shapes[i][0] - 1):
                for col in range(1, shapes[i][1] - 1):
                    k1, k2, k3 = 6, 1, 0

                    sum_ = k1 * level[row][col] + k2 * level[row + 1][col] + k2 * level[row][col + 1] + k2 * level[row - 1][col] + k2 * level[row][col - 1]
                    count = k1 + k2 * 4

                    sum_ += level[row + 1][col + 1] * k3 + level[row - 1][col + 1] * k3 + level[row + 1][col - 1] * k3 + level[row - 1][col - 1] * k3
                    count += k3 * 4

                    threshold_map[i][row][col] = sum_ // count
            # ==========================================================================================
        else:
            # alternative mode with pyramid convolution
            for row in range(1, shapes[i][0] - 1):
                for col in range(1, shapes[i][1] - 1):
                    k1, k2, k3 = 6, 0, 0

                    sum_min = k1 * level_min[row][col] + k2 * level_min[row + 1][col] + k2 * level_min[row][col + 1] + k2 * \
                           level_min[row - 1][col] + k2 * level_min[row][col - 1]
                    sum_max = k1 * level_max[row][col] + k2 * level_max[row + 1][col] + k2 * level_max[row][col + 1] + k2 * \
                              level_max[row - 1][col] + k2 * level_max[row][col - 1]
                    count = k1 + k2 * 4

                    sum_min += level_min[row + 1][col + 1] * k3 + level_min[row - 1][col + 1] * k3 + level_min[row + 1][col - 1] * k3 + \
                            level_min[row - 1][col - 1] * k3
                    sum_max += level_max[row + 1][col + 1] * k3 + level_max[row - 1][col + 1] * k3 + level_max[row + 1][col - 1] * k3 + \
                               level_max[row - 1][col - 1] * k3
                    count += k3 * 4

                    level_min[row][col] = sum_min // count
                    level_max[row][col] = sum_max // count

        if i <= k_level:
            # ======================= Updating of threshold values =========================
            for row in range(shapes[i][0]):
                for col in range(shapes[i][1]):
                    # check of the threshold change condition
                    if level_max[row][col] - level_min[row][col] > noise_threshold:
                        threshold_map[i][row][col] = (int(level_max[row][col]) + int(level_min[row][col])) // 2
                        #threshold_map[i][row][col] = int(avgs[n_levels - 1 - i][row][col])
            # =============================================================================

    return threshold_map, n_levels


def binarize_image(img_gray, threshold_map):
    height, width = img_gray.shape

    # initialization of binary image by the same size as gray image
    img_bin = copy.deepcopy(img_gray)

    # check of the individual pixels thresholds and image binarization
    for row in range(height):
        for col in range(width):
            if img_gray[row][col] > threshold_map[row][col]:
                img_bin[row][col] = 255
            else:
                img_bin[row][col] = 0

    return img_bin


def main(input_name):
    img_orig = Image.open(input_name)

    # conversion of image into gray shapes
    img_gray = cv2.cvtColor(np.asarray(img_orig), cv2.COLOR_RGB2GRAY)

    # generation of 3 different pyramids of local minimums, maximums and average values
    mins, avgs, maxs = create_pyramidal_decomposition(img_gray)

    # noise threshold definition
    #noise_threshold = compute_noise_threshold(img_gray, True)
    noise_threshold = 30

    # creation of threshold map
    threshold_map, n_levels = create_threshold_map(mins, avgs, maxs, noise_threshold, True)

    # image binarization
    img_bin = binarize_image(img_gray, threshold_map[n_levels-1])

    # saving of the binarized image
    output_name = input_name[:-4] + '_binarized.png'
    save_image(img_bin, output_name)

    # showing of the original and binarized images
    show_image(input_name, 'Original Image')
    show_image(output_name, 'Binarized Image')

#main('text_normal_small.png')

#======================= MAIN =========================
if __name__ == '__main__':
    n_args = len(argv)
    assert 2 <= n_args <= 3
    assert os.path.exists(argv[1])
    main(*argv[1:])