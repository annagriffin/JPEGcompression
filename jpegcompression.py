import cv2
from PIL import Image
import os.path
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt




quantization_table = [[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]]


dct_matrix = [[.3536, .3536, .3536, .3536, .3536, .3536, .3536, .3536],
              [.4904, .4157, .2778, .0975, -.0975, -.2778, -.4157, -.4904],
              [.4619, .1913, -.1913, -.4619, -.4619, -.1913, .1913, .4619],
              [.4157, -.0975, -.4904, -.2778, .2778, .4904, .0975, -.4157],
              [.3536, -.3536, -.3536, .3536, .3536, -.3536, -.3536, .3536],
              [.2278, -.4904, .0975, .4157, -.4157, -.0975, .4904, -.2778],
              [.1913, -.4619, .4619, -.1913, -.1913, .4619, -.4619, .1913],
              [.0975, -.2778, .4157, -.4904, .4904, -.4157, .2778, -.0975]]


def convert_RGB_to_YCbCr(filepath):
    """ Converts image from RGB colorspace to YCbCr colorspace

    filepath: string path to image file
    returns: image in YCC_Scale """

    img = cv2.imread(filepath)
    bgr_img = img[...,::-1]
    YCC_scale = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCR_CB)

    return YCC_scale

def partition(image):
    """ Separates image into 8x8 blocks

    image: image in YCbCr scale
    returns: list of 8x8 blocks """

    width = image.shape[1]
    height = image.shape[0]
    blocks = []
    image = np.array(image)
    delta_x = int(width/20)
    delta_y = int(height/30)
    index_x=0
    index_y=0
    for i in range(0,20):
        for j in range(0,30):
            temp_image = image[index_y:index_y+delta_y,index_x:index_x+delta_x]
            blocks.append(temp_image)
            index_y += delta_y
        index_y=0
        index_x += delta_x
    return blocks

def inverse_partition(y_component, cb_component, cr_component):

    image = []

    i = 0
    j = 0
    print(y_component)
    print(cb_component)
    print(cr_component)

    while i < len(y_component):
        row = []
        while j < len(y_component):
            temp2 = []
            temp2.append(y_component[i][j])
            temp2.append(cb_component[i][j])
            temp2.append(cr_component[i][j])
            row.append(temp2)
            j += 1

        image.append(row)
        j = 0
        i += 1

    return image


def get_component_value(image, component):
    """ Gets the Y component of an image

    image: 8x8 block
    returns: list of Y component values """

    image = np.array(image,dtype = np.integer)
    list_of_values = []
    for row in range(len(image[0])):
        list_of_values1 = []
        for column in range(len(image[0])):
            pixel = image[row][column][component]
            list_of_values1.append(pixel)
        list_of_values.append(list_of_values1)
    return list_of_values


def shift(matrix):
    """ Shifts one compoenent's values by 128

    component: matrix of just one component
    returns: matrix of shifted component """

    shifted = []
    for row in range(len(matrix)):
        shifted1 = []
        for column in range(len(matrix)):
            new = matrix[row][column] - 128
            shifted1.append(new)
        shifted.append(shifted1)

    return shifted

def get_transpose(matrix):
    return np.transpose(matrix)

def matrix_multiply(matrix_a, matrix_b):
    return np.matmul(matrix_a, matrix_b)

def dct(matrix):

    transpose = get_transpose(dct_matrix)
    temp = np.matmul(dct_matrix, matrix)
    coefficient_matrix = np.matmul(temp, transpose)
    return coefficient_matrix

def quantization(matrix):

    quantized = []
    for row in range(len(matrix)):
        quantized1 = []
        for column in range(len(matrix)):
            new = int(matrix[row][column] / quantization_table[row][column])
            quantized1.append(new)
        quantized.append(quantized1)

    return quantized

def dct_process(block, component):


    block_component = get_component_value(block, component)
    # print(block_component)
    shifted_block_compoenet = shift(block_component)
    cosine_transform_coefficients = dct(shifted_block_compoenet)
    quantized_matrix = quantization(cosine_transform_coefficients)

    return quantized_matrix

def inverse_dct_process(quantized_matrix):

    inverse_quantized_matrix = inverse_quantization(quantized_matrix)
    inverse_cosine_transform_coefficients = inverse_dct(inverse_quantized_matrix)
    inverse_shifted_block_component = inverse_shift(inverse_cosine_transform_coefficients)
    inverse_shifted_block_component = round_func(inverse_shifted_block_component)

    return inverse_shifted_block_component



def inverse_dct(matrix):

    transpose = get_transpose(dct_matrix)
    temp = np.matmul(transpose, matrix)
    inverse_coefficient_matrix = np.matmul(temp, dct_matrix)

    return inverse_coefficient_matrix


def inverse_quantization(matrix):

    iquantized = []
    for row in range(len(matrix)):
        iquantized1 = []
        for column in range(len(matrix)):
            new = int(matrix[row][column] * quantization_table[row][column])
            iquantized1.append(new)
        iquantized.append(iquantized1)

    return iquantized

def inverse_shift(matrix):
    ishifted = []
    for row in range(len(matrix)):
        ishifted1 = []
        for column in range(len(matrix)):
            new = matrix[row][column] + 128
            ishifted1.append(new)
        ishifted.append(ishifted1)

    return ishifted

def round_func(matrix):

    for row in range(len(matrix)):
        for column in range(len(matrix)):
            matrix[row][column] = int(round(matrix[row][column]))

    return matrix

def array_to_image(array):

    img = PIL.Image.fromarray(array)
    return img

def convert_YCbCr_to_RGB(matrix):
    """ Converts image from RGB colorspace to YCbCr colorspace

    filepath: string path to image file
    returns: image in YCC_Scale """

    RGB_scale = cv2.cvtColor(matrix, cv2.COLOR_YCR_CB2RGB)
    # bgr_img = img[...,::-1]
    # YCC_scale = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCR_CB)

    return RGB_scale


if __name__=="__main__":

    y_component_position = 0
    cb_component_position = 1
    cr_component_position = 2
    image_filepath = '/home/anna/JPEGcompression/flower.jpg'
    YCC_image = convert_RGB_to_YCbCr(image_filepath)
    all_blocks = partition(YCC_image)
    test_block = all_blocks[0]


    y_component = dct_process(test_block, y_component_position)
    y_component = inverse_dct_process(y_component)
    # print(y_component)

    cb_component = dct_process(test_block, cb_component_position)
    cb_component = inverse_dct_process(cb_component)
    # print(cb_component)

    cr_component = dct_process(test_block, cr_component_position)
    cr_component = inverse_dct_process(cr_component)
    # print(cr_component)

    inverse_partition = inverse_partition(y_component, cb_component, cr_component)
    print(inverse_partition)

    RGB = convert_YCbCr_to_RGB(inverse_partition)
    print(RGB)
