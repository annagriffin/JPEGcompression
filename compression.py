import cv2
from PIL import Image
import os.path
import numpy as np
from scipy import fftpack



quantization_table = [[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]]


def get_dimentions(filepath):
    """ Get the dimention of the image

    filepath: string path to image file
    returns: width and height"""

    image = Image.open(filepath)
    width, height = image.size
    return width, height

def convert_RGB_to_YCbCr(filepath):
    """ Converts image from RGB colorspace to YCbCr colorspace

    filepath: string path to image file
    returns: image in YCC_Scale """

    img = cv2.imread(filepath)
    YCC_scale = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # cv2.imshow('dst_rt', YCC_scale)
    # cv2.waitKey(0)
    # cv2.destoryAllWindows()
    return YCC_scale

def partition(image):
    """ Separates image into 8x8 blocks

    image: image in YCbCr scale
    returns: list of 8x8 blocks """

    width = image.shape[1]
    height = image.shape[0]
    images = []
    image = np.array(image)
    delta_x = int(width/20)
    delta_y = int(height/30)
    index_x=0
    index_y=0
    for i in range(0,20):
        for j in range(0,30):
            temp_image = image[index_y:index_y+delta_y,index_x:index_x+delta_x]
            images.append(temp_image)
            index_y += delta_y
            # cv2.imshow('pa',temp_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows
        index_y=0
        index_x += delta_x
    return images

def get_value_Y(image):
    """ Gets the Y component of an image

    image: 8x8 block
    returns: list of Y component values """

    image = np.array(image,dtype = np.integer)
    Y = []
    for row in range(len(image[0])):
        Y1 = []
        for column in range(len(image[0])):
            pixel = image[row][column][0]
            Y1.append(pixel)
        Y.append(Y1)
    return Y

def get_value_Y_list(image):
    """ Gets the Y component of an image

    image: 8x8 block
    returns: list of Y component values """

    image = np.array(image,dtype = np.integer)
    Y = []
    for row in range(len(image[0])):
        Y1 = []
        for column in range(len(image[0])):
            pixel = image[row][column][0]
            Y1.append(pixel)
        Y.extend(Y1)
    return Y

def get_value_Cb(image):
    """ Gets the Cb component of an image

    image: 8x8 block
    returns: list of Cb component values """

    image = np.array(image,dtype = np.integer)
    Cb = []
    for row in range(len(image[0][0])):
        Cb1 = []
        for column in range(len(image[0][0])):
            pixel = image[row][column][1]
            Cb1.append(pixel)
        Cb.append(Cb1)
    return Cb


def get_value_Cr(image):
    """ Gets the Cr component of an image

    image: 8x8 block
    returns: Cr component values """

    image = np.array(image,dtype = np.integer)
    Cr = []
    for row in range(len(image[0][0])):
        Cr1 = []
        for column in range(len(image[0][0])):
            pixel = image[row][column][2]
            Cr1.append(pixel)
        Cr.append(Cr1)
    return Cb

def shift_components(component):
    """ Shifts one compoenent's values by 128

    component: matrix of just one component
    returns: matrix of shifted component """

    shifted = []
    for row in range(len(component)):
        shifted1 = []
        for column in range(len(component)):
            new = component[row][column] - 128
            shifted1.append(new)
        shifted.append(shifted1)

    return shifted

def get_2D_dct(image):
    """ Using the dct module from Scipy"""
    matrix = fftpack.dct(fftpack.dct(image, norm = 'ortho'), norm = 'ortho')
    return matrix


def quantization(component_matrix):

    quantized = []
    for row in range(len(component_matrix)):
        quantized1 = []
        for column in range(len(component_matrix)):
            new = int(round(component_matrix[row][column] // quantization_table[row][column]))
            quantized1.append(new)
        quantized.append(quantized1)

    return quantized

def inverse_quantization(component_matrix):

    iquantized = []
    for row in range(len(component_matrix)):
        iquantized1 = []
        for column in range(len(component_matrix)):
            new = int(component_matrix[row][column] * quantization_table[row][column])
            iquantized1.append(new)
        iquantized.append(iquantized1)

    return iquantized


def get_2D_idct(image):
    """ Using the dct module from Scipy"""
    matrix = fftpack.idct(fftpack.idct(image, norm = 'ortho'), norm = 'ortho')
    return matrix


def inverse_shift(component):
    ishifted = []
    for row in range(len(component)):
        ishifted1 = []
        for column in range(len(component)):
            new = component[row][column] + 128
            ishifted1.append(new)
        ishifted.append(ishifted1)

    return ishifted

def matrix_to_list(component):

    longlist = []
    for row in range(len(component)):
        for column in range(len(component)):
            longlist.append(component[row][column])

    return longlist



if __name__=="__main__":

    image_filepath = '/home/anna/JPEGcompression/flower.jpg'


    image_YCbCr_scale = convert_RGB_to_YCbCr(image_filepath)
    all_blocks = partition(image_YCbCr_scale)
    first_block = all_blocks[0]
    cv2.imshow('imges[0]', all_blocks[0])
    cv2.waitKey(0)
    first_block_Y_components = get_value_Y(first_block)
    print("First block Y components: ")
    print(first_block_Y_components)
    shifted_first_block_Y = shift_components(first_block_Y_components)
    print("Shifted Y block ")
    print(shifted_first_block_Y)
    cosine_transform = get_2D_dct(shifted_first_block_Y)
    print("DCT")
    print(cosine_transform)
    quantized_Y = quantization(cosine_transform)
    print("quantized")
    print(quantized_Y)

    inverse_quantized_Y = inverse_quantization(quantized_Y)
    print("inverse quantized")
    print(inverse_quantized_Y)
    inverse_cosine_transform = get_2D_idct(inverse_quantized_Y)
    print("inverse DCT")
    print(inverse_cosine_transform)
    inverse_shifted = inverse_shift(inverse_cosine_transform)
    print("inverse shifted")
    print(inverse_shifted)
    first_block_Y_components = get_value_Y(first_block)
    print("First block Y components: ")
    print(first_block_Y_components)


    # print(matrix_to_list(inverse_shifted))
