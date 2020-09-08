"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::      *******EX2 :)
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import sys
from copy import deepcopy
from math import atan2
import scipy.ndimage as nd
from math import sqrt, pi, cos, sin, atan2
from PIL import Image, ImageDraw
from collections import defaultdict
# from canny import edgeDetectionCanny
from collections import defaultdict
from typing import List
import imageio
import numpy as np
import matplotlib.pyplot as plt
import cv2
# from scipy import misc
from PIL import Image
from typing import List
from imageio import imread

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
import cv2
import numpy as np

import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 204901417


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = imread(filename)

    img_fl = img.astype(np.float64)

    if np.max(img_fl) > 1:
        img_fl /= 255  # normalization

    if representation == 1:
        img_fl = rgb2gray(img_fl)

    return img_fl


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """

    img = imReadAndConvert(filename, representation)

    plt.imshow(img, cmap=plt.cm.gray)

    plt.axis('off')

    plt.show()



def conv1D(inSignal: np.ndarray, kernel1: np.ndarray) -> np.ndarray:
    """
 Convolve a 1-D array with a given kernel
 :param inSignal: 1-D array
 :param kernel1: 1-D array as a kernel
 :return: The convolved array
  """
    if len(inSignal) < len(kernel1):  #if bigger replace
        temp = inSignal
        inSignal = kernel1
        kernel1 = temp

    sizeSignal = len(inSignal)
    sizeKernel = len(kernel1)

    k = sizeKernel - 2
    if sizeKernel == 2: k = k + 1

    result = np.zeros(sizeSignal)  #filling the array with zeros

    for i in range(0, sizeSignal):
        sums = 0 #restart the sam every loop
        for j in range(0, sizeKernel):
            # print("i: ",i)
            if i - k + j >= 0 and i - k + j < sizeSignal:
                if i == 1:
                    print('inSignal : %d kernel1 : %d', inSignal[i - k + j], kernel1[sizeKernel - 1 - j])
                sums += inSignal[i - k + j] * kernel1[sizeKernel - 1 - j] #the convolution
        result[i] = sums

    return result


def conv2D(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
 Convolve a 2-D array with a given kernel
 :param inImage: 2D image
 :param kernel2: A kernel
 1
 :return: The convolved image
 """
    m, n = kernel2.shape
    #print("conv2D    m: ", m, "n: ", n)
    # new_image = inImage
    y, x = inImage.shape
    # y = y - m + 1
    # x = x - n + 1
    image_padded = np.pad(inImage, (m // 2, n // 2), 'edge') #pedding the image
    new_image = np.zeros((y, x))  #full by zero
    for i in range(y):
        for j in range(x):
            new_image[i][j] = np.sum(image_padded[i:i + m, j:j + n] * kernel2) #the conv

    return new_image



def convDerivative(inImage: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
Calculate gradient of an image
:param inImage: Grayscale iamge
:return: (directions, magnitude,x_der,y_der)
 """
    ker1 = np.array([[-1, 0, 1]])
    ker2 = np.array([[-1], [0], [1]])
    Ix_n = cv2.filter2D(inImage, -1, ker1)  #passing the kernel f' in x
    Iy_n = cv2.filter2D(inImage, -1, ker2)  #passing the kernel f' in y

    # m, n = Ix_n.shape
    # m, n = Iy_n.shape

    mag1 = (Ix_n * Ix_n) + (Iy_n * Iy_n)
    mag = np.sqrt(mag1)  #finish calculate the mag
    # mag = (mag / np.max(mag)) * 255
    der = np.arctan(Iy_n / Ix_n)  #calculate the der

    # print("mag:   ", mag)
    # plt.imshow(mag)
    # plt.show()

    return der, mag, Ix_n, Iy_n


def gaussianKernel(kernel_size: np.ndarray, sigma) -> np.ndarray:
    # filter_size = 2 * int(4 * sigma + 0.5) + 1
    filter_size = kernel_size[0]
    gaussian_filter = np.zeros((filter_size, filter_size), np.float32)
    m = filter_size // 2
    n = filter_size // 2

    for x in range(-m, m + 1):     #the gaussian calculation
        for y in range(-n, n + 1):
            x1 = 2 * np.pi * (sigma ** 2)
            x2 = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            gaussian_filter[x + m, y + n] = (1 / x1) * x2
    return gaussian_filter


def blurImage1(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    row = kernel_size[0]
    sigma = 0.3 * ((row - 1) * 0.5 - 1) + 0.8
    kernel = gaussianKernel(kernel_size, sigma)    #sending to get our kernel to do the conv

    blurImage = conv2D(in_image , kernel)

    return blurImage



def blurImage2(in_image: np.ndarray, kernel_size: np.ndarray) -> np.ndarray:
    row = kernel_size[0]
    sigma = 0.3 * ((row - 1) * 0.5 - 1) + 0.8
    blurImage = cv2.GaussianBlur(in_image, kernel_size, sigma) #you told me i can use this fanctoin of GaussianBlur:)
    return blurImage


def edgeDetectionSobel(img: np.ndarray, thresh: float = 0.7) -> (np.ndarray, np.ndarray):
    # CV implementation
    h = cv2.Sobel(img, -1, 0, 1, thresh)
    v = cv2.Sobel(img, -1, 1, 0, thresh)
    magCv = cv2.magnitude(h, v)

    #our sobel
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])     #f' in x and smooth in y
    Gy = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])    ##f' in y and smooth in x

    Gcx = conv2D(img, Gx)
    Gcy = conv2D(img, Gy)

    rs = (Gcx * Gcx) + (Gcy * Gcy)
    rs = np.sqrt(rs)   #the mag
    rs[rs > thresh] = 1    #filter by the tresh
    return magCv, rs


def edgeDetectionZeroCrossingSimple(img: np.ndarray) -> (np.ndarray):
    # CV implementation
    lap = cv2.Laplacian(img, cv2.CV_64F)   #f"
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    # lap = conv2D(img, kernel)
    # lap = nd.gaussian_laplace(img, 2) #for testing

    thres = np.absolute(lap).mean() * 0.75
    output = sp.zeros(lap.shape)
    w = output.shape[1]
    h = output.shape[0]

    #the zero crossing
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = lap[y - 1:y + 2, x - 1:x + 2]
            p = lap[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1
    return output



def edgeDetectionZeroCrossingLOG(img: np.ndarray) -> (np.ndarray):
    # # img= img/255
    kernel_size = (19, 19)   #for the gaussian
    # blur = blurImage2(img, kernel_size)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])    #for the lap

    LoG1 = blurImage1(img, kernel_size) #gaussian
    LoG = cv2.filter2D(LoG1, -1, kernel) #laplacian
    # LoG = nd.gaussian_laplace(img, 2)
    thres = np.absolute(LoG).mean() * 0.75
    #print("thres", thres)
    output = sp.zeros(LoG.shape)
    w = output.shape[1]
    h = output.shape[0]

    #the zero croxxing
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = LoG[y - 1:y + 2, x - 1:x + 2]
            p = LoG[y, x]
            maxP = patch.max()
            minP = patch.min()
            if (p > 0):
                zeroCross = True if minP < 0 else False
            else:
                zeroCross = True if maxP > 0 else False
            if ((maxP - minP) > thres) and zeroCross:
                output[y, x] = 1
    return output


def edgeDetectionCanny(img: np.ndarray, thrs_1: float, thrs_2: float) -> (np.ndarray, np.ndarray):
    # imgblurgauusian= blurImage1(img, (5,5))
    mag, der = newSobel(img)  # step 1 from the lesson
    mag = mag / mag.max() * 255
    # der, mag, ix, iy = convDerivative(cvSobel)  # step 2+3
    derQ = non_max_suppression(mag, der)  # steps 4+5

    res, weak, strong = threshold(derQ, thrs_1, thrs_2)  # step 6
    ansImg = hysteresis(res, weak, strong)  # step 6

    imgg = img * 255
    # img1= imgg.astype(int)
    input = imgg.astype('uint8')
    highThreshold = img.max() * thrs_2
    lowThreshold = highThreshold * thrs_1
    edges = cv2.Canny(input, highThreshold * 100, lowThreshold * 1000)

    # plt.gray()
    # plt.imshow(edges)
    # plt.show()
    # plt.gray()
    # plt.imshow(ansImg)
    # plt.show()
    return ansImg, edges


def newSobel(img):
    #like the sobel we did just difrrent signature , return difffrent parametters : rs, der
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Kx = conv2D(img, Kx)
    Ky = conv2D(img, Ky)

    rs = (Kx * Kx) + (Ky * Ky)
    rs = np.sqrt(rs)
    rs[rs > 0.77] = 1  # 77 399
    der = np.arctan(Ky / Kx)

    return rs, der


def non_max_suppression(img, D):
    #Quantize the gradient directions
    #For each pixel (x,y) compare to pixels along its gradient direction.
    #If |G(x,y)| is not a local maximum, set it to zero.

    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                neiber1 = 255
                neiber2 = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    neiber1 = img[i, j + 1]
                    neiber2 = img[i, j - 1]
                # angle 45
                elif (22.5 <= angle[i, j] < 67.5):
                    neiber1 = img[i + 1, j - 1]
                    neiber2 = img[i - 1, j + 1]
                # angle 90
                elif (67.5 <= angle[i, j] < 112.5):
                    neiber1 = img[i + 1, j]
                    neiber2 = img[i - 1, j]
                # angle 135
                elif (112.5 <= angle[i, j] < 157.5):
                    neiber1 = img[i - 1, j - 1]
                    neiber2 = img[i + 1, j + 1]

                if (img[i, j] >= neiber1) and (img[i, j] >= neiber2):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass
    return Z


def threshold(img, lowThresholdRatio, highThresholdRatio):  # 255/3 , 255
    #the firs filttering with the treshes , we check wht is biiger then the highThreshold and lowThreshold
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=255):
# Any edge above high – is a true edge (keep it)
# Any edge below low – a false edge (remove it)
# For any edge pixel that is in between, keep it only
# if it is connected to a strong edge

    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if (img[i, j] == weak):
                try:
                    # chek if i am connect to strong
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:  # it is not connecting to the strong
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def HoughCircles(input, circles, min_radius, max_radius):
    rows = input.shape[0]
    cols = input.shape[1]

    # initializing the angles to be computed
    sinang = dict()
    cosang = dict()

    # initializing the angles
    for angle in range(0, 360):
        sinang[angle] = np.sin(angle * np.pi / 180)
        cosang[angle] = np.cos(angle * np.pi / 180)


    radius = [i for i in range(min_radius, max_radius)]  #12 17 best time

    # threshold value
    for r in radius:
        # fulling with zero
        acc_cells = np.full((rows, cols), fill_value=0, dtype=np.uint64)

        # Iterating through the original image
        for x in range(rows):
            for y in range(cols):
                if input[x][y] == 255:  # edge
                    # calculate the canter of the circle
                    for angle in range(0, 360):
                        b = y - round(r * sinang[angle])
                        a = x - round(r * cosang[angle])
                        if a >= 0 and a < rows and b >= 0 and b < cols:
                            acc_cells[int(a)][int(b)] += 1  #increase by 1

        #print('For radius: ', r)
        acc_cell_max = np.amax(acc_cells)  # the max point in the arry thet have the most circle
        #print('max acc value: ', acc_cell_max)

        if (acc_cell_max > 150):

            #print("Detecting the circles for radius: ", r)

            #threshold
            temp= acc_cell_max * 0.996
            acc_cells[acc_cells < temp] = 0


            # find the circles for this radius
            for i in range(rows):
                for j in range(cols):
                    if (i > 0 and j > 0 and i < rows - 1 and j < cols - 1 and acc_cells[i][j] >= temp):
                        avg_sum = np.float32((acc_cells[i][j] + acc_cells[i - 1][j] + acc_cells[i + 1][j] +
                                              acc_cells[i][j - 1] + acc_cells[i][j + 1] + acc_cells[i - 1][j - 1] +
                                              acc_cells[i - 1][j + 1] + acc_cells[i + 1][j - 1] + acc_cells[i + 1][
                                                  j + 1]) / 9)
                        #print("Intermediate avg_sum: ", avg_sum)
                        if (avg_sum >= 33):
                            #print("For radius: ", r, "average: ", avg_sum, "\n")
                            circles.append((i, j, r))
                            acc_cells[i:i + 5, j:j + 7] = 0
    return circles


def houghCircle(img: np.ndarray, min_radius: float, max_radius: float) -> list:
    #the img for this : cirr.jpg
    #this function took 5 min.. sorry !
    orig_img = img
    input_img = deepcopy(orig_img)
    # Steps
    # 2. Detect Edges using Canny Edge Detector
    edged_image, img2 = edgeDetectionCanny(input_img, 0.5, 0.999)
    # 3. Detect Circle radius
    # 4. Perform Circle Hough Transform
    circles = []
    circless = HoughCircles(edged_image, circles, min_radius, max_radius)

    #testing my self it was on the same spot in the original pictur
    # for myy in circless:
    #     cv2.circle(orig_img, (myy[1], myy[0]), myy[2], (0, 255, 0), 1)
    #     cv2.rectangle(orig_img, (myy[1] - 2, myy[0] - 2), (myy[1] - 2, myy[0] - 2), (0, 0, 255), 3)
    #
    # print(circless)
    #
    # cv2.imshow('Circle Detected Image', orig_img)
    # cv2.imwrite('Circle_Detected_Image.jpg', orig_img)

    return circless
