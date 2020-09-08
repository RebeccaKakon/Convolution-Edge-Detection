from ex2_utils import *
from gamma import gammaDisplay
import numpy as np
import matplotlib.pyplot as plt
import time


def main():
    #a not befor you exam the cod, in some of the fuction the kernel size is given as an array .. and just a few days ago you said
    #it can be as int,, its not a big deal but we worte it as we expect an array .. of curse it would have been better if would have known
    #you want an int from the beggining ..

    # print("ID:", myID())
    img_path = 'C:/Users/User/Desktop/vision/kkkkkkk.jpeg'
    img = imReadAndConvert(img_path, LOAD_GRAY_SCALE)
    #cirr.jpg // circles
    #boxman.jpg
    #kkkkkkk.jpeg #lenna
    #sss.jpeg //sobel
    #codeMonkey.jpeg

    # ///////////// blurImage1 ////////////
    # the puctur for this fanction : kkkkkkk.jpeg
    testblure1 = blurImage1(img, (15, 15))
    print("blure1 : ", testblure1)
    plt.gray()
    plt.imshow(testblure1)
    plt.show()

    # ///////////// blurImage2 ////////////
    # the puctur for this fanction : kkkkkkk.jpeg
    testblur2 = blurImage2(img, (15, 15))
    print("blure2 : ", testblur2)
    plt.gray()
    plt.imshow(testblur2)
    plt.show()

    # ///////////// #edgeDetectionCanny ////////////
    #for this cose use the box img : boxman.jpg
    cannyImg, cvimg = edgeDetectionCanny(img, 0.5, 0.23390)
    plt.gray()
    plt.imshow(cannyImg)
    plt.show()
    plt.gray()
    plt.imshow(cvimg)
    plt.show()

    #for this cose use the lenna img : kkkkkkk.jpeg
    cannyImg, cvimg  = edgeDetectionCanny(img, 0.5, 0.699)
    plt.gray()
    plt.imshow(cannyImg)
    plt.show()
    plt.gray()
    plt.imshow(cvimg)
    plt.show()

    # ///////////// edgeDetectionZeroCrossingSimple ////////////
    # the pictur for this cod : codeMonkey.jpeg
    img2 = edgeDetectionZeroCrossingSimple(img)
    plt.gray()
    plt.imshow(img2)
    plt.show()

    # ///////////// edgeDetectionZeroCrossingLOG ////////////
    # the pictur for this cod : boxman.jpg
    img3 = edgeDetectionZeroCrossingLOG(img)
    plt.gray()
    plt.imshow(img3)
    plt.show()

    # ///////////// edgeDetectionSobel ////////////
    #the pictur for here : sss.jpeg
    imggg, img2 = edgeDetectionSobel(img, 0.7)
    plt.gray()
    plt.imshow(imggg)
    plt.show()
    plt.gray()
    plt.imshow(img2)
    plt.show()



    # ///////////// convDerivative ////////////
    testder, mag, x, y = convDerivative(img)


    # ///////////// #houghCircle ////////////
    #the pictur here : cirr.jpg
    testcircle = houghCircle(img, 12, 17)
    plt.gray()
    plt.imshow(testcircle)
    plt.show()


    # ///////////// conv1D ////////////
    kernell = [0, 2, 0]
    f = [1, 2, 3, 4, 5]
    test= conv1D(f,kernell)
    print("test conv1D::", test)

    # ///////////// conv2D ////////////
    kernel = np.array([[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]])
    test2D = conv2D(img, kernel)
    print("test conv2D" ,test2D)









if __name__ == '__main__':
    main()
