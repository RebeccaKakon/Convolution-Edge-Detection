
our assignment includ a few function about detection by performing simple manipulations on images. Convolution & Edge Detection, 
1.
conv1D-
this function implement convolution of 1D array with a given kernel
2.
conv1D-
this function implement convolution of 2D array with a given kernel
3.
convDerivative-
this function computes the magnitude and the direction of an image gradient by derive
the image in each direction separately (rows and column) 
using simple convolution with [1, 0, -1]T and
[1, 0, -1] to get the two image derivatives.
and then use these derivative images to compute the magnitude
and direction matrix.

4.
blurImage1- this function performs image blurring using convolution between the image f and
 a Gaussian kernel g.
to get the gaussian kernel we bild a function gaussianKernel(5)
5.
gaussianKernel- in this function we got the kernel size and the sigma. and then we bild our gaussian kernel so we can do with is convolution/

6.
blurImage2-
this function performs image blurring using convolution between the image f and
a Gaussian kernel g.

this out put will give us the result of the 
implementeition by using pythons internal functions:
filter2D and GaussianBlur. (you apruved me to use this function:)

these coming functions find the edge in the imge. 
The intensity image and edgeImage is binary image of -zero/one with ones in the places the function
identifies edges. 

Each function from these 4 functions implements edge detections accroding to a different method.
7.
edgeDetectionSobel-
first we imlemented with cv function of sobel with the kernel once for Ix once for Iy . then we used cv function magnitude. 
for our implemintation we took the sobel kernel again once for Ix and Iy and with our function conv2D we did the convulution . aftter this we foud the mag and we past 
over the picture with the given treshold .
8.
edgeDetectionZeroCrossingSimple- 
as we know Derivative can help us find edges, so we do laplacian (f"), now we want to find all the places we have -1,0,1 .... this is why we have the zero crossing to 
help us with this . so aftter the ziro crossing we have our edge imge.
9.
edgeDetectionZeroCrossingLOG-
as we know Derivative can help us find edges, but we can also loos some of them . thos is why on thos function we will do something diffrent 
we will pass the gauusian kernelk to blur the imge and then er will do the laplacian . you can to it in one time by doing convolution with gaussian and laplassian.
and from her its the same as (8) we will do zero crossing 
10.
edgeDetectionCanny-
The function finds edges in the input image and marks them in the output map edges using the Canny algorithm. 
The smallest value between threshold1 and threshold2 is used for edge linking. The largest value is used to find initial segments of strong edges.
we bild few functions for help: for the treshold : threshold,  for sobel so we have the parametters we need : newSobel, for the nms: non_max_suppression, 
for the hysteresis : hysteresis 
11.
Hough Circles-

this function find Circles in an image using a Hough Transform algorithm extension.



we get the intensity image, minradius, maxradius should positive numbers and minradius < maxradius.

we use the Canny Edge detector as the edge detector. 
The functions return a list of all the circles
found, each circle will be represented by:(x,y,radius). 
Circle center x, Circle center y, Circle radius.
HoughCircles- in thos function we serch for the fircle






