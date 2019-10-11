import cv2
import math
import numpy as np
from dynamicThreshold import OtsuThresholdMethod

"""
The Canny edge detection algorithm is composed of 5 steps:
1) Noise reduction;
2) Gradient calculation;
3) Non-maximum suppression;
4) Double threshold;
5) Edge Tracking by Hysteresis.
"""

class CannyEdgeDetect:
    
    def find_edges(self, im):
        smoothed = self.smooth_image(im)
        mag, phi = self.get_gradient_magnitude_and_angle(smoothed)
        thinNormalMag = self.get_combined_thinned_image(mag, phi)
        weak = self.double_threshold(thinNormalMag)
        return weak * 255
    
    """
    step 1: Noise reduction.
    One way to get rid of the noise on the image, is by applying Gaussian blur to smooth.
    To do so, image convolution technique is applied with a Gaussian Kernel (3x3, 5x5, 7x7 etc…).
    The kernel size depends on the expected blurring effect.
    Basically, the smallest the kernel, the less visible is the blur.
    """

    def convolve(inImage: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
        """ Convolution method: convolve a 2-D array with a given kernel, mode='same' """
        inv_k = kernel2[::, ::]
        kernel_shape = np.array([x for x in kernel2.shape])
        img_shape = np.array([x for x in inImage.shape])
        out_len = np.max([kernel_shape, img_shape], axis=0)
        midKernel = kernel_shape // 2
        paddedSignal = np.pad(inImage.astype(np.float32),
                              ((kernel_shape[0], kernel_shape[0]),
                               (kernel_shape[1], kernel_shape[1]))
                              , 'edge')
        outSignal = np.ones(out_len)
        for i in range(out_len[0]):
            for j in range(out_len[1]):
                st_x = j + midKernel[1] + 1
                end_x = st_x + kernel_shape[1]
                st_y = i + midKernel[0] + 1
                end_y = st_y + kernel_shape[0]
                outSignal[i, j] = (paddedSignal[st_y:end_y, st_x:end_x] * inv_k).sum()
        return outSignal
    
    def smooth_image(self, im):
        gaussian = [2, 4, 5, 4, 2,
                    4, 9, 12, 9, 4,
                    5, 12, 15, 12, 5,
                    2, 4, 5, 4, 2,
                    4, 9, 12, 9, 4]
        gaussian = 1.0 / sum(gaussian) * np.reshape(gaussian, (5, 5))
        return convolve(im, gaussian)
    
    """
    Step 2: Gradient calculation.
    The Gradient calculation step detects the edge intensity and direction by
    calculating the gradient of the image using edge detection operators.
    Edges correspond to a change of pixels’ intensity. To detect it,
    the easiest way is to apply filters that highlight this intensity change
    in both directions: horizontal (x) and vertical (y)
    """
    
    def get_gradient_magnitude_and_angle(self, im):
        gy, gx = self.sobel_filter(im)
        mag = self.get_magnitude(gy, gx)
        phi = self.get_angle(gy, gx)
        return mag, phi
    
    # because of how gradients are calculated, a gradient in the x direction = a vertical line.
    def sobel_filter(self, im):
        y = [-1, -2, -1,
             0, 0, 0,
             +1, +2, +1]
        x = [-1, 0, +1,
             -2, 0, +2,
             -1, 0, +1]
        return self._apply_filter(im, y, x)

    def _apply_filter(self, im, y, x):
        y = np.reshape(y, (3, 3))
        x = np.reshape(x, (3, 3))
        Gy = convolve(im, y)
        Gx = convolve(im, x)
        return Gy, Gx

    def scharr_filter(self, im):
        y = [-3, -10, -3,
             0, 0, 0,
             +3, +10, +3]
        x = [-3, 0, +3
             - 10, 0, +10,
             -3, 0, +3]
        return self._apply_filter(im, y, x)
    
    def get_magnitude(self, gy, gx):
        """ calculate gradient magnitude from Gx and Gy, the gradients in x and y, respectively """
        return np.hypot(gy, gx)  # == np.sqrt(sobelX**2 + sobelY**2)

    def get_angle(self, gy, gx):
        """ calculate gradient angle. For each pixel determine direction of gradient in radians. 0 - 2pi """
        phi = np.arctan2(gy, gx)
        phi += 2 * math.pi  # because all of phi is negative for some reason
        phi %= (2 * math.pi)  # ensure that angle values are only between 0 and 2pi
        return phi
    
    """
    Step 3: Non-maximum suppression.
    Edges constriction- ideally, the final image should have thin edges.
    The algorithm goes through all the points on the gradient intensity
    matrix and finds the pixels with the maximum value in the edge directions.
    """

    def get_4_thinned_bidirectional_edges(self, mag, phi):
        """ only keep pixel if is strongest of nieghbors that point in the same direction.
        1 . compare to direction. There are 4 directions. Horizontal, Vertical, And DiagB \ and DiagF /
        2. compare to neighbors. Keep pixels that are stronger than both neighbors
        """
        shape = mag.shape
        higher, lower = np.zeros(shape), np.zeros(shape)
        toLeft, toRight = np.zeros(shape), np.zeros(shape)
        downLeft, upRight = np.zeros(shape), np.zeros(shape)
        upLeft, downRight = np.zeros(shape), np.zeros(shape)
        # ------ vertical ------- #
        higher[:-1, :] = mag[1:, :]  # shift rows up
        lower[1:, :] = mag[:-1, :]  # shift rows down
        # ------ horizontal ------- #
        toLeft[:, :-1] = mag[:, 1:]  # shift rows left
        toRight[:, 1:] = mag[:, :-1]  # shift rows right
        # ------ diagForward ------- #  /
        downLeft[1:, :-1] = mag[:-1, 1:]
        upRight[:-1, 1:] = mag[1:, :-1]
        # ------ diagBackward ------- #  \
        downRight[1:, 1:] = mag[:-1, :-1]
        upLeft[:-1, :-1] = mag[1:, 1:]
        # -------------------------------
        diagFphi, diagBphi, horizPhi, vertPhi = self.get_4_bidirectional_matrices(phi)
        thinVert = vertPhi & (mag > higher) & (mag >= lower)
        thinHoriz = horizPhi & (mag > toLeft) & (mag >= toRight)
        thinDiagF = diagFphi & (mag > downRight) & (mag >= upLeft)
        thinDiagB = diagBphi & (mag > downLeft) & (mag >= upRight)
        return [thinDiagF, thinDiagB, thinHoriz, thinVert]

    def get_4_bidirectional_matrices(self, phi):
        """determine which of the bidirectional groups to which a pixel belongs.
        note that I use the rare & , | symbols, which do boolean logic element-wise (bitwise)
        """
        phi = phi % math.pi  # take advantage of symmetry. You only need to analyze 0-pi
        pi = math.pi
        diagForward = (phi > 2 * pi / 16) & (phi < 6 * pi / 16)  # /
        diagBackward = (phi > 10 * pi / 16) & (phi < 14 * pi / 16)  # \
        horizontal = (phi <= 2 * pi / 16) | (
                    phi >= 14 * pi / 16)  # _    horizontal is only one using the | operator because it's
        # got two relevant portions
        vertical = (phi >= 6 * pi / 16) & (phi <= 10 * pi / 16)  # |
        return [diagForward, diagBackward, horizontal, vertical]
    
    def get_combined_thinned_image(self, mag, phi):
        thinDiagF, thinDiagB, thinVert, thinHoriz = self.get_4_thinned_bidirectional_edges(mag, phi)
        normalMag = self.normalize_magnitude(mag)
        thinNormalMag = np.array(normalMag * (thinDiagF + thinDiagB + thinVert + thinHoriz),
                                 dtype=np.uint8)  # convert to uint8 image format.
        return thinNormalMag
    
    def normalize_magnitude(self, mag):
        """ scales magnitude matrix back to 0 - 255 values """
        offset = mag - mag.min()  # offset mag so that minimum value is always 0
        if offset.dtype == np.uint8:
            raise
        normalized = offset * 255 / offset.max()  # now.. if this image isn't float, you're screwed
        return offset * 255 / offset.max()
    
    """
    Step 4 + 5: Double threshold + Edge Tracking by Hysteresis
    Based on the threshold results, the hysteresis consists of transforming weak
    pixels into strong ones, if and only if at least one of the pixels around the one
    being processed is a strong one.
    """

    def double_threshold(self, im):
        """ obtain two thresholds for determining weak and strong pixels. return two images, weak and strong,
        where strong contains only strong pixels, and weak contains both weak and strong
        """
        otsu = OtsuThresholdMethod(im, 4)  # speedup of 4 keeps things pretty accurate but much faster
        _, lowThresh, highThresh, tooHigh = otsu.calculate_n_thresholds(4)
        weakLines = im > lowThresh
        return weakLines

   


# ------------ TEST THE ALGORITHM: ------------ #
"""
canny = CannyEdgeDetect()
im4canny = cv2.imread("shot1.jpg")
#cv2.imshow('image', im4canny)
#cv2.waitKey(0)


im4canny = cv2.cvtColor(im4canny, cv2.COLOR_BGR2GRAY)
#cv2.imshow('grayscale', im4canny)
#cv2.waitKey(0)

# --------------------------------------------- #
# ------ Run the algorithm step-by-step ------- #
# --------------------------------------------- #

# step 1:
smoothed = canny.smooth_image(im4canny)
smoothed = smoothed.astype('uint8')
#cv2.imshow('smoothed', smoothed)
#cv2.waitKey(0)

# step 2:
gradient_img_y, gradient_img_x= canny.sobel_filter(im4canny)
gy = gradient_img_y.astype('uint8')
gx = gradient_img_x.astype('uint8')
#cv2.imshow('gradient calculation', gy)
#cv2.waitKey(0)
#cv2.imshow('gradient calculation', gx)
#cv2.waitKey(0)

# step 3:
mag = canny.get_magnitude(gradient_img_y, gradient_img_x)
phi = canny.get_angle(gradient_img_y, gradient_img_x)
thinNormalMag = canny.get_combined_thinned_image(mag, phi)
#cv2.imshow('non-maximum suppression', thinNormalMag)
#cv2.waitKey(0)

# step 4 + 5:
weak = canny.double_threshold(thinNormalMag)
weak = weak * 255
weak = weak.astype('uint8')
#cv2.imshow('double threshold', weak)
#cv2.waitKey(0)

# --------------------------------------------- #
# - activate the canny edge detector at once: - #
# --------------------------------------------- #
uncanny = canny.find_edges(im4canny)
uncanny = uncanny.astype('uint8')
cv2.imshow('canny', uncanny)
cv2.waitKey(0)
"""

