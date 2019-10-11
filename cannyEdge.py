import cv2
import scipy
from scipy import signal
import math
import numpy as np
from dynamicThreshold import OtsuThresholdMethod

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


class CannyEdgeDetect:

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

    # because of how gradients are calculated, a gradient in the x direction = a vertical line.
    def sobel_filter(self, im):
        y = [-1, -2, -1,
             0, 0, 0,
             +1, +2, +1]
        x = [-1, 0, +1,
             -2, 0, +2,
             -1, 0, +1]
        return self._apply_filter(im, y, x)

    def get_gradient_magnitude_and_angle(self, im):
        gy, gx = self.sobel_filter(im)
        mag = self.get_magnitude(gy, gx)
        phi = self.get_angle(gy, gx)
        return mag, phi

    def get_magnitude(self, gy, gx):
        """ calculate gradient magnitude from Gx and Gy, the gradients in x and y, respectively """
        return np.hypot(gy, gx)  # == np.sqrt(sobelX**2 + sobelY**2)

    def get_angle(self, gy, gx):
        """ calculate gradient angle. For each pixel determine direction of gradient in radians. 0 - 2pi """
        phi = np.arctan2(gy, gx)
        phi += 2 * math.pi  # because all of phi is negative for some reason
        phi %= (2 * math.pi)  # ensure that angle values are only between 0 and 2pi
        return phi

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

    def get_2d_gaussian_filter(self, k):
        horizontalG = scipy.signal.general_gaussian(k, 1, 0.8)
        verticalG = np.reshape(horizontalG, (k, 1))
        gaussian2d = horizontalG * verticalG
        normalized = gaussian2d / gaussian2d.sum()  # so the net sum will equal 1
        return normalized

    def smooth_image(self, im):
        gaussian = [2, 4, 5, 4, 2,
                    4, 9, 12, 9, 4,
                    5, 12, 15, 12, 5,
                    2, 4, 5, 4, 2,
                    4, 9, 12, 9, 4]
        gaussian = 1.0 / sum(gaussian) * np.reshape(gaussian, (5, 5))
        return convolve(im, gaussian)

    def normalize_magnitude(self, mag):
        """ scales magnitude matrix back to 0 - 255 values """
        offset = mag - mag.min()  # offset mag so that minimum value is always 0
        if offset.dtype == np.uint8:
            raise
        normalized = offset * 255 / offset.max()  # now.. if this image isn't float, you're screwed
        return offset * 255 / offset.max()

    def get_combined_thinned_image(self, mag, phi):
        thinDiagF, thinDiagB, thinVert, thinHoriz = self.get_4_thinned_bidirectional_edges(mag, phi)
        normalMag = self.normalize_magnitude(mag)
        thinNormalMag = np.array(normalMag * (thinDiagF + thinDiagB + thinVert + thinHoriz),
                                 dtype=np.uint8)  # convert to uint8 image format.
        return thinNormalMag


    def double_threshold(self, im):
        """ obtain two thresholds for determining weak and strong pixels. return two images, weak and strong,
        where strong contains only strong pixels, and weak contains both weak and strong
        """
        otsu = OtsuThresholdMethod(im, 4)  # speedup of 4 keeps things pretty accurate but much faster
        _, lowThresh, highThresh, tooHigh = otsu.calculate_n_thresholds(4)
        weakLines = im > lowThresh
        return weakLines

    def find_edges(self, im):
        smoothed = self.smooth_image(im)
        mag, phi = self.get_gradient_magnitude_and_angle(smoothed)
        thinNormalMag = self.get_combined_thinned_image(mag, phi)
        weak = self.double_threshold(thinNormalMag)
        return weak * 255
