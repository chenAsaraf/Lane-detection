import cv2
import numpy as np

# otsu's method: http://web-ext.u-aizu.ac.jp/course/bmclass/documents/otsu1979.pdf
# https://en.wikipedia.org/wiki/Otsu's_method
# a good explaination
# http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
# a paper that explains Otsu's method and helps explain n-levels of thresholding
# http://www.iis.sinica.edu.tw/page/jise/2001/200109_01.pdf


class OtsuThresholdMethod(object):

    def __init__(self, im, speedup=1):
        """ initializes the Otsu method to argument image. Image is only analyzed as greyscale.
        since it assumes that image is greyscale, it will choose blue channel to analyze. MAKE SURE YOU PASS GREYSCALE
        choosing a bins # that's smaller than 256 will help speed up the image generation,
        BUT it will also mean a little more inaccurate tone-mapping.
        You'll have to rescale the passed thresholds up to 256 in order to accurately map colors
        """
        if not (im.max() <= 255 and im.min() >= 0):
            raise ValueError('image needs to be scaled 0-255, AND dtype=uint8')
        images = [im]
        channels = [0]
        mask = None
        bins = 256 / speedup
        if int(bins) != bins:
            raise ValueError('speedup must be integer that evenly divides 256')
        bins = int(bins)
        self.speedup = speedup  # you are binning using speedup; remember to scale up threshold by speedup
        self.L = bins  # L = number of intensity levels
        bins = [bins]
        ranges = [0, 256]  # range of pixel values. I've tried setting this to im.min() and im.max() but I get errors...
        self.hist = cv2.calcHist(images, channels, mask, bins, ranges)
        self.N = float(sum(self.hist[:]))
        self.probabilityLevels = []
        for i in range(self.L):
            p_i = self.hist[i] / self.N
            self.probabilityLevels.append(p_i)

        self.probabilityLevels = [self.hist[i] / self.N for i in range(self.L)]  # percentage of pixels at each intensity level i
                                                                                                               # => P_i
        s = 0.0
        self.omegas = []  # sum of probability levels up to k
        for i in range(self.L):
            s += float(self.probabilityLevels[i])
            self.omegas.append(s)
        self.meanLevels = [i * self.hist[i] / self.N for i in range(self.L)]  # mean level of pixels at intensity level i
                                                                                                          # => i * P_i
        s = 0.0
        self.mus = []
        for i in range(self.L):
            s += float(self.meanLevels[i])
            self.mus.append(s)
        self.muT = s
        self.totalMeanLevel = sum(self.meanLevels)
        self.classVariances = [self.variance_at_threshold(k) for k in range(self.L)]  # sigmaB for each threshold level 0- L

    def calculate_n_thresholds(self, n):
        shape = [self.L for i in range(n)]
        sigmaBspace = np.zeros(shape)
        thresholdGen = self.dimensionless_thresholds_generator(n)
        for kThresholds in thresholdGen:
            thresholds = [0] + kThresholds + [self.L - 1]
            thresholdSpace = tuple(kThresholds)  # accessing a numpy array using the list gives us an array, rather than a point like we want
            sigmaBspace[thresholdSpace] = self.between_classes_variance_given_thresholds(thresholds)
        maxSigma = sigmaBspace.max()
        bestSigmaSpace = sigmaBspace == maxSigma
        locationOfBestThresholds = np.nonzero(bestSigmaSpace)
        coordinates = np.transpose(locationOfBestThresholds)
        return list(coordinates[0] * self.speedup)

    def dimensionless_thresholds_generator(self, n, minimumThreshold=0):
        """ generates thresholds in a list. Due to its recursive nature, this
        will fail only if self.L > 1024 (which won't happen normally) """
        if n == 1:
            for threshold in range(minimumThreshold, self.L):
                yield [threshold]
        elif n > 1:
            m = n - 1  # number of additional thresholds
            for threshold in range(minimumThreshold, self.L - m):
                moreThresholds = self.dimensionless_thresholds_generator(n - 1, threshold + 1)
                for otherThresholds in moreThresholds:
                    allThresholds = [threshold] + otherThresholds
                    yield allThresholds
        else:
            raise ValueError('# of dimensions should be > 0:' + str(n))

    def between_classes_variance_given_thresholds(self, thresholds):
        numClasses = len(thresholds) - 1
        sigma = 0
        for i in range(numClasses):
            k1 = thresholds[i]
            k2 = thresholds[i+1]
            sigma += self.between_thresholds_variance(k1, k2)
        return sigma

    def between_thresholds_variance(self, k1, k2):
        """ to be used in calculating between class variance only! """
        omega = self.omegas[k2] - self.omegas[k1]
        mu = self.mus[k2] - self.mus[k1]
        muT = self.muT
        return omega * ( (mu - muT)**2)

    def variance_at_threshold(self, k):
        """ works for a single threshold value k """
        omega = self.probability_at_threshold(k)  # omega(K)
        mu = self.mean_level_at_threshold(k)  # mu(k)
        numerator = (self.totalMeanLevel * omega - mu)**2
        denominator = omega * (1 - omega)
        if denominator == 0:
            return 0
        return numerator / denominator

    def probability_at_threshold(self, k):
        """ return sum percentage of all pixels at and below threshold level k.
        == omega == w(k)
        """
        return sum(self.probabilityLevels[:k+1])

    def mean_level_at_threshold(self, k):

        return sum(self.meanLevels[:k+1])
