import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.integrate import quad
from scipy.stats import norm, gaussian_kde
from scipy.misc import derivative


def integral(f, a, b, point_count=1000):
    # a = float(a)
    # b = float(b)
    step = (b - a) / point_count
    sum = reduce(lambda s, x: s + f(x) * step,
                 [a + step * i for i in range(point_count)],
                 0)
    # print b
    return sum


def norm_K(f1, f2, x_array):
    return np.amax(np.absolute(f1(x_array) - f2(x_array)))


def norm_L1(f1, f2, x_array):
    return quad(lambda x: abs(f1(x) - f2(x)), x_array[0], x_array[-1])


def norm_L2(f1, f2, x_array):
    return quad(lambda x: (f1(x) - f2(x)) ** 2, x_array[0], x_array[-1])



class NormDistribution(object):
    def __init__(self, mean, dispersion, color=None):
        self.mean = mean
        self.std_deav = np.sqrt(dispersion)
        # self.std_deav = dispersion
        self.color = color
    
    def __str__(self):
        return "N(%s,%s)" % (int(self.mean), int(self.std_deav ** 2))
    
    def get_color(self):
        return self.color
    
    def cdf(self, x_arrange):
        return norm(self.mean, self.std_deav).cdf(x_arrange)
    
    def pdf(self, x_arrange):
        return norm(self.mean, self.std_deav).pdf(x_arrange)
    
    def gen_sample(self, size=1000):
        # return np.random.normal(self.mean, self.std_deav, size)
        return np.array([random.normalvariate(self.mean, self.std_deav) for i in np.arange(size)])
    
    def ecdf_in_point(self, x, sample):
        return reduce(lambda sum, z: sum + 1 if z < x else sum, sample, 0) / float(len(sample))
    
    def cdf_in_point(self, x):
        return norm(self.mean, self.std_deav).cdf(x)
    
    def _pdf_in_point(self, x):
        return norm(self.mean, self.std_deav).pdf(x)
    
    def epdf_in_point(self, x0, sample):
        return derivative(
            func=lambda x: self.ecdf_in_point(x, sample),
            x0=x0
        )
    
    def ecdf(self, x_array, sample=None, size=1000):
        if sample is None:
            sample = self.gen_sample(size)
            # sample = np.array([random.normalvariate(self.mean, self.std_deav) for i in range(n)])
        return np.array(map(lambda x: self.ecdf_in_point(x, sample), x_array))
    
    def epdf(self, x_array, sample=None, size=1000):
        if sample is None:
            sample = self.gen_sample(size)
        return np.array([
            self.epdf_in_point(xi, sample)
            for xi in x_array
        ])
    #
    # def gauss_kernel(self, x):
    #     # return lambda x: gaussian_kde(sample)(x)[0]
    #     return 1 / np.sqrt(2 * np.pi) * np.exp(- 0.5 * x * x)

    def gauss_kernel(self, sample):
        return lambda x: gaussian_kde(sample)(x)

    def evaluate_pdf(self, sample, h=0.5, kernel=None):
        if kernel is None:
            kernel = self.gauss_kernel
        return lambda x: sum([kernel((x - xi) / h) for xi in sample]) / len(sample) / h




#
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.stats import norm
# from scipy.integrate import quad
# from norm_lib import NormDistribution, integral
#
# X_MIN, X_MAX = float(-20), float(30)
# X_GRID_STEP = (X_MAX - X_MIN) / 15.0
# X_GRID = np.arange(X_MIN, X_MAX, X_GRID_STEP)
# Y_MIN, Y_MAX = float(.0), float(1.)
# Y_GRID_STEP = (Y_MAX - Y_MIN) / 10.0
# Y_GRID = np.arange(Y_MIN - Y_GRID_STEP, Y_MAX + Y_GRID_STEP, Y_GRID_STEP)
#
# X_FUNC_STEP = .1
# X_FUNC = np.arange(X_MIN, X_MAX, X_FUNC_STEP)
# IMG_X_SIZE, IMG_Y_SIZE = 15, 5
# #
# # n = 0
# # def count_f(x):
# #     global n
# #     n += 1
# #     return kernel_pdf(x)
#
# # print "integral = %s" % quad(lambda x: x ** 2, -5., 5.)
# fig = plt.figure(figsize=(IMG_X_SIZE, IMG_Y_SIZE))
# for (median, dispersion) in [(0, 1),]:
#     # print "ok"
#     # continue
# # for median, dispersion in [(0, 1), (5, 1), (0, 25)]:
#     norm_distribution = NormDistribution(median, dispersion, None)
#     sample = norm_distribution.gen_sample(1000)
#     kernel_pdf = norm_distribution.gauss_kernel(sample)
#     pdf_array = kernel_pdf(X_FUNC)
#     # print "pdf_array"
#
#     kernel_cdf = lambda x: quad(kernel_pdf, X_MIN, x)
#     # pdf_array = np.array([kernel_pdf(x) for x in X_FUNC])
#     # print "pdf = %s" % pdf_array
#     # cdf_array = np.array([kernel_cdf(x) for x in X_FUNC])
#
#     vectorized_function = np.vectorize(kernel_cdf)
#
#     cdf_array = vectorized_function(X_FUNC)
#     #
#     #
#     # cdf_array = kernel_cdf(X_FUNC)
#     print "cdf_array"
