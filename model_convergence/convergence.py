# https://github.com/davisking/dlib/blob/6b581d91f6b9b847a8163420630ef947e7cc88db/dlib/statistics/running_gradient.h
import numpy as np
from scipy.stats import t as student_t, norm
import math

class RunningSimpleLinearRegression:
    def __init__(self):
        self.reset()

    def reset(self):
        self.xsum = 0
        self.ysum = 0
        self.xsqr_sum = 0
        self.ysqr_sum = 0
        self.residuals_sqr_sum = 0
        self.xysum = 0
        self.nb = 0

    def update(self, x, y):
        a, b = self.slope, self.intercept
        self.xsum += x
        self.ysum += y
        self.xsqr_sum += x ** 2
        self.ysqr_sum += y ** 2
        self.xysum += x * y
        self.nb += 1
        new_a, new_b = self.slope, self.intercept
        # check: https://www.cs.tut.fi/~tabus/course/ASP/LectureNew10.pdf
        #'Recursion for MSE criterion'
        self.residuals_sqr_sum += (y - (a * x + b)) * (y - (new_a * x + new_b))

    @property
    def slope(self):
        if self.nb <= 1:
            return 0
        # based on
        # - https://en.wikipedia.org/wiki/Simple_linear_regression
        # (the expression of \hat{beta}
        # https://en.wikipedia.org/wiki/Correlation_and_dependence#Sample_correlation_coefficient
        xmean = self.xsum / self.nb
        ymean = self.ysum / self.nb
        # sxy = self.xysum / self.nb - (xmean * ymean)
        # sx = np.sqrt(self.xsqr_sum / self.nb - xmean ** 2)

        sxy = self.xysum - (self.xsum * self.ysum) / self.nb
        sx = self.xsqr_sum - self.xsum ** 2 / self.nb
        return sxy / sx

    @property
    def slope_std(self):
        # https://en.wikipedia.org/wiki/Simple_linear_regression
        # in "Normality assumption"
        if self.nb <= 2:
            return 0
        xmean = self.xsum / self.nb
        sx_sqr_sum = self.xsqr_sum - self.nb * xmean ** 2
        slope_var = self.residuals_sqr_sum / (sx_sqr_sum * (self.nb - 2))
        slope_std = np.sqrt(slope_var)
        return slope_std

    @property
    def intercept(self):
        if self.nb < 1:
            return 0
        xmean = self.xsum / self.nb
        ymean = self.ysum / self.nb
        return ymean - self.slope * xmean

    def prob_slope_less_than(self, value=0):
        mean = self.slope
        std = self.slope_std
        # return student_t(loc=mean, scale=std, df=self.nb - 2).cdf(value)
        # return norm(loc=mean, scale=std).cdf(value)
        return phi((value - mean) / (std+1e-7))
    
    def prob_slope_greater_than(self, value=0):
        return 1 - self.prob_slope_less_than(value)

def phi(x):
    #'Cumulative distribution function for the standard normal distribution'
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def count_steps_without_decrease(values, threshold=0.51, quantile=1.0):
    rslr = RunningSimpleLinearRegression()
    n = len(values)
    max_val = np.quantile(values, quantile)
    steps_without_decrease = 0
    for i, value in reversed(list(enumerate(values))):
        if value <= max_val:
            rslr.update(i, value)
        if rslr.nb >= 2 and rslr.prob_slope_less_than(0) < threshold:
            steps_without_decrease = n - i
    return steps_without_decrease


def simple_test():
    X = np.random.uniform(size=10000)
    Y = X * (-1) + 10 + np.random.normal(size=len(X)) * 10

    lsq = RunningSimpleLinearRegression()
    for i, (x, y) in enumerate(zip(X, Y)):
        lsq.update(x, y)
        print(lsq.slope, lsq.slope_std, lsq.intercept, lsq.slope_cdf(0))


if __name__ == "__main__":
    from joblib import load
    import matplotlib.pyplot as plt
    data = load("losses.pkl")
    losses =[loss for loss, lr in data]
    r = RunningSimpleLinearRegression()
    for i, loss in enumerate(losses):
        r.update(i, loss)
        if i > 0 and i % 1000 == 0:
            if r.prob_slope_less_than(0) < 0.51:
                print(i, r.slope)
            r.reset()
    plt.plot(losses)
    plt.show()
