from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6,7,8], dtype = np.float64)
# ys = np.array([3,5,2,1,9,5,6,0], dtype = np.float64)

def create_dataset(num, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(num):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(40, 5, 2, 'pos')

def thetas(xs, ys):
    m = ( (mean(xs)*mean(ys)) - (mean(xs*ys)) ) / ( (mean(xs)**2) - (mean(xs**2)) )
    b = mean(ys) - m*mean(xs)
    return m,b

def squared_error(ys_orig, ys_line) :
    return sum((ys_line - ys_orig)**2)

def coeff_det(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for i in ys_orig]
    se_y_line = squared_error(ys_orig, ys_line)
    se_y_mean_line = squared_error(ys_orig, y_mean_line)
    return 1 - (se_y_line / se_y_mean_line)

m,b = thetas(xs, ys)
best_fit_line = [(m*x)+b for x in xs]

predict_x = 10
predict_y = (m*predict_x) + b

rs = coeff_det(ys, best_fit_line)
print(rs)

plt.scatter(xs, ys)
plt.scatter(predict_x, predict_y, color='r')
plt.plot(xs, best_fit_line)
plt.show()
