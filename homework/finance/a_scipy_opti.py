import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as spo

def f(x):
    y = (x-1.5)**2 + 0.5
    print("X={}, Y={}".format(x, y))
    return y


xguess = 2.0
min_result = spo.minimize(fun=f, x0=2.0, method='SLSQP', options={'disp':True})
print("minima found at X={}, Y={}".format(min_result.x, min_result.fun))