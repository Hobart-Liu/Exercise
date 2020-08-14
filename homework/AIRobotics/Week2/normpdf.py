#For this problem, you aren't writing any code.
#Instead, please just change the last argument 
#in f() to maximize the output.

from math import *

def NormPDF(mu, sigma2, x):
    return 1/sqrt(2.*pi*sigma2) * exp(-.5*(x-mu)**2 / sigma2)

# Combine two normal distribution into one
def update(mean1, var1, mean2, var2):
    new_mean = (var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1/(1/var1 + 1/var2)
    return [new_mean, new_var]

# predict new normal distribution by move from mean1 to (mean1+mean2)
# var2 is the information lost during the motion, like unknown moved or not. 
def predict(mean1, var1, mean2, var2):
    new_mean = mean1+mean2
    new_var = var1+var2
    return [new_mean, new_var]
    
    
    
    
measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.

#Please print out ONLY the final values of the mean
#and the variance in a list [mu, sig]. 

# Insert code here

for i in range(len(measurements)):
    [mu, sig] = update(mu, sig, measurements[i], measurement_sig)
    print "update  ", [mu, sig]
    [mu, sig] = predict(mu, sig, motion[i], motion_sig)
    print "predict ", [mu, sig]
