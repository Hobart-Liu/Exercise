rm(list=ls(all=TRUE))
while (dev.cur()>1) dev.off()
library(astsa)
my.data = rec
# plot rec
plot(rec, main="Recruitment time series", col='blue', lwd=3)

# subtract mean to get a time seriess with mean zero
ar.process = my.data - mean(my.data)

# ACF and PACF
par(mfrow=c(2, 1))
acf(ar.process, main="Recruitment", col='red', lwd=3)
pacf(ar.process, main="Recruitment", col='green', lwd=3)

# order p=2
p=2

# sample autocorrelation function r
r = NULL
r = acf(ar.process, plot=F)$acf[2:(p+1)]
cat('r=', r, '\n')

# Yule Waler 
# Rx = b

# matrix R
R = matrix(1, p, p) # matrix of dimension p by p, with entries all 1's. (actually 2x2)
# define non-diagonal entires of R, Note:注意到R(i, j)的下标差的绝对值，对应的就是r[] 的读数, 对角线是1
for(i in 1:p){
  for(j in 1:p){
    if(i!=j)
      R[i,j]=r[abs(i-j)]
  }
}

# b-column vector on the right
b=NULL
b=matrix(r,p,1)# b- column vector with no entries   [reshape]

# solve(R,b) solves Rx=b, and gives x=R^(-1)b vector
phi.hat=NULL
phi.hat=solve(R,b)[,1]
cat('phi=' ,phi.hat, '\n')



# compute noise variance

#variance estimation using Yule-Walker Estimator
c0=acf(ar.process, type='covariance', plot=F)$acf[1]
var.hat=c0*(1-sum(phi.hat*r))
cat("noise variance", var.hat, '\n')



# Last, constant term in the model
phi0.hat=mean(my.data)*(1-sum(phi.hat))


cat("Constant:", phi0.hat," Coeffcinets:", phi.hat, " and Variance:", var.hat, '\n')