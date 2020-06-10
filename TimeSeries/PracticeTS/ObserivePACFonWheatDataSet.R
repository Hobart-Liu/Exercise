setwd('~/exercise/TimeSeries/PracticeTs/')
rm(list=ls(all=TRUE))
while (dev.cur()>1) dev.off()
beveridge = read.table('dataset/beveridge_wheat.txt')
beveridge.ts= ts(beveridge[,2], start=1500)
beveridge.MA = filter(beveridge.ts, rep(1/31, 31), sides = 2)
# rep解释：
# > rep(1,5)
# [1] 1 1 1 1 1
# filter 解释:
# filter(x, filter, method = c("convolution", "recursive"),
#        sides = 2, circular = FALSE, init)
# 
# 一个反向的时间顺序滤波器系数向量
# filter=c(1,2,3)的话，意味着t的系数为1，t-1的系数为2, t-3的系数为3
# 1. method = "convolution"
#   平均移动，即系数主要作用于输入序列x  
#   a. sides=1: 只考虑输入序列x在时间t之前的数据
#   b. sides=2: 以x的输入时间t为中心的，考虑前后
#               如果filter的长度不是奇数的话，过滤器优先考虑向前的数据
# 2. method = "recursive"
#   自回归模式，即系数主要作用于自身的历史数据 
#   此时sides取值无效
par(mfrow=c(4,1))
plot( beveridge.ts, ylab="price", main="Beveridge Wheat Price Data")
lines(beveridge.MA, col='red')
Y =  beveridge.ts/beveridge.MA
plot( Y, ylab="scaled price", main="Transformed Beveridge Wheat Price Data")
acf(na.omit(Y),main="Autocorrelation Function of Transformed Beveridge Data")
acf(na.omit(Y), type="partial", main="Partial Autocorrelation Function of Transformed Beveridge Data")

# We believe we have AR(p) model for transformed Beveridge Wheat Price
# Let's compute the coefficient of AR(p), search max order less than 5
print(ar(na.omit(Y), order.max = 5))

# An autoregressiveprocess of order p, 
# an AR(p), has a PACF that cuts off after plags.

