x = NULL
x[1] = 0
for (i in 2:1000){
  x[i] = x[i-1] + rnorm(1)
}

random_walk = NULL
random_walk = ts(x)
plot(random_walk, main=' A Random Walk ', ylab='', xlab='Days', col='blue', lwd=1)


# acf 显示出强的前后相关性
acf(random_walk, main='acf')
# pacf 则无相关性特征
pacf(random_walk, main='pacf')

d <- density(random_walk)
plot(d, main='density of random walk')

acf(diff(random_walk), main='ACF after diff')
pacf(diff(random_walk), main='PACF after diff')

plot(diff(random_walk), main='ts of noise')
d1 <- density(diff(random_walk))
plot(d1, main='density of noise')