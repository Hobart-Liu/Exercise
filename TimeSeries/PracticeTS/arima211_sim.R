phi = c(.7, .2)
beta = 0.5
sigma = 3
m = 10000

set.seed(5)
Simulated.Arima=arima.sim(n=m,list(order = c(2,1,1), ar = phi, ma=beta))
plot(Simulated.Arima, ylab=' ',main='Simulated time series from ARIMA(2,1,1) process', col='blue', lwd=2)
acf(Simulated.Arima)
Diff.Simulated.Arima=diff(Simulated.Arima)
plot(Diff.Simulated.Arima)
acf(Diff.Simulated.Arima)
pacf(Diff.Simulated.Arima)
library(astsa)
sarima(Simulated.Arima,2,1,1,0,0,0) # astsa not installed
library(forecast)
auto.arima(Simulated.Arima) # forecast not installed
fit1<-arima(Diff.Simulated.Arima, order=c(4,0,0))
fit1
fit2<-arima(Diff.Simulated.Arima, order=c(2,0,1))
fit2
fit3<-arima(Simulated.Arima, order=c(2,1,1))
fit3