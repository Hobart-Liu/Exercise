rm(list=ls(all=TRUE)) # remove all variables
par(mfrow=c(3, 1))
phi.1=0.6;phi.2=0.2;data.ts = arima.sim(n=500, list(ar=c(phi.1, phi.2)))
plot(data.ts, main=paste("Autoregressive Process with phi1=", phi.1, "phi2=",phi.2))
acf(data.ts, main="Autocorrelation Function")
acf(data.ts, type="partial", main="Partial Autocorrelation Function")

# Note:
# If we run this program several times,
# while the actual time series itself changes
# from simulation to simulation, the ACF
# and PACF are relatively constant.