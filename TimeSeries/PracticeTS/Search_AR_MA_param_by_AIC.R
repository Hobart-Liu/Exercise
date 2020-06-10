rm(list=ls(all=TRUE))

# We will use dataset "discoveries"
# Yearly numbers of important Discoveries

plot(discoveries, main="Number of Major Scientific Discoveries in a Year")
stripchart(discoveries, method="stack", offset = 0.5, at=.15, pch=19,
           main = "Number of Discoveries Dotplot",
           xlab = "Number of Major Scientific Discoveries in a Year", 
           ylab = "Frequency")

par(mfcol=c(2,1))
acf(discoveries, main="ACF of Number of Major Scientific Discoveries in a Year")
acf(discoveries, type="partial", main="PACF of Number of Major Scientific Discoveries in a Year")


# Now, this is challenge
# both ACF and PACF cuts off after some lags. 
# We know how to judge AR and MA, but how to handle mixed AR/MA
#         AR(p)                   MA(q)                    ARMA(p, q)
# -----+------------------------+------------------------+------------------
# ACF  | Tails off              | Cuts off after lag q   | Tails off
# PACF | Cuts off after lag p   | Tails off              | Tails off
#

# our proposal is to use AIC to search for right parameters

cat("c=(0, 0, 1) , AIC =", AIC(arima(discoveries, order=c(0,0,1))), "\n")
cat("c=(0, 0, 2) , AIC =",AIC(arima(discoveries, order=c(0,0,2))), "\n")
cat("c=(0, 0, 3) , AIC =",AIC(arima(discoveries, order=c(0,0,3))), "\n")
cat("c=(1, 0, 0) , AIC =",AIC(arima(discoveries, order=c(1,0,0))), "\n")
cat("c=(1, 0, 1) , AIC =",AIC(arima(discoveries, order=c(1,0,1))), "\n")
cat("c=(1, 0, 2) , AIC =",AIC(arima(discoveries, order=c(1,0,2))), "\n")
cat("c=(1, 0, 3) , AIC =",AIC(arima(discoveries, order=c(1,0,3))), "\n")
cat("c=(2, 0, 0) , AIC =",AIC(arima(discoveries, order=c(2,0,0))), "\n")
cat("c=(2, 0, 1) , AIC =",AIC(arima(discoveries, order=c(2,0,1))), "\n")
cat("c=(2, 0, 2) , AIC =",AIC(arima(discoveries, order=c(2,0,2))), "\n")
cat("c=(2, 0, 3) , AIC =",AIC(arima(discoveries, order=c(2,0,3))), "\n")
cat("c=(3, 0, 0) , AIC =",AIC(arima(discoveries, order=c(3,0,0))), "\n")
cat("c=(3, 0, 1) , AIC =",AIC(arima(discoveries, order=c(3,0,1))), "\n")
cat("c=(3, 0, 2) , AIC =",AIC(arima(discoveries, order=c(3,0,2))), "\n")
cat("c=(3, 0, 3) , AIC =",AIC(arima(discoveries, order=c(3,0,3))), "\n")

# we will see c=(1,0,1) & c=(3, 0, 2) gives smallest value, to simplify the model, we choose 1,0,1
# c=(0, 0, 1) , AIC = 445.5895 
# c=(0, 0, 2) , AIC = 444.6742 
# c=(0, 0, 3) , AIC = 441.323 
# c=(1, 0, 0) , AIC = 443.3792 
# c=(1, 0, 1) , AIC = 440.198 
# c=(1, 0, 2) , AIC = 442.0428 
# c=(1, 0, 3) , AIC = 442.6747 
# c=(2, 0, 0) , AIC = 441.6155 
# c=(2, 0, 1) , AIC = 442.0722 
# c=(2, 0, 2) , AIC = 443.7021 
# c=(2, 0, 3) , AIC = 441.6594 
# c=(3, 0, 0) , AIC = 441.5658 
# c=(3, 0, 1) , AIC = 443.5655 
# c=(3, 0, 2) , AIC = 439.9263 
# c=(3, 0, 3) , AIC = 441.2941 
