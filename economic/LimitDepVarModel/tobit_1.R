library(censReg)
library(ggplot2)
N = 500
x = rnorm(N, 3, 1)
epsi = rnorm(N, 0, 1.5)
y = 2*x -5 + epsi
y_obs = pmax(y, 0)

df = data.frame(x, y, y_obs)
mask <- df$y_obs > 0 
lm_model <- lm(df$y[mask]~df$x[mask])

cens_model <- censReg(df$y_obs ~ df$x)

g <- ggplot(data=df, mapping=aes(x=x, y=y))
print(
  ##g+geom_point(aes(y=y_obs), col='blue', size=3.5) +
  g+geom_point(aes(y=y), col='red', size=1) +
  geom_abline(slope=2, intercept=-5, col='orange', size=1.5)
  
)

print(
  g+geom_point(aes(y=y_obs), col='blue', size=3.5) +
  geom_point(aes(y=y), col='red', size=1) +
  geom_abline(slope=2, intercept=-5, col='orange', size=1.5)
  
)



g <- ggplot(data=df, mapping=aes(x=x, y=y))
print(
  g+geom_point(aes(y=y_obs), col='blue', size=3.5) +
  geom_point(aes(y=y), col='red', size=1) +
  geom_abline(slope=lm_model$coefficients[2], intercept = lm_model$coefficients[1], col='white', size=1.5) +
  geom_abline(slope=cens_model$estimate[2], intercept = cens_model$estimate[1], col='green', size=1.5) +
  geom_abline(slope=2, intercept=-5, col='orange', size=1.5) 

)

