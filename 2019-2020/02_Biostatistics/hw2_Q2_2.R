##################################
#    eBay Selling Prices (2)     #
##################################
set.seed(2020)

prices = c(212,249,250,240,210,234,195,199,222,213,233,251)
prices.sum = sum(prices)
N = length(prices)

tot.draws <- 10000 # num_of samples

miu.init = 0; sigma.init = 1
miu.vec = c(miu.init, rep(NULL, times = tot.draws))
sigma.vec = c(sigma.init, rep(NULL, times = tot.draws))

for (j in 2:(tot.draws+1)){
  miu.vec[j] = rnorm(n=1, mean=prices.sum/N, sd=sigma.vec[j-1]/sqrt(N))
  inv_sigma2 = rgamma(n=1, shape=N/2, rate=sum((prices - miu.vec[j-1])^2)/2)
  sigma.vec[j] = 1/sqrt(inv_sigma2)
}

# Remove initial values:
miu.vec = miu.vec[-1]
sigma.vec = sigma.vec[-1]

# remove first 2000 sampled values as "burn-in":
miu.post = miu.vec[-(1:2000)]
sigma.post = sigma.vec[-(1:2000)]

## Posterior summary for miu:
plot(density(miu.post))  # plot of estimated posterior for theta
mean(miu.post)  # Posterior mean of the other 8000 sampled theta values
quantile(miu.post, probs=c(0.025,0.975) )  # approximate 95% quantile-based interval for theta



