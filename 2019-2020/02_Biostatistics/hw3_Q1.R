#####################
#  Phase I Trials   #
#####################

library(R2WinBUGS)
setwd("C:/Users/Administrator.AM-201207261050/Documents/Rcodes")

#### Create model ###
# parameter: Xmin, theta, g_max, r_max
model <- function(){
  for (i in 1:N){  # N subjects
    # Likelihood
    Y[i] ~ dbern(p[i])
    logit(p[i]) <- (1 / (gamma - Xmin)) * 
      (gamma * logit(rho0) - Xmin* logit(theta)
       + (logit(theta) - logit(rho0)) * X[i])
  }
  
  # Prior Distribution
  gamma ~ dunif(Xmin, g_max)
  rho0 ~ dunif(0, r_max)
}

file_name = file.path(getwd(), "phaseOne.bug") # get path
write.model(model, file_name) # save model for future use


#####  Data  #####

X <- c(50, 100, 150, 200, 225, 250, 275, 300, 325, 325, 325, 350)
Y <- c(0,0,0,0,0,0,0,0,1,0,0,1)

inits0 <-  function(){ list(gamma=200, rho0=0.1) }
parameters <- c("gamma", "rho0")


### Run Tests  ###

# [1]
data <- list(N=2, X=X[0:2], Y=Y[0:2],
             Xmin=50, theta=0.25, g_max=400, r_max=0.20)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim

# [2]
data <- list(N=12, X=X, Y=Y,
             Xmin=50, theta=0.25, g_max=400, r_max=0.20)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim

# [3]
data <- list(N=2, X=X[0:2], Y=Y[0:2],
             Xmin=50, theta=0.25, g_max=650, r_max=0.20)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim

# [4]
data <- list(N=12, X=X, Y=Y,
             Xmin=50, theta=0.25, g_max=650, r_max=0.20)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim

# [5]
data <- list(N=2, X=X[0:2], Y=Y[0:2],
             Xmin=50, theta=0.25, g_max=400, r_max=0.25)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim

# [6]
data <- list(N=12, X=X, Y=Y,
             Xmin=50, theta=0.25, g_max=400, r_max=0.25)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim

# [7]
data <- list(N=2, X=X[0:2], Y=Y[0:2],
             Xmin=50, theta=0.25, g_max=650, r_max=0.25)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim

# [8]
data <- list(N=12, X=X, Y=Y,
             Xmin=50, theta=0.25, g_max=650, r_max=0.25)

posterior.sim <- bugs(data, inits0, parameters, file_name,
                      n.chains=3, n.iter=6000, n.burnin=1000)
posterior.sim





  
  