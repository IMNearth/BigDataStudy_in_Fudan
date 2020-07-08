#####################
#  Phase II Trials  #
#####################
library(extraDistr)
      
#####################
#  2.2 PP           #
#####################  
predictiveProbability <- function(X=0, n=10, N_max=41, 
                                  p0=0.4, Q_T=0.9, a0=0.3, b0=0.7){
  pp = 0
  for (y in 1:(N_max-n)){
    futureProb = dbbinom(y, N_max-n, a0+X, b0+n-X)
    indicator = 0
    if (1-pbeta(p0, a0+X+y, b0+N_max-X-y) > Q_T){
      indicator = 1
    }
    pp = pp + futureProb * indicator
  }
  list(predictiveprobability=pp)
}

for (x in 0:10){
  print(x)
  print(predictiveProbability(X=x))
}














