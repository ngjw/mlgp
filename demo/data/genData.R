# 1d data

n <- 500

start <- -10
end   <- 10

X <- seq(start,end,by=(end-start)/(n-1))

y <- sin(X) + rnorm(n,0,0.3)

plot(X,y)

Xs <- runif(30,-10,10)


write(X,"1d_training_inputs",1)
write(y,"1d_training_outputs",1)
write(Xs,"1d_test_inputs",1)


# 2d data
rm(list=ls())

nx1 <- 20
nx2 <- 20

start <- -10
end   <- 10

x1 <- seq(start,end,by=(end-start)/(nx1-1))
x2 <- seq(start,end,by=(end-start)/(nx2-1))

X1 <- rep(x1,nx2)
X2 <- rep(x2,rep(nx1,length(x2)))

X <- cbind(X1,X2)

y <- sin(rowSums(X))
y <- y + rnorm(length(y),0,sd(y)/2)

Xs <- X[order(runif(400,0,1))[1:30],]


write(t(X),"2d_training_inputs",2)
write(t(y),"2d_training_outputs",1)
write(t(Xs),"2d_test_inputs",2)
