library(parallel)
library(AMORE)

# load data
neg <- read.csv('/Users/kellanfluette/dev/Metabolite-Prediction/Tyrosine/negative_train.csv')
neg$label = 0
pos <- read.csv('/Users/kellanfluette/dev/Metabolite-Prediction/Tyrosine/positive_train.csv')
pos$label = 1
data <- rbind(pos,neg)
data[,names(data)!='label'] <- scale(data[,names(data)!='label']) # center data

net <- newff(
	n.neurons=c(95,95,1),
	learning.rate.global=1e-2,
	momentum.global=0.5,
	error.criterium='LMS',
	hidden.layer='custom',  # the user must manually define the f0 and f1 elements of the neurons
	output.layer='custom',
	method="BATCHgd")

# error function
net$deltaE$fname <- as.integer(3) # last one in newff.R is TAO_NAME=2
net$deltaE$f <- function(arguments) {
   prediction <- arguments[[1]]	# arg1 is the prediction
   target     <- arguments[[2]]	# arg2 is the target
   residual   <- prediction - target
   return(residual)
}

# activation function
for (ind.MLPneuron in 1:length(net$neurons)) {
	net$neurons[[ind.MLPneuron]]$f0 <- function(x) { (1 / (1 + exp(-x))) }
	net$neurons[[ind.MLPneuron]]$f1 <- function(x) { exp(x) / ((1+exp(x))^2) }
}

# train and test model
result <- train(net,data[,names(data)!='label'],data$label,error.criterium='LMS',
	report=TRUE,show.step=100,n.shows=5,n.threads=detectCores(logical=TRUE))
y <- sim(result$net,data[,names(data)!='label'])
plot(data[,names(data)!='label'],y,col="blue",pch="+")
points(data[,names(data)!='label'],data$label,col="red",pch="x")
