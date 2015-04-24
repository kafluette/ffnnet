library(RSNNS)

# load data
neg <- read.csv('/Users/kellanfluette/dev/Metabolite-Prediction/Tyrosine/negative_train.csv')
neg$label = 0
pos <- read.csv('/Users/kellanfluette/dev/Metabolite-Prediction/Tyrosine/positive_train.csv')
pos$label = 1
data <- rbind(pos,neg)
data <- data[sample(1:nrow(data),length(1:nrow(data))),1:ncol(data)] # shuffle
#data[,names(data)!='label'] <- scale(data[,names(data)!='label']) 	 # center and scale
data <- splitForTrainingAndTest(data[,names(data)!='label'],data$label,ratio=0.15)
data <- normTrainingAndTestSet(data)
model <- mlp(data$inputsTrain, data$targetsTrain, size=5, learnFuncParams=c(0.1),
             maxit=50, inputsTest=data$inputsTest, targetsTest=data$targetsTest)

# test model
summary(model)
model
weightMatrix(model)
extractNetInfo(model)
par(mfrow=c(2,2))
plotIterativeError(model)
predictions <- predict(model,data$inputsTest)
confusionMatrix(data$targetsTrain,fitted.values(model))
confusionMatrix(data$targetsTest,predictions)
#confusion matrix with 402040-method
confusionMatrix(data$targetsTrain, encodeClassLabels(fitted.values(model),
                                                     method="402040", l=0.4, h=0.6))