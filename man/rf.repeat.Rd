\name{rf.repeat}
\alias{rf.repeat}

\title{
Build random forest multiple times and return AUC for both training and test set if available
}
\description{
Due to the randomness of random forest, RF models can be built multiple times to get a better estimation of model performance by assessing AUC on both out-of-bag prediction of training and test predictions on test set. Work for classification only.
}
\usage{
rf.repeat(X.train, Y.train, X.test, Y.test, fea, times = 10)
}
\arguments{
  \item{X.train}{
a data frame or matrix (like x) containing predictors for the training set.
}
  \item{Y.train}{
response for the training set. If a factor, classification is assumed, otherwise regression is assumed. If omitted, will run in unsupervised mode.
}
  \item{X.test}{
a data frame or matrix (like x) containing predictors for the test set.
}
  \item{Y.test}{
response for the test set. 
}
  \item{fea}{
feature index or feature names used to train the model.
}
  \item{times}{
the number of RF models built. The more repeats, the less standard error.
}
}

\value{
return a list, including
  \item{AUC}{AUC calculated from out-of-bag prediction from random forest classification model}
  \item{Test.AUC}{AUC calculated from test prediction from random forest classification model. Only available when test set is given}
}
\references{
Guan, X., & Liu, L. (2018). Know-GRRF: Domain-Knowledge Informed Biomarker Discovery with Random Forests. 
}
\author{
Li Liu, Xin Guan
}

\examples{
##---- Example: classification  ----

set.seed(1)
X<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(0.1, 2.2, 0.2) 
##y has a linear relationship with first 10 variables
y=b[5]*X$X4+b[6]*X$X5+b[7]*X$X6+b[8]*X$X7+b[9]*X$X8+b[10]*X$X9+b[11]*X$X10 
y=as.factor(ifelse(y>0, 1, 0)) ##classification

##split training and test set
X.train=X[1:70,]
X.test=X[71:100,]
y.train=y[1:70]
y.test=y[71:100]

rf.repeat(X.train, y.train, fea=1:20)  ##no test set
rf.repeat(X.train, y.train, X.test, y.test, 1:10)  ##relevant feature set
rf.repeat(X.train, y.train, X.test, y.test, 11:20)  ##irrelevant feature set

## The function is currently defined as

function(X.train, Y.train, X.test=NULL, Y.test=NULL, fea, times=10) {
	result <- list(AUC.train=rep(NA, times), AUC.test=rep(NA, times));

	for(t in 1:times) {
	  result$AUC.train[[t]] <- rf.once(X.train, Y.train, X.test, Y.test, fea)[[1]]
	  if(!is.null(X.test))
      result$AUC.test[[t]] <- rf.once(X.train, Y.train, X.test, Y.test, fea)[[2]]
	}
	if(!is.null(X.test)){
	  return(result)
	}
	else return(result[1])
}
}

