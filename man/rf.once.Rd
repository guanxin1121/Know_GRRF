\name{rf.once}
\alias{rf.once}

\title{
Build random forest once and return AUC for both training and test set if available
}
\description{
Build a random forest model once and return AUC for both training prediction (out-of-bag predictions) and test prediction. Work for classification only.
}
\usage{
rf.once(X.train, Y.train, X.test, Y.test, fea)
}

\arguments{
  \item{X.train}{
a data frame or matrix (like x) containing predictors for the training set.
}
  \item{Y.train}{
response for the training set. If a factor, classification is assumed, otherwise regression is assumed. If omitted, will run in unsupervised mode.
}
  \item{X.test}{
an optional data frame or matrix (like x) containing predictors for the test set.
}
  \item{Y.test}{
optional response for the test set. 
}
  \item{fea}{
feature index or feature names used to train the model.
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
##---- Example: classification ----
library(randomForest)
library(PRROC)
set.seed(1)
X<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(0.1, 2.2, 0.2) 
##y has a linear relationship with first 10 variables
y=b[6]*X$X5+b[7]*X$X6+b[8]*X$X7+b[9]*X$X8+b[10]*X$X9+b[11]*X$X10 
y=as.factor(ifelse(y>0, 1, 0)) ##classification

##split training and test set
X.train=X[1:70,]
X.test=X[71:100,]
y.train=y[1:70]
y.test=y[71:100]

rf.once(X.train, y.train, fea=1:20)  ##no test set
rf.once(X.train, y.train, X.test, y.test, 1:10)  ##relevant feature set
rf.once(X.train, y.train, X.test, y.test, 11:20)  ##irrelevant feature set


## The function is currently defined as

function(X.train, Y.train, X.test=NULL, Y.test=NULL, fea) {
  	f <- length(fea);
  	if(class(Y.train) == 'factor') {
    	model <- randomForest(X.train[, fea], Y.train)
    	pred.train <- model$votes;
    	pred.train <- data.frame(response=Y.train, pred=pred.train[, 2]);
    	roc.train <- roc.curve(pred.train[which(pred.train$response==1), 'pred'], pred.train[which(pred.train$response==0), 'pred'])
    	auc.train <- roc.train$auc
    	if(!is.null(X.test)){
    	  pred.test <- predict(model, X.test, type='prob');
    	  pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
      	roc.test <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'])
      	auc.test <- roc.test$auc
      	return(list(AUC.train=auc.train, AUC.test=auc.test))
    	}
    	else return(list(AUC.train=auc.train));
  	}
  	else{
  	  return("AUC not calculated for regression problem")
  	}
  }
}

