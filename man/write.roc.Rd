\name{write.roc}
\alias{write.roc}

\title{
write test ROC to a data table. 
}
\description{
write a data table including False Positive Rate, True Positive Rate and cutoff on test dataset. Work for classification only.
}
\usage{
write.roc(X.train, Y.train, X.test, Y.test, fea, file.name="")
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
  \item{file.name}{
directory and name of files that write to. If directory is not given, will write to working directory.
}
}

\value{
a data table in csv format which columns FPR, TPR and cutoff.
}
\references{
Guan, X., & Liu, L. (2018). Know-GRRF: Domain-Knowledge Informed Biomarker Discovery with Random Forests. 
}
\author{
Li Liu, Xin Guan
}



\examples{
##---- Example: classification ----

set.seed(1)
X<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(0.1, 2.2, 0.2) 
##y has a linear relationship with first 10 variables
y=b[7]*X$X6+b[8]*X$X7+b[9]*X$X8+b[10]*X$X9+b[11]*X$X10 
y=as.factor(ifelse(y>0, 1, 0)) ##classification

##split training and test set
X.train=X[1:70,]
X.test=X[71:100,]
y.train=y[1:70]
y.test=y[71:100]

write.roc(X.train, y.train, X.test, y.test, fea=1:20, "example.csv")  

## The function is currently defined as

function(X.train, Y.train, X.test, Y.test, fea, file.name) {
  	model <- randomForest(X.train[, fea], Y.train)
  	pred.test <- predict(model, X.test, type='prob');
  	pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
  	roc <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'], curve=T)
  	cat('auroc: ', roc$auc, '\n'); flush.console();
  	curv <- roc$curve;
  	colnames(curv) <- c('FPR', 'TPR', 'cutoff');
  	write.csv(curv, file.name,  row.names=F, quote=F);
  	cat("if directory is not given, file was wrote to", getwd());
  }
}

