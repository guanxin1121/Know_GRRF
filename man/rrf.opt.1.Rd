\name{rrf.opt.1}
\alias{rrf.opt.1}
\title{
KnowGRRF with weights from one knowledge domain
}
\description{
Regularize on the weights to guide RRF feature selection. Weights can from either one knowledge domain, or use statistics-based weights, e.g., p/q value, variable importance, etc. Feature set selected is also based on stability, that is the frequency of selection from multiple runs. Features that are consistently selected from multiple runs will be used in a random forest model, from which AIC and AUC will be calculated to evaluate the model performance. Only AIC will be calculated for regression.
}
\usage{
rrf.opt.1(X.train, Y.train, X.test=NULL, Y.test=NULL, pwr, weight, iter=1,total=10, cutoff=0.5)
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
  \item{pwr}{
Regularization term to adjust the scale of weights. Larger regularization will differentiate the importance of variables more significantly. Fewer variables tend to be selected with large pwr. This parameter can be tuned using optimization methods or grid searching with opt.pwr function.
}
  \item{weight}{
A vector of weights corresponding to each of predictors. Weights are between 0 and 1.
}
  \item{iter}{
The number of RF model built to evaluate AIC and AUC. AIC is calculated using out-of-bag prediction from random forest using feature selected. AUC is calculated for classification problem only.
}
  \item{total}{
the number of times to repeat the selection for stability test in select.stable function.
}
  \item{cutoff}{
The minimum percentage of times that the feature is selected, ranges between 0 and 1.
}
}

\value{
return a list, including
  \item{AIC}{AIC calculated from random forest model out-of-bag predicted probability for classification, or out-of-bag prediction for classification}
  \item{AUC}{AUC calculated from out-of-bag prediction from random forest classification model}
  \item{Test.AUC}{AUC calculated from test prediction from random forest classification model}
  \item{AUC}{AUC calculated from out-of-bag prediction from random forest classification model}
  \item{feaSet}{feature set selected}

}
\references{
Guan, X., & Liu, L. (2018). Know-GRRF: Domain-Knowledge Informed Biomarker Discovery with Random Forests. 
}
\author{
Li Liu, Xin Guan
}
\note{
This function can be used after weights and regularization term are determined. Weights are from knowledege domain and regularization term can be determined by optimization. See opt.pwr().
}


\examples{
##---- Example: classification ----
library(randomForest)

set.seed(1)
X.train<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(1, 6, 0.5) 
##y has a linear relationship with first 10 variables
y.train=b[7]*X.train$X6+b[8]*X.train$X7+b[9]*X.train$X8+b[10]*X.train$X9+b[11]*X.train$X10 
y.train=as.factor(ifelse(y.train>0, 1, 0)) ##classification

##use weights from domain knowledge. If not available, can use statistic-based weights, e.g., variable importance, p/q value, etc
imp<-randomForest(X.train, y.train)$importance 
coefReg=0.5+0.5*imp/max(imp) 

##use optimization function to find the appropriate regularization term to scale weights and then apply the weights to guide the RRF
par=optimize(opt.pwr, interval=c(0.1, 10), X.train=X.train, Y.train=y.train, weight=coefReg, iter=10)  ##error because optimization cannot find a parameter if no feature is selected, need to find another interval for initiation
par=optimize(opt.pwr, interval=c(0.01, 0.5), X.train=X.train, Y.train=y.train, weight=coefReg, iter=10)  ##work
rrf.opt.1(X.train, y.train, pwr=par$minimum, weight=coefReg)  



## The function is currently defined as
rrf.opt.1 <- function(X.train, Y.train, X.test=NULL, Y.test=NULL, pwr, weight, iter=1,total=10, cutoff=0.5) {

  surrogate <- min(weight[which(weight>0)])*0.1;
  weight[which(weight==0)] <- surrogate;
  coefReg <- weight^pwr;
  coefReg <- coefReg/max(coefReg);
  feature.rrf <- select.stable(X.train, Y.train, coefReg, total, cutoff)

  n <- length(Y.train);
  m <- length(feature.rrf);
  aic <- c();
  auc <- c();
  auc.test <- c();
  if(length(feature.rrf) >= 1) {
    for(i in 1:iter) {
      model.rrf_rf <- randomForest(X.train[, feature.rrf], Y.train)
      if(class(Y.train) == 'factor') {
        p1 <- model.rrf_rf$votes[,2]
        pred <- data.frame(response=Y.train, pred=p1);
        roc <- roc.curve(pred[which(pred$response==1), 'pred'], pred[which(pred$response==0), 'pred'])
        auc <- c(auc, roc$auc);
        
        p1 <- sapply(p1, function(x) ifelse(x==0, 1/(2*n), ifelse(x==1, 1-1/(2*n), x))) ##convert 0 or 1 to non-zero
        y <- data.frame(p1=p1, class=as.numeric(as.character(Y.train)));
        lh <- sum(y$class*log(y$p1) + (1-y$class)*log(1-y$p1), na.rm=F);
        aic <- c(aic, 2*m - 2*lh);
        
        if(!is.null(X.test)){
          pred.test <- predict(model.rrf_rf, X.test[, feature.rrf], type='prob')
          pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
          roc.test <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'])
          auc.test <- c(auc.test, roc.test$auc);
        }
      } else {
        mse <- mean((model.rrf_rf$predicted - Y.train)^2);
        aic <- c(aic, 2*m + n*log(mse));
        auc <- NA
        auc.test <- NA
      }
    }
    cat('par ', pwr, ' ... number of features ', m, ' ... aic ', mean(aic), ' ... auc ', mean(auc), ' ... auc.test ', mean(auc.test), '...\n');flush.console();
    return(list(AIC=aic, AUC=auc, Test.AUC=auc.test, feaSet=feature.rrf));
  } else {
    return("No feature is selected");
  }
}



}

