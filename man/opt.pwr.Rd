\name{opt.pwr}
\alias{opt.pwr}

\title{
Functions used with optimization to find regularization parameter to minimize AIC.
}
\description{
KnowGRRF need a regularization term to scale relative weights of different variables. This function use select.stable to select features based on selection frequency from multiple runs. Features then would be used in a random forest model, from which AIC will be calculated to evaluate the model performance and can be used for optimization.
}
\usage{
opt.pwr(X.train, Y.train, pwr, weight, iter=1,total=10, cutoff=0.5)
}
\arguments{
  \item{X.train}{
a data frame or matrix (like x) containing predictors for the training set.
}
  \item{Y.train}{
response for the training set. If a factor, classification is assumed, otherwise regression is assumed. If omitted, will run in unsupervised mode.
}
  \item{pwr}{
Regularization term to adjust the scale of weights. Larger regularization will differentiate the importance of variables more significantly. Fewer variables tend to be selected with large pwr. This parameter can be tuned using optimization methods or grid searching.
}
  \item{weight}{
A vector of weights corresponding to each of predictors. Weights are between 0 and 1.
}
  \item{iter}{
The number of RF model built to evaluate AIC. AIC is calculated using out-of-bag prediction from random forest using feature selected.
}
  \item{total}{
the number of times to repeat the process.
}
  \item{cutoff}{
The minimum percentage of times that the feature is selected by RRF, ranges between 0 and 1.
}
}

\value{
mean of AIC from a number of RF model using feature selected by this function.
setting of pwr, number of features and feature set selected will be printed as well as mean of AIC.

}
\references{
Guan, X., & Liu, L. (2018). Know-GRRF: Domain-Knowledge Informed Biomarker Discovery with Random Forests. 
}
\author{
Li Liu, Xin Guan
}
\note{
This function can be used with optimization function together to search the best value of regularization parameter while minimizing AIC. Minimum returned by optimization can be used as pwr for this function. See example.
}




\examples{
##---- Example: regression ----
library(randomForest)

set.seed(1)
X.train<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(1, 6, 0.5) 
##y has a linear relationship with first 10 variables
y.train=b[7]*X.train$X6+b[8]*X.train$X7+b[9]*X.train$X8+b[10]*X.train$X9+b[11]*X.train$X10 


##use variable importance as weights
imp<-randomForest(X.train, y.train)$importance 


##use optimization function to find the appropriate regularization term to scale weights and then apply the weights to guide the RRF
par=optimize(opt.pwr, interval=c(0.1, 10), X.train=X.train, Y.train=y.train, weight=imp/max(imp), iter=10)  ##will take some time to finish
opt.pwr(X.train, y.train, par$minimum, imp/max(imp))  ##after pwr determined, can call the function directly  ##can use any time to substitute par$minimum



## The function is currently defined as
opt.pwr <-
  function(X.train, Y.train, pwr, weight, iter=1,total=10, cutoff=0.5) {
    
    surrogate <- min(weight[which(weight>0)])*0.1;
    weight[which(weight==0)] <- surrogate;
    coefReg <- weight^pwr;
    coefReg <- coefReg/max(coefReg);
    feature.rrf <- select.stable(X.train, Y.train, coefReg, total, cutoff)
    
    n <- length(Y.train);
    m <- length(feature.rrf);
    aic <- c();

    if(length(feature.rrf) >= 1) {
      for(i in 1:iter) {
        model.rrf_rf <- randomForest(X.train[, feature.rrf], Y.train)
        if(class(Y.train) == 'factor') { ##classification
          p1 <- model.rrf_rf$votes[,2]
          p1 <- sapply(p1, function(x) ifelse(x==0, 1/(2*n), ifelse(x==1, 1-1/(2*n), x))) ##convert 0 or 1 to non-zero
          y <- data.frame(p1=p1, class=as.numeric(as.character(Y.train)));
          lh <- sum(y$class*log(y$p1) + (1-y$class)*log(1-y$p1), na.rm=F);
          aic <- c(aic, 2*m - 2*lh);
          
          
        } else {  ##regression
          mse <- mean((model.rrf_rf$predicted - Y.train)^2);
          aic <- c(aic, 2*m + n*log(mse));
          
        }
      }
      cat('par ', pwr, ' ... features ', m, ':', feature.rrf, ' ... aic ', mean(aic, na.rm=T),  '...\n');flush.console();
      ##return(data.frame(AIC=mean(aic), AUC=mean(auc), AUC_test=mean(auc.test)));
      return(mean(aic))
    } else {
      return("No feature is selected");
    }
  }




}

