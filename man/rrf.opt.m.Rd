\name{rrf.opt.m}
\alias{rrf.opt.m}
\title{
KnowGRRF with weights from multiple knowledge domain
}
\description{
Regularize on the weights to guide RRF feature selection. Weights can from multiple knowledge domain and/or combination with statistics-based weights, e.g., p/q value, variable importance, etc. Proportion of weights can be scaled by regularization parameters. Feature set selected is also based on stability, that is the frequency of selection from multiple runs. Features that are consistently selected from multiple runs will be used in a random forest model, from which AIC and AUC will be calculated to evaluate the model performance. Only AIC will be calculated for regression.
}
\usage{
rrf.opt.m(X.train, Y.train, X.test=NULL, Y.test=NULL, par, wt, iter=1,total=10, cutoff=0.5)
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
  \item{par}{
a vector of parameters to adjust proportion of weights. The length of parameters is equal to the number of domain.
}
  \item{wt}{
A matrix of weights from each of domain knowledge. Columns are domains and rows are predictors.  
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

\examples{
##---- Should be DIRECTLY executable !! ----
##-- ==>  Define data, use random,
##--	or do  help(data=index)  for the standard data sets.

## The function is currently defined as
function(X.train, Y.train, X.test=NULL, Y.test=NULL, par, wt, iter=1,total=10, cutoff=0.5) {
	pwr <- par[1];
	ratio <- par[-1]/sum(par[-1]);
	weight <- wt %*% ratio;
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
	if(length(feature.rrf) > 1) {
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

