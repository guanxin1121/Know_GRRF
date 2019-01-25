\name{select.stable}
\alias{select.stable}

\title{
Select a set of stable features based on frequency picked by GRRF.
}
\description{
Perform feature selection by GRRF. 
Repeat it multiple times to select a stable set of features that are consistently selected according to the selection frequency.
}
\usage{
select.stable(X.train, Y.train, coefReg, total=10, cutoff=0.5) 
}

\arguments{
  \item{X.train}{
a data frame or matrix (like x) containing predictors for the training set.
}
  \item{Y.train}{
response for the training set. If a factor, classification is assumed, otherwise regression is assumed. If omitted, will run in unsupervised mode.
}
  \item{coefReg}{
regularization coefficient chosen for RRF, ranges between 0 and 1.
}
  \item{total}{
the number of times to repeat the process.
}
  \item{cutoff}{
The minimum percentage of times that the feature is selected by RRF, ranges between 0 and 1.
}
}

\value{
a stable set of features selected by GRRF
}
\references{
Guan, X., & Liu, L. (2018). Know-GRRF: Domain-Knowledge Informed Biomarker Discovery with Random Forests. 
}
\author{
Li Liu, Xin Guan
}
\note{
For customized hyperparameter setting, can directly call RRF function from RRF package repeatly in a for loop.
}



\examples{
##---- Example: classification ----

set.seed(1)
X.train<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(0.1, 2.2, 0.2) 
##y has a linear relationship with first 10 variables
y.train=b[7]*X.train$X6+b[8]*X.train$X7+b[9]*X.train$X8+b[10]*X.train$X9+b[11]*X.train$X10 
y.train=ifelse(y.train>0, 1, 0) ##classification

##use RRF to impute regularized coefficients
imp<-randomForest(X.train, as.factor(y.train))$importance 
coefReg=0.5+0.5*imp/max(imp) 

##select a stable set of feature that are consistently selected more than half of times
select.stable(X.train, as.factor(y.train), coefReg)




## The function is currently defined as
function(X.train, Y.train, coefReg, total=10, cutoff=0.5) {
	selected <- c();
	for(i in 1:total) {
	  temp=RRF(X.train, Y.train, coefReg=coefReg, flagReg=1, importance=T)$feaSet
		selected <- c(selected, temp)
	}
	freq <- as.data.frame(table(selected))
	if(cutoff < 1) {
		cutoff <- total*cutoff;
	}
	selected <- as.numeric(as.character(freq[which(freq$Freq >= cutoff), 'selected']))
	return(names(X.train[,selected]));
}
}
