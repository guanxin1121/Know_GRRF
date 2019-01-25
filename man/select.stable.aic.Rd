\name{select.stable.aic}
\alias{select.stable.aic}
\title{
Select a set of stable features based on AIC after an initial selection by GRRF
}
\description{
Perform feature selection by GRRF and followed by stepwise model selection by AIC. 
Repeat it multiple times to select a stable set of features that are selected according to AIC.
}
\usage{
select.stable.aic(X.train, Y.train, coefReg, total=10)
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
##---- Example: classification  ----

set.seed(1)
X.train<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(0.1, 2.2, 0.2) 
##y has a linear relationship with first 10 variables
y.train=b[7]*X.train$X6+b[8]*X.train$X7+b[9]*X.train$X8+b[10]*X.train$X9+b[11]*X.train$X10 
y.train=ifelse(y.train>0, 1, 0) ##classification

##use RRF to impute regularized coefficients
imp<-randomForest(X.train, as.factor(y.train))$importance 
coefReg=0.5+0.5*imp/max(imp) 

##select a stable set of feature that are selected by GRRF followed by stepAIC
select.stable.aic(X.train, as.factor(y.train), coefReg)

## The function is currently defined as
function (X.train, Y.train, coefReg, total=10) 
{
    selected <- c()
    for (i in 1:total) {
  	  temp=RRF(X.train, Y.train, coefReg=coefReg, flagReg=1, importance=T)$feaSet
  	  selected <- c(selected, temp)
    }
    selected <- unique(selected)
    df <- data.frame(Y.train, X.train[, selected])
    colnames(df) <- c("resp", selected)
    if (class(Y.train) == "factor") {
        model.full <- glm(resp ~ ., data = df, family = binomial(link = "logit"))
        model.step <- stepAIC(model.full, direction = "both", 
            trace = 0)
        selected <- rownames(summary(model.step)$coef)[-1]
    }
    else {
        model.full <- lm(resp ~ ., data = df)
        model.step <- stepAIC(model.full, direction = "both", 
            trace = 0)
        selected <- rownames(summary(model.step)$coef)[-1]
    }
    return(selected)
  }
}

