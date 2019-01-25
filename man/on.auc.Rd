\name{on.auc}
\alias{on.auc}
\title{
AUC from model built with KnowGRRF
}
\description{
User can directly get model performance (AUC: Area under Curve) by providing weights and regularization term. Feature selection will be done by RRF and RF model will be built.
}
\usage{
on.auc(par, wt, num = 1, iter = 1)
}
\arguments{
  \item{par}{
A scaler or a vector of parameters to adjust proportion of weights. The length of parameters is equal to the number of domain.
}
  \item{wt}{
A vector or a matrix of weights corresponding to each of predictors from one or multiple domains.
}
  \item{num}{
The number of domains the weights come from
}
  \item{iter}{
The number of RF model built to evaluate AUC (Area under Curve)
}
}

\value{
mean of AUC from a number of RF model using feature selected by RRF
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
function (par, wt, num = 1, iter = 1) 
{
    if (num == 1) {
        return(rrf.opt.1(par, wt, iter)[2])
    }
    else {
        return(rrf.opt.m(par, wt, iter)[2])
    }
  }
}

