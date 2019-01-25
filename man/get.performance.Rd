\name{get.performance}
\alias{get.performance}
\title{
Get performance of feature selection
}
\description{
Comparing feature selected by algorithms and the ground truth in simulation
}
\usage{
get.performance(set.truth, set.sel, set.all)
}
\arguments{
  \item{set.truth}{
A vector of important features set in simulation (ground truth)
}
  \item{set.sel}{
A vector of important features selected by algorithms
}
  \item{set.all}{
A vector of all candidate features for feature selection
}
}

\value{
returned a vector of feature selection performance including JI (ratio of intersect of two sets and union of two sets), TPR (percentage of correctly selected features in all true important features) and FPR (percentage of wrongly selected features in true non-important features)
}
\references{
Guan, X., & Liu, L. (2018). Know-GRRF: Domain-Knowledge Informed Biomarker Discovery with Random Forests. 
}
\author{
Li Liu, Xin Guan
}


\examples{
set.truth=1:10  ##true important feature from ground truth
set.sel=c(8:10, 95)  ##selected feature by an algorithm
set.all=1:100  ##all candidate features

get.performance(set.truth, set.sel, set.all)


## The function is currently defined as
function (set.truth, set.sel, set.all)
  {
    ji <- 0
    tpr <- 0
    fpr <- 0
    u <- length(set.truth)
    s <- length(set.sel)
    p <- length(set.all)
    a <- length(intersect(set.truth, set.sel))
    b <- length(union(set.truth, set.sel))
    if (b > 0) {
      ji <- round(a/b, 4)
    }
    if (u > 0) {
      tpr <- round(a/u, 4)
    }
    if (s > 0) {
      fpr <- round(length(setdiff(set.sel, set.truth))/(p-u), 4)
    }
    return(data.frame(JI=ji, TPR=tpr, FPR=fpr))
  }
}
