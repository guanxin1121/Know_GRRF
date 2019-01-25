get.performance <-
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
