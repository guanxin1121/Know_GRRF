rf.repeat <-
function(X.train, Y.train, X.test=NULL, Y.test=NULL, fea, times=10) {
	result <- list(AUC.train=rep(NA, times), AUC.test=rep(NA, times));

	for(t in 1:times) {
	  result$AUC.train[[t]] <- rf.once(X.train, Y.train, X.test, Y.test, fea)[[1]]
	  if(!is.null(X.test))
      result$AUC.test[[t]] <- rf.once(X.train, Y.train, X.test, Y.test, fea)[[2]]
	}
	if(!is.null(X.test)){
	  return(result)
	}
	else return(result[1])
}


