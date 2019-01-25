select.stable <-
function(X.train, Y.train, coefReg, total=10, cutoff=0.5) {
	selected <- c();
	for(i in 1:total) {
		##perf.once <- rrf.once(X.train, Y.train, X.test, Y.test, coefReg);
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
