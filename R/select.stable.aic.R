select.stable.aic <-
function(X.train, Y.train, coefReg, total=10) {
	selected <- c();
	for(i in 1:total) {
		##perf.once <- rrf.once(X.train, Y.train, X.test, Y.test, coefReg);
	  temp=RRF(X.train, Y.train, coefReg=coefReg, flagReg=1, importance=T)$feaSet
	  selected <- c(selected, temp)
		##selected <- c(selected, perf.once[['selected']]);
	}
	selected <- unique(selected);
	df <- data.frame(resp=Y.train, X.train[, selected]);
	##colnames(df) <- c('resp', selected);
	if(class(Y.train) == 'factor') {
		model.full <- glm(resp ~ ., data=df, family=binomial(link='logit'));
		model.step <- stepAIC(model.full, direction='both', trace=0);
		selected <- rownames(summary(model.step)$coef)[-1];
	} else {
		model.full <- lm(resp ~ ., data=df);
		model.step <- stepAIC(model.full, direction='both', trace=0);
		selected <- rownames(summary(model.step)$coef)[-1];
	}
	return(selected);
}
