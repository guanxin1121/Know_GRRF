################## Improved Know-GRRF ################
### Added functions to integrate prior information
###   from multiple domains.
### Authors: Li Liu, Xin Guan
### Arizona State University
### August 19, 2018
######################################################

library(randomForest)
library(RRF)
library(PRROC)
library(MASS)

write.roc <- function(X.train, Y.train, X.test, Y.test, feature.index, file.name) {
	model <- randomForest(X.train[, feature.index], Y.train)
	pred.test <- predict(model, X.test, type='prob');
	pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
	roc <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'], curv=T)
	cat('auroc: ', roc$auc, '\n'); flush.console();
	curv <- roc$curve;
	colnames(curv) <- c('FPR', 'TPR', 'cutoff');
	write.table(curv, file.name, sep='\t', row.names=F, quote=F);
	return();
}


rf.once <- function(X.train, Y.train, X.test, Y.test, feature.index) {
	f <- length(feature.index);
	model <- randomForest(X.train[, feature.index], Y.train, mtry=ifelse(f<2, f, sqrt(f)))
	pred.train <- model$votes;
	pred.train <- data.frame(response=Y.train, pred=pred.train[, 2]);
	pred.test <- predict(model, X.test, type='prob');
	pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
	roc.train <- roc.curve(pred.train[which(pred.train$response==1), 'pred'], pred.train[which(pred.train$response==0), 'pred'])
	auc.train <- roc.train$auc
	roc.test <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'])
	auc.test <- roc.test$auc
	return(c(auc.train, auc.test));
}

rf.repeat <- function(X.train, Y.train, X.test, Y.test, feature.index, times=10) {
	result <- c();
	for(t in 1:times) {
		result <- rbind(result, rf.once(X.train, Y.train, X.test, Y.test, feature.index));
	}
	return(result);
}

rrf.once <- function(X.train, Y.train, X.test, Y.test, coefReg) {
	model.rrf <- RRF(X.train, Y.train, coefReg=coefReg, flagReg=1, importance=T)
	feature.rrf <- model.rrf$feaSet;
	if(class(Y.train) == 'factor') {
		pred.rrf <- predict(model.rrf, X.test, type='prob')
		pred.rrf <- data.frame(response=Y.test, pred=pred.rrf[, 2]);
		roc.rrf <- roc.curve(pred.rrf[which(pred.rrf$response==1), 'pred'], pred.rrf[which(pred.rrf$response==0), 'pred'])
		(auc.rrf <- roc.rrf$auc)
		auc.rrf_rf <- auc.rrf;
		if(length(feature.rrf) > 1) {
			model.rrf_rf <- randomForest(X.train[, feature.rrf], Y.train)
			pred.rrf_rf <- predict(model.rrf_rf, X.test[, feature.rrf], type='prob')
			pred.rrf_rf <- data.frame(response=Y.test, pred=pred.rrf_rf[, 2]);
			roc.rrf_rf <- roc.curve(pred.rrf_rf[which(pred.rrf_rf$response==1), 'pred'], pred.rrf_rf[which(pred.rrf_rf$response==0), 'pred'])
			(auc.rrf_rf <- roc.rrf_rf$auc)
		}
		perf <- c(length(feature.rrf), auc.rrf, auc.rrf_rf);
	} else {
		mse <- mean((model.rrf$predicted - Y.train)^2);
		perf <- c(length(feature.rrf), mse, mse);
	}
	return(list(perf=perf, model=model.rrf, selected=feature.rrf));
}

rrf.opt.1 <- function(par, wt, iter=1) {
	pwr <- par;	
	weight <- wt;
	surrogate <- min(weight[which(weight>0)])*0.1;
	weight[which(weight==0)] <- surrogate;
	coefReg <- weight^pwr;
	coefReg <- coefReg/max(coefReg);
	model.rrf <- RRF(X.train, Y.train, coefReg=coefReg, flagReg=1)
	feature.rrf <- model.rrf$feaSet;
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
				
				pred.test <- predict(model.rrf_rf, X.test[, feature.rrf], type='prob')
				pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
				roc.test <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'])
				auc.test <- c(auc.test, roc.test$auc);
			} else {
				mse <- mean((model.rrf_rf$predicted - Y.train)^2);
				aic <- c(aic, 2*m + n*log(mse));
				auc <- c(auc, 0);
				auc.test <- c(auc.test, 0);
			}
		}
		cat('par ', par, ' ... features ', m, ' ... aic ', mean(aic, na.rm=T), ' ... auc ', mean(auc, na.rm=T), ' ... auc.test ', mean(auc.test, na.rm=T), '...\n');flush.console();
		return(c(mean(aic, na.rm=T), mean(auc, na.rm=T)));
	} else {
		return(c(10000, 0));
	}
}

rrf.opt.m <- function(par, wt, iter=1) {
	pwr <- par[1];
	ratio <- par[-1]/sum(par[-1]);
	weight <- wt %*% ratio;
	surrogate <- min(weight[which(weight>0)])*0.1;
	weight[which(weight==0)] <- surrogate;
	coefReg <- weight^pwr;
	coefReg <- coefReg/max(coefReg);
	model.rrf <- RRF(X.train, Y.train, coefReg=coefReg, flagReg=1)
	feature.rrf <- model.rrf$feaSet;
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
				
				pred.test <- predict(model.rrf_rf, X.test[, feature.rrf], type='prob')
				pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
				roc.test <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'])
				auc.test <- c(auc.test, roc.test$auc);
			} else {
				mse <- mean((model.rrf_rf$predicted - Y.train)^2);
				aic <- c(aic, 2*m + n*log(mse));
				auc <- c(auc, 0);
				auc.test <- c(auc.test, 0);
			}
		}
		cat('par ', par, ' ... features ', m, ' ... aic ', mean(aic, na.rm=T), ' ... auc ', mean(auc, na.rm=T), ' ... auc.test ', mean(auc.test, na.rm=T), '...\n');flush.console();
		return(c(mean(aic, na.rm=T), mean(auc, na.rm=T)));
	} else {
		return(c(10000, 0));
	}
}

on.aic <- function(par, wt, num=1, iter=1) {	
	if(num == 1) {
		return(rrf.opt.1(par, wt, iter)[1]);
#	} else if(num == 2) {
#		return(rrf.opt(par, wt[, 1], wt[, 2], iter)[1]);
	} else {
		return(rrf.opt.m(par, wt, iter)[1]);
	}
}

on.auc <- function(par, wt, num=1, iter=1) {
	if(num == 1) {
		return(rrf.opt.1(par, wt, iter)[2]);
#	} else if(num == 2) {
#		return(rrf.opt(par, wt[, 1], wt[, 2], iter)[2]);
	} else {
		return(rrf.opt.m(par, wt, iter)[2]);
	}
}

get.performance <- function(set.truth, set.sel) {
	ji <- 0;
	tpr <- 0;
	fpr <- 0;
	u <- length(set.truth);
	s <- length(set.sel);
	a <- length(intersect(set.truth, set.sel));
	b <- length(union(set.truth, set.sel));
	if(b > 0) {
		ji <- a / b;
	}
	if(u > 0) {
		tpr <- a / u;
	}
	if(s > 0) {
		fpr <- length(setdiff(set.sel, set.truth)) / s;
	}
	return(round(c(ji, tpr, fpr), digits=4));
}

select.stable <- function(total=10, cutoff=0.5, coefReg) {
	selected <- c();
	for(i in 1:total) {
		perf.once <- rrf.once(X.train, Y.train, X.test, Y.test, coefReg);
		selected <- c(selected, perf.once[['selected']]);
	}
	freq <- as.data.frame(table(selected))
	if(cutoff < 1) {
		cutoff <- total*cutoff;
	}
	selected <- as.numeric(as.character(freq[which(freq$Freq >= cutoff), 'selected']))
	return(selected);
}


select.stable.aic <- function(total, coefReg) {
	selected <- c();
	for(i in 1:total) {
		perf.once <- rrf.once(X.train, Y.train, X.test, Y.test, coefReg);
		selected <- c(selected, perf.once[['selected']]);
	}
	selected <- unique(selected);
	df <- data.frame(Y.train, X.train[, selected]);
	colnames(df) <- c('resp', selected);
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
