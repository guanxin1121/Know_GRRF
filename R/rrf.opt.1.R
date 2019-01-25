rrf.opt.1 <- function(X.train, Y.train, X.test=NULL, Y.test=NULL, pwr, weight, iter=1,total=10, cutoff=0.5) {

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
  if(length(feature.rrf) >= 1) {
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
