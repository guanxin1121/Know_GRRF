opt.pwr <-
  function(X.train, Y.train, pwr, weight, iter=1,total=10, cutoff=0.5) {
    
    surrogate <- min(weight[which(weight>0)])*0.1;
    weight[which(weight==0)] <- surrogate;
    coefReg <- weight^pwr;
    coefReg <- coefReg/max(coefReg);
    feature.rrf <- select.stable(X.train, Y.train, coefReg, total, cutoff)
    
    n <- length(Y.train);
    m <- length(feature.rrf);
    aic <- c();

    if(length(feature.rrf) >= 1) {
      for(i in 1:iter) {
        model.rrf_rf <- randomForest(X.train[, feature.rrf], Y.train)
        if(class(Y.train) == 'factor') { ##classification
          p1 <- model.rrf_rf$votes[,2]
          p1 <- sapply(p1, function(x) ifelse(x==0, 1/(2*n), ifelse(x==1, 1-1/(2*n), x))) ##convert 0 or 1 to non-zero
          y <- data.frame(p1=p1, class=as.numeric(as.character(Y.train)));
          lh <- sum(y$class*log(y$p1) + (1-y$class)*log(1-y$p1), na.rm=F);
          aic <- c(aic, 2*m - 2*lh);
          
          
        } else {  ##regression
          mse <- mean((model.rrf_rf$predicted - Y.train)^2);
          aic <- c(aic, 2*m + n*log(mse));
          
        }
      }
      cat('par ', pwr, ' ... features ', m, ':', feature.rrf, ' ... aic ', mean(aic, na.rm=T),  '...\n');flush.console();
      ##return(data.frame(AIC=mean(aic), AUC=mean(auc), AUC_test=mean(auc.test)));
      return(mean(aic))
    } else {
      return("No feature is selected");
    }
  }

