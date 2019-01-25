rrf.once <-
function(X.train, Y.train, X.test, Y.test, coefReg) {
  feature.rrf <- RRF(X.train, Y.train, coefReg=coefReg, flagReg=1, importance=T)$feaSet
  model.rrf <- randomForest(X.train, Y.train)
  
  if(length(feature.rrf) >= 1){
    model.rrf_rf <- randomForest(X.train[, feature.rrf], Y.train)
    
  	if(class(Y.train) == 'factor') {
  	  ##using all features
  		pred.rrf <- predict(model.rrf, X.test, type='prob')
  		pred.rrf <- data.frame(response=Y.test, pred=pred.rrf[, 2]);
  		roc.rrf <- roc.curve(pred.rrf[which(pred.rrf$response==1), 'pred'], pred.rrf[which(pred.rrf$response==0), 'pred'])
  		auc.rrf <- roc.rrf$auc
  
  		##using selected features
			pred.rrf_rf <- predict(model.rrf_rf, X.test[, feature.rrf], type='prob')
			pred.rrf_rf <- data.frame(response=Y.test, pred=pred.rrf_rf[, 2]);
			roc.rrf_rf <- roc.curve(pred.rrf_rf[which(pred.rrf_rf$response==1), 'pred'], pred.rrf_rf[which(pred.rrf_rf$response==0), 'pred'])
			auc.rrf_rf <- roc.rrf_rf$auc
  		
  		perf <- data.frame(NumberofFeatures=length(feature.rrf), AUC.all=auc.rrf, AUC.selected=auc.rrf_rf)
  	} else {
  	  pred.rrf <- predict(model.rrf, X.test, type='response')
  	  pred.rrf_rf <- predict(model.rrf_rf, X.test[, feature.rrf], type='response')
  		mse1 <- mean((pred.rrf - Y.test)^2)
  		mse2 <- mean((pred.rrf_rf - Y.test)^2)
  		perf <- data.frame(NumberofFeatures=length(feature.rrf), MSE.all=mse1, MSE.selected=mse2);
  	}
	return(list(perf=perf, FullModel=model.rrf, ReducedModel=model.rrf_rf, featureIndex=feature.rrf));
  }
  else return("No feature is selected by RRF")
}
