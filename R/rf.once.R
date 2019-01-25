rf.once <-
function(X.train, Y.train, X.test=NULL, Y.test=NULL, fea) {
	f <- length(fea);
	if(class(Y.train) == 'factor') {
  	model <- randomForest(X.train[, fea], Y.train)
  	pred.train <- model$votes;
  	pred.train <- data.frame(response=Y.train, pred=pred.train[, 2]);
  	roc.train <- roc.curve(pred.train[which(pred.train$response==1), 'pred'], pred.train[which(pred.train$response==0), 'pred'])
  	auc.train <- roc.train$auc
  	if(!is.null(X.test)){
  	  pred.test <- predict(model, X.test, type='prob');
  	  pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
    	roc.test <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'])
    	auc.test <- roc.test$auc
    	return(list(AUC.train=auc.train, AUC.test=auc.test))
  	}
  	else return(list(AUC.train=auc.train));
	}
	else{
	  return("AUC not calculated for regression problem")
	}
}
