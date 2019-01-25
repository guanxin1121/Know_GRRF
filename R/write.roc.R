write.roc <-
function(X.train, Y.train, X.test, Y.test, fea, file.name) {
	model <- randomForest(X.train[, fea], Y.train)
	pred.test <- predict(model, X.test, type='prob');
	pred.test <- data.frame(response=Y.test, pred=pred.test[, 2]);
	roc <- roc.curve(pred.test[which(pred.test$response==1), 'pred'], pred.test[which(pred.test$response==0), 'pred'], curve=T)
	cat('auroc: ', roc$auc, '\n'); flush.console();
	curv <- roc$curve;
	colnames(curv) <- c('FPR', 'TPR', 'cutoff');
	write.csv(curv, file.name,  row.names=F, quote=F);
	cat("if directory is not given, file was wrote to", getwd());
}
