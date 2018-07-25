require(RRF)
require(VSURF)
require(randomForest)
require(varSelRF)

###########################simulation#############################################
###### classification simulation
data<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(0.1, 2.2, 0.2)
##linear
y=b[2]*data$X1+b[3]*data$X2+b[4]*data$X3+b[5]*data$X4+b[6]*data$X5+b[7]*data$X6+b[8]*data$X7+b[9]*data$X8+b[10]*data$X9+b[11]*data$X10
y=ifelse(y>0, 1, 0)
##higher-order term
y2=b[5]*data$X4+b[6]^2*data$X5+b[7]*data$X6+b[9]*data$X8+b[10]*data$X9+b[11]*data$X10+b[9]*data$X11^2
y2=ifelse(y2>0, 1, 0)
##interaction terms
y3=b[1]+b[5]*data$X4+b[6]^2*data$X5+b[7]*data$X6+b[9]*data$X8+b[10]*data$X9+b[11]*data$X10+b[9]*data$X11*data$X12
y3=ifelse(y3>0, 1, 0)
##redundant features
data[91:100]=data[1:10]

x.test=data.frame(matrix(rnorm(100*100), nrow=100))
y.test=b[2]*x.test$X1+b[3]*x.test$X2+b[4]*x.test$X3+b[5]*x.test$X4+b[6]*x.test$X5+b[7]*x.test$X6+b[8]*x.test$X7+b[9]*x.test$X8+b[10]*x.test$X9+b[11]*x.test$X10
y.test=ifelse(y.test>0, 1, 0)
y2.test=b[5]*x.test$X4+b[6]^2*x.test$X5+b[7]*x.test$X6+b[9]*x.test$X8+b[10]*x.test$X9+b[11]*x.test$X10+b[9]*x.test$X11^2
y2.test=ifelse(y2.test>0, 1, 0)
y3.test=b[1]+b[5]*x.test$X4+b[6]^2*x.test$X5+b[7]*x.test$X6+b[9]*x.test$X8+b[10]*x.test$X9+b[11]*x.test$X10+b[9]*x.test$X11*x.test$X12
y3.test=ifelse(y3.test>0, 1, 0)


##10 runs
test<-function(feature, y){
  oob<-c()
  for (i in 1:10){
    rf<-randomForest(data[feature], as.factor(y))
    oob[i]<-(rf$confusion[1,2]+rf$confusion[2,1])/100
  }
  list(mean=mean(oob), sd=sd(oob))
}

test2<-function(feature, x, y){
  oob<-c()
  for (i in 1:10){
    rf<-randomForest(x[feature], as.factor(y))
    oob[i]<-(rf$confusion[1,2]+rf$confusion[2,1])/100
  }
  list(mean=mean(oob), sd=sd(oob))
}

#####################################feature selection#############################
##no feature selection
e1<-test(1:100, y)
e1.2<-test(1:100, y2)
e1.3<-test(1:100, y3)

##varSelRF
var.data<-varSelRF(data, as.factor(y))
e2<-test(var.data$selected.vars, y)

var.data.2<-varSelRF(data, as.factor(y2))
e2.2<-test(var.data.2$selected.vars, y2)

var.data.3<-varSelRF(data, as.factor(y3))
e2.3<-test(var.data.3$selected.vars, y3)

##RRF
imp<-randomForest(data, as.factor(y))$importance
coefReg=0.5+0.5*imp/max(imp)
RRF.data<-RRF(data, as.factor(y), coefReg = coefReg)
e3<-test(RRF.data$feaSet, y)

imp.2<-randomForest(data, as.factor(y2))$importance
coefReg.2=0.5+0.5*imp.2/max(imp.2)
RRF.data.2<-RRF(data, as.factor(y2), coefReg = coefReg.2)
e3.2<-test(RRF.data.2$feaSet, y2)

imp.3<-randomForest(data, as.factor(y3))$importance
coefReg.3=0.5+0.5*imp.3/max(imp.3)
RRF.data.3<-RRF(data, as.factor(y3), coefReg = coefReg.3)
e3.3<-test(RRF.data.3$feaSet, y3)

##################################################
##Know-GRRF
##aw is the exponential term of knowledge score, the parameter can be tuned using optimization methods or grid searching
opt.fn<-function(q.norm, aw, y){
  coefReg=q.norm^aw
  candidate=list()
  aic=c()
  for (i in 1:10){  
    RRF.data<-RRF(data, as.factor(y),mtry=dim(data)[2], coefReg = coefReg)
    RF<-randomForest(data[RRF.data$feaSet], as.factor(y))
    p1<-RF$votes[,2]
    p1<-sapply(p1, function(x) ifelse(x==0, 1/(2*length(y)), ifelse(x==1, 1-1/(2*length(y)), x)))
    l=sum(y*log(p1)+(1-y)*log(1-p1), na.rm=F)
    aic[i]=-2*l+2*length(RRF.data$feaSet)
    candidate[[i]]=RRF.data$feaSet
  }
  list(aic=mean(aic), min=min(aic), max=max(aic),candidate=candidate)
}

###We can use optimization to find out the exponential term "aw" that minimize AIC
opt.aic<-function(imp.norm, aw, y){   ###imp.norm is the domain knowledge weights
  coefReg=imp.norm^aw
  candidate=list()
  aic=c()
  for (i in 1:10){  
    RRF.data<-RRF(data, as.factor(y),mtry=dim(data)[2], coefReg = coefReg)
    RF<-randomForest(data[RRF.data$feaSet], as.factor(y))
    p1<-RF$votes[,2]
    p1<-sapply(p1, function(x) ifelse(x==0, 1/(2*length(y)), ifelse(x==1, 1-1/(2*length(y)), x)))
    l=sum(y*log(p1)+(1-y)*log(1-p1), na.rm=F)
    aic[i]=-2*l+2*length(RRF.data$feaSet)    #####AIC
  }
  mean(aic)
}
opt1<-optimize(opt.aic, interval=c(0.1,5), imp.norm=q.norm, y=y)
               
               

############################################Know-GRRF for regression########################
######## the difference of regression and classification is the objective function of AIC
 opt.fn<-function(imp.norm, aw, y){
  coefReg=imp.norm^aw
  candidate=list()
  aic=c()
  for (i in 1:10){  
    RRF.data<-RRF(data, y,mtry=dim(data)[2], coefReg = coefReg)
    RF<-randomForest(data[RRF.data$feaSet], y)
    mse<-mean((RF$predicted-y)^2)
    aic[i]=dim(data)[1]*log(mse)+2*length(RRF.data$feaSet)   ######MSE with regularization
    candidate[[i]]=RRF.data$feaSet
  }
  list(aic=mean(aic), min=min(aic), max=max(aic),candidate=candidate)
}
