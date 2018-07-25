require(RRF)
require(VSURF)
require(randomForest)
require(varSelRF)


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

##RRF_opt
fn<-function(aw){
  coefReg=(1-q)^aw
  aic<-c()
  for (i in 1:5){  
    RRF.data<-RRF(data, as.factor(y),mtry=dim(data)[1], coefReg = coefReg)
    RF<-randomForest(data[RRF.data$feaSet], as.factor(y))
    p1<-RF$votes[,2]
    p1<-sapply(p1, function(x) ifelse(x==0, 1/(2*length(y)), ifelse(x==1, 1-1/(2*length(y)), x)))
    l=sum(y*log(p1)+(1-y)*log(1-p1), na.rm=F)
    aic[i]=-2*l+2*length(RRF.data$feaSet)
  }
  return(mean(aic))
}


p<-apply(data, 2, function(x)t.test(x~y)$p.value)
q<-p.adjust(p, "BH")
# opt<-optimize(fn, interval=c(0,5))
# aw<-opt$minimum
# RRF.data2<-RRF(data, as.factor(y),coefReg = (1-q)^0.5)
# e4<-test(RRF.data2$feaSet, y)

q.norm=(1-q)/max(1-q)
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

fn2<-function(q.norm, aw, y){
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
  list(aic=aic,candidate=candidate)
}

fn3<-function(feature, y){
  aic=c()
  for (i in 1:5){  
    RF<-randomForest(data[feature], as.factor(y))
    p1<-RF$votes[,2]
    p1<-sapply(p1, function(x) ifelse(x==0, 1/(2*length(y)), ifelse(x==1, 1-1/(2*length(y)), x)))
    l=sum(y*log(p1)+(1-y)*log(1-p1), na.rm=F)
    aic[i]=-2*l+2*length(RRF.data$feaSet)
  }
 mean(aic)
}

can1=list()
min1=c()
max1=c()
aic1<-c()
aw=c(0.1, 0.2, 0.4, 0.8, 1:10)
for (j in 1:14){
  result=opt.fn(q.norm, aw[j], y)
  aic1[j]=result$aic
  min1[j]=result$min
  max1[j]=result$max
  can1[[j]]=result$candidate
}
temp=data.frame(delta=aw, AIC=aic1, min=min1, max=max1)
ggplot(temp, aes(delta, AIC, ymin=min, ymax=max))+geom_pointrange()+
   labs(title="Linear Classification",x=expression(delta))+theme(plot.title = element_text(hjust = 0.5))
table=table(unlist(can1[[6]]))
win=table[table>9]
test(paste("X", names(win), sep=""), y)

opt.fn(q.norm, 2.3, y)

test2(paste("X", c(1:100), sep=""), x.test, y.test)
test2(paste("X", c(5, 9, 28, 36, 42), sep=""), x.test, y.test)
test2(paste("X", c(4 , 5 , 9, 10, 61, 85), sep=""), x.test, y.test)
test2(paste("X", c(5,  6,  7,  9, 10, 42), sep=""), x.test, y.test)
test2(paste("X", c(1:10), sep=""), x.test, y.test)


test2(paste("X", c(1:100), sep=""), x.test, y2.test)
test2(paste("X", c(4,5, 9,10, 30, 61), sep=""), x.test, y2.test)
test2(paste("X", c(4,5, 9:11, 30,32), sep=""), x.test, y2.test)
test2(paste("X", c(4:6,  9, 10, 61), sep=""), x.test, y2.test)
test2(paste("X", c(4:11), sep=""), x.test, y2.test)


test2(paste("X", c(1:100), sep=""), x.test, y3.test)
test2(paste("X", c(5, 8:10, 29), sep=""), x.test, y3.test)
test2(paste("X", c(5,8,10), sep=""), x.test, y3.test)
test2(paste("X", c(5, 8:10, 29,52), sep=""), x.test, y3.test)
test2(paste("X", c(4:12), sep=""), x.test, y3.test)



# fn.2<-function(aw){
#   coefReg=(1-q.2)^aw
#   aic<-c()
#   for (i in 1:5){  
#     RRF.data<-RRF(data, as.factor(y2),mtry=dim(data)[1], coefReg = coefReg)
#     RF<-randomForest(data[RRF.data$feaSet], as.factor(y2))
#     p1<-RF$votes[,2]
#     p1<-sapply(p1, function(x) ifelse(x==0, 1/(2*length(y2)), ifelse(x==1, 1-1/(2*length(y2)), x)))
#     l=sum(y2*log(p1)+(1-y2)*log(1-p1), na.rm=F)
#     aic[i]=-2*l+2*length(RRF.data$feaSet)
#   }
#   return(mean(aic))
# }

p.2<-apply(data, 2, function(x)t.test(x~y2)$p.value)
q.2<-p.adjust(p.2, "BH")
#opt.2<-optimize(fn.2, interval=c(0,1))
#aw.2<-opt.2$minimum
#RRF.data2.2<-RRF(data, as.factor(y2),coefReg = (1-q.2)^aw.2)
#e4.2<-test(RRF.data2.2$feaSet, y2)
q2.norm=(1-q.2)/max(1-q.2)
can2=list()
min2=c()
max2=c()
aic2<-c()
aw=c(0.1, 0.2, 0.4, 0.8, 1:10)
for (j in 1:14){
  result=opt.fn(q2.norm, aw[j], y2)
  aic2[j]=result$aic
  min2[j]=result$min
  max2[j]=result$max
  can2[[j]]=result$candidate
}
temp=data.frame(delta=aw, AIC=aic2, min=min2, max=max2)
ggplot(temp, aes(delta, AIC, ymin=min, ymax=max))+geom_pointrange()+
  labs(title="Classification with higher-order term",x=expression(delta))+theme(plot.title = element_text(hjust = 0.5))
table=table(unlist(can2[[4]]))
win=table[table>9]
test(paste("X", names(win), sep=""), y2)

temp2<-fn2(q2.norm, 1.97, y2)
table(unlist(temp2$candidate))
a<-fn3(paste("X", c(4,5, 9,10), sep=""), y2)
b<-fn3(paste("X", c(4,5, 9,10,61), sep=""), y2)
c<-fn3(paste("X", c(4,5, 9,10, 61, 6), sep=""), y2)
d<-fn3(paste("X", c(4,5, 9,10, 61, 6, 30), sep=""), y2)

par(mfrow=c(1,1))
plot(c(1,0.9,0.6, 0.4), c(a,b,c,d), xlab="Stability", ylab="AIC", main="Stability Test in Scenario 2")

# fn.3<-function(aw){
#   coefReg=(1-q.3)^aw
#   aic<-c()
#   for (i in 1:5){  
#     RRF.data<-RRF(data, as.factor(y3),mtry=dim(data)[1], coefReg = coefReg)
#     RF<-randomForest(data[RRF.data$feaSet], as.factor(y3))
#     p1<-RF$votes[,2]
#     p1<-sapply(p1, function(x) ifelse(x==0, 1/(2*length(y3)), ifelse(x==1, 1-1/(2*length(y3)), x)))
#     l=sum(y3*log(p1)+(1-y3)*log(1-p1), na.rm=F)
#     aic[i]=-2*l+2*length(RRF.data$feaSet)
#   }
#   return(mean(aic))
# }

p.3<-apply(data, 2, function(x)t.test(x~y3)$p.value)
q.3<-p.adjust(p.3, "BH")
# opt.3<-optimize(fn.3, interval=c(0,1))
# aw.3<-opt.3$minimum
# RRF.data2.3<-RRF(data, as.factor(y3),coefReg = (1-q.3)^aw.3)
# e4.3<-test(RRF.data2.3$feaSet, y3)

q3.norm=(1-q.3)/max(1-q.3)
can3=list()
min3=c()
max3=c()
aic3<-c()
aw=c(0.1, 0.2, 0.4, 0.8, 1:10)
for (j in 1:14){
  result=opt.fn(q3.norm, aw[j], y3)
  aic3[j]=result$aic
  min3[j]=result$min
  max3[j]=result$max
  can3[[j]]=result$candidate
}
temp=data.frame(delta=aw, AIC=aic3, min=min3, max=max3)
ggplot(temp, aes(delta, AIC, ymin=min, ymax=max))+geom_pointrange()+
  labs(title="Classification with interaction term",x=expression(delta))+theme(plot.title = element_text(hjust = 0.5))
table=table(unlist(can3[[4]]))
win=table[table>6]
test(paste("X", names(win), sep=""), y3)


temp3<-fn2(q3.norm, 0.81, y3)
table(unlist(temp3$candidate))
a<-fn3(paste("X", c(5, 8:10, 52), sep=""), y3)
b<-fn3(paste("X", c(5, 8:10, 52,29), sep=""), y3)
c<-fn3(paste("X", c(5, 8:10, 52,29, 16), sep=""), y3)

plot(c(1,0.7,0.6), c(a,b,c), xlab="Stability", ylab="AIC", main="Stability Test in Scenario 3")

##coefficient plot
par(mfrow=c(2,3))
plot(q.norm, ylab="q'", xlab="variable", main="linear")
plot(q2.norm, ylab="q'", xlab="variable", main="polynomial")
plot(q3.norm, ylab="q'", xlab="variable", main="interaction")
plot(q.norm^2, ylab="q'^2", xlab="variable", main="linear")
plot(q2.norm^0.8, ylab="q'^0.8", xlab="variable", main="polynomial")
plot(q3.norm^0.8, ylab="q'^0.8", xlab="variable", main="interaction")


opt.aic<-function(imp.norm, aw, y){
  coefReg=imp.norm^aw
  candidate=list()
  aic=c()
  for (i in 1:10){  
    RRF.data<-RRF(data, as.factor(y),mtry=dim(data)[2], coefReg = coefReg)
    RF<-randomForest(data[RRF.data$feaSet], as.factor(y))
    p1<-RF$votes[,2]
    p1<-sapply(p1, function(x) ifelse(x==0, 1/(2*length(y)), ifelse(x==1, 1-1/(2*length(y)), x)))
    l=sum(y*log(p1)+(1-y)*log(1-p1), na.rm=F)
    aic[i]=-2*l+2*length(RRF.data$feaSet)
  }
  mean(aic)
}

opt1<-optimize(opt.aic, interval=c(0.1,5), imp.norm=q.norm, y=y)
opt2<-optimize(opt.aic, interval=c(0.1,5), imp.norm=q2.norm, y=y2)
opt3<-optimize(opt.aic, interval=c(0.1,5), imp.norm=q3.norm, y=y3)

opt.fn(q2.norm, 1.97, y2)



##known fact
test(c(1:10),y)

par(mfrow=c(1,3))
plot(imp/max(imp),ylim=c(0,1), main="normalized variable importance score")
plot(1-q,ylim=c(0,1),  main="q value")
plot((1-q)*aw, ylim=c(0,1), main="scaled q value")


##################################### regression simulation
data<-data.frame(matrix(rnorm(100*100), nrow=100))
b=seq(0.1, 2.2, 0.2)
##linear
y=b[2]*data$X1+b[3]*data$X2+b[4]*data$X3+b[5]*data$X4+b[6]*data$X5+b[7]*data$X6+b[8]*data$X7+b[9]*data$X8+b[10]*data$X9+b[11]*data$X10
##higher-order term
y2=b[5]*data$X4+b[6]^2*data$X5+b[7]*data$X6+b[9]*data$X8+b[10]*data$X9+b[11]*data$X10+b[9]*data$X11^2
##interaction terms
y3=b[1]+b[5]*data$X4+b[6]^2*data$X5+b[7]*data$X6+b[9]*data$X8+b[10]*data$X9+b[11]*data$X10+b[9]*data$X11*data$X12

##10 runs
test<-function(feature, y){
  mse<-c()
  for (i in 1:10){
    rf<-randomForest(data[feature],y)
    mse[i]<-mean((rf$predicted-y)^2)
  }
  list(mean=mean(mse), sd=sd(mse))
}

test2<-function(feature, x, y){
  mse<-c()
  for (i in 1:10){
    rf<-randomForest(x[feature],y)
    mse[i]<-mean((rf$predicted-y)^2)
  }
  list(mean=mean(mse), sd=sd(mse))
}

##no feature selection
e1<-test(1:100, y)
e1.2<-test(1:100, y2)
e1.3<-test(1:100, y3)
e1.4<-test(1:100, y)  ##different data

##VSURF
var.data<-VSURF(data, y)
e2<-test(var.data$varselect.pred, y)

var.data.4<-VSURF(data, y)  ##different data
e2.4<-test(var.data.4$varselect.pred, y)

var.data.2<-VSURF(data, y2)
e2.2<-test(var.data.2$varselect.pred, y2)

var.data.3<-VSURF(data, y3)
e2.3<-test(var.data.3$varselect.pred, y3)

##RRF
imp<-randomForest(data, y)$importance
coefReg=0.1+0.9*imp/max(imp)
RRF.data<-RRF(data, y, mtry=dim(data)[1],coefReg = coefReg)
e3<-test(RRF.data$feaSet, y)

imp.2<-randomForest(data, y2)$importance
coefReg.2<-0.1+0.9*imp.2/max(imp.2)
RRF.data.2<-RRF(data, y2, mtry=dim(data)[1],coefReg = coefReg.2)
e3.2<-test(RRF.data.2$feaSet, y2)

imp.3<-randomForest(data, y3)$importance
coefReg.3<-0.1+0.9*imp.3/max(imp.3)
RRF.data.3<-RRF(data, y3, mtry=dim(data)[1],coefReg = coefReg.3)
e3.3<-test(RRF.data.3$feaSet, y3)

##RRF_opt
# fn<-function(aw){
#   coefReg=(imp/sum(imp))^aw
#   aic<-c()
#   for (i in 1:5){  
#     RRF.data<-RRF(data,y,mtry=dim(data)[1], coefReg = coefReg)
#     RF<-randomForest(data[RRF.data$feaSet], y)
#     mse<-mean((RF$predicted-y)^2)
#     aic[i]=dim(data)[1]*log(mse)+2*length(RRF.data$feaSet)
#   }
#   return(mean(aic))
# }

##p<-apply(data, 2, function(x)anova(lm(y~x))[1,5])
##q<-p.adjust(p, "BH")
# opt<-optimize(fn, interval=c(0,1))
# aw<-opt$minimum
# RRF.data2<-RRF(data, y, mtry=dim(data)[1], coefReg = (imp/sum(imp))^aw)
# e4<-test(RRF.data2$feaSet, y)

opt.fn<-function(imp.norm, aw, y){
  coefReg=imp.norm^aw
  candidate=list()
  aic=c()
  for (i in 1:10){  
    RRF.data<-RRF(data, y,mtry=dim(data)[2], coefReg = coefReg)
    RF<-randomForest(data[RRF.data$feaSet], y)
    mse<-mean((RF$predicted-y)^2)
    aic[i]=dim(data)[1]*log(mse)+2*length(RRF.data$feaSet)
    candidate[[i]]=RRF.data$feaSet
  }
  list(aic=mean(aic), min=min(aic), max=max(aic),candidate=candidate)
}

opt.aic<-function(imp.norm, aw, y){
  coefReg=imp.norm^aw
  candidate=list()
  aic=c()
  for (i in 1:10){  
    RRF.data<-RRF(data, y,mtry=dim(data)[2], coefReg = coefReg)
    RF<-randomForest(data[RRF.data$feaSet], y)
    mse<-mean((RF$predicted-y)^2)
    aic[i]=dim(data)[1]*log(mse)+2*length(RRF.data$feaSet)
  }
  mean(aic)
}

opt.aic2<-function(feature, y){
  aic=c()
  for (i in 1:5){  
    RF<-randomForest(data[feature], y)
    mse<-mean((RF$predicted-y)^2)
    aic[i]=dim(data)[1]*log(mse)+2*length(RRF.data$feaSet)
  }
  mean(aic)
}

imp1.norm=imp/max(imp)
can1=list()
min1=max1=aic1=c()
aw=c(0.1, 0.2, 0.4, 0.8, 1:10)
for (j in 1:14){
  result=opt.fn(imp1.norm, aw[j], y)
  aic1[j]=result$aic
  min1[j]=result$min
  max1[j]=result$max
  can1[[j]]=result$candidate
}
temp=data.frame(delta=aw, AIC=aic1, min=min1, max=max1)
ggplot(temp, aes(delta, AIC, ymin=min, ymax=max))+geom_pointrange()+
  labs(title="Linear Regression",x=expression(delta))+theme(plot.title = element_text(hjust = 0.5))
table=table(unlist(can1[[4]]))
win=table[table>7]
test(paste("X", names(win), sep=""), y)


temp<-opt.fn(imp1.norm, 0.69,y)
table(unlist(temp$candidate))
a=opt.aic2(paste("X", c(3,4,6:8, 10), sep=""), y)
b=opt.aic2(paste("X", c(3,4,6:8, 10, 94), sep=""), y)
plot(c(1,0.4), c(a,b), xlab="Stability", ylab="AIC", main="Stability Test in Scenario 4")


test2(paste("X", c(1:100), sep=""), x.test, y.test)
test2(paste("X", c(6:10), sep=""), x.test, y.test)
test2(paste("X", c(3,4,6:8, 10), sep=""), x.test, y.test)
test2(paste("X", c(3,4,6:10), sep=""), x.test, y.test)
test2(paste("X", c(1:10), sep=""), x.test, y.test)



test2(paste("X", c(1:100), sep=""), x.test, y2.test)
test2(paste("X", c(8, 10, 11), sep=""), x.test, y2.test)
test2(paste("X", c(6,8:11,41,51), sep=""), x.test, y2.test)
test2(paste("X", c(8, 10, 11), sep=""), x.test, y2.test)
test2(paste("X", c(4:11), sep=""), x.test, y2.test)


test2(paste("X", c(1:100), sep=""), x.test, y3.test)
test2(paste("X", c(6,8:10), sep=""), x.test, y3.test)
test2(paste("X", c(6,8:11,51, 93), sep=""), x.test, y3.test)
test2(paste("X", c(6,8:10), sep=""), x.test, y3.test)
test2(paste("X", c(4:12), sep=""), x.test, y3.test)


opt<-optimize(opt.aic, interval=c(0.1,5), imp.norm=imp1.norm, y=y)
opt<-optimize(opt.aic, interval=c(0.1,5), imp.norm=imp2.norm, y=y2)
opt<-optimize(opt.aic, interval=c(0.1,5), imp.norm=imp3.norm, y=y3)


imp2.norm=imp.2/max(imp.2)
can2=list()
min2=max2=aic2=c()
aw=c(0.1, 0.2, 0.4, 0.8, 1:10)
for (j in 1:14){
  result=opt.fn(imp2.norm, aw[j], y2)
  aic2[j]=result$aic
  min2[j]=result$min
  max2[j]=result$max
  can2[[j]]=result$candidate
}
temp=data.frame(delta=aw, AIC=aic2, min=min2, max=max2)
ggplot(temp, aes(delta, AIC, ymin=min, ymax=max))+geom_pointrange()+
  labs(title="Regression with higher-order term",x=expression(delta))+theme(plot.title = element_text(hjust = 0.5))
table=table(unlist(can2[[6]]))
win=table[table>9]
test(paste("X", names(win), sep=""), y2)


temp<-opt.fn(imp2.norm, 2.75,y2)
table(unlist(temp$candidate))



imp3.norm=imp.3/max(imp.3)
can3=list()
min3=max3=aic3=c()
aw=c(0.1, 0.2, 0.4, 0.8, 1:10)
for (j in 1:14){
  result=opt.fn(imp3.norm, aw[j], y3)
  aic3[j]=result$aic
  min3[j]=result$min
  max3[j]=result$max
  can3[[j]]=result$candidate
}
temp=data.frame(delta=aw, AIC=aic3, min=min3, max=max3)
ggplot(temp, aes(delta, AIC, ymin=min, ymax=max))+geom_pointrange()+
  labs(title="Regression with interaction term",x=expression(delta))+theme(plot.title = element_text(hjust = 0.5))
table=table(unlist(can3[[4]]))
win=table[table>9]
test(paste("X", names(win), sep=""), y3)

temp<-opt.fn(imp3.norm, 0.98,y3)
table(unlist(temp$candidate))
a=opt.aic2(paste("X", c(6,8:10), sep=""), y3)
b=opt.aic2(paste("X", c(6,8:10,51), sep=""), y3)
plot(c(1,0.9), c(a,b), xlab="Stability", ylab="AIC", main="Stability Test in Scenario 6")

##coefficient plot
par(mfrow=c(2,3))
plot(imp1.norm, ylab="VI'", xlab="variable", main="linear")
plot(imp2.norm, ylab="VI'", xlab="variable", main="polynomial")
plot(imp3.norm, ylab="VI'", xlab="variable", main="interaction")
plot(imp1.norm^0.6, ylab="VI'^0.6", xlab="variable", main="linear")
plot(imp2.norm^2, ylab="VI'^2", xlab="variable", main="polynomial")
plot(imp3.norm^0.8, ylab="VI'^0.8", xlab="variable", main="interaction")

# fn.2<-function(aw){
#   coefReg=(imp.2/sum(imp.2))^aw
#   aic<-c()
#   for (i in 1:5){  
#     RRF.data<-RRF(data,y2,mtry=dim(data)[1], coefReg = coefReg)
#     RF<-randomForest(data[RRF.data$feaSet], y2)
#     mse<-mean((RF$predicted-y2)^2)
#     aic[i]=dim(data)[1]*log(mse)+2*length(RRF.data$feaSet)
#   }
#   return(mean(aic))
# }
# ##p.2<-apply(data, 2, function(x)anova(lm(y2~x))[1,5])
# ##q.2<-p.adjust(p.2, "BH")
# opt.2<-optimize(fn.2, interval=c(0,1))
# aw.2<-opt.2$minimum
# RRF.data2.2<-RRF(data, y2, mtry=dim(data)[1], coefReg = (imp.2/sum(imp.2))^aw.2)
# e4.2<-test(RRF.data2.2$feaSet, y2)
# 
# 
# fn.3<-function(aw){
#   coefReg=(imp.3/sum(imp.3))^aw
#   aic<-c()
#   for (i in 1:5){  
#     RRF.data<-RRF(data,y3,mtry=dim(data)[1], coefReg = coefReg)
#     RF<-randomForest(data[RRF.data$feaSet], y3)
#     mse<-mean((RF$predicted-y3)^2)
#     aic[i]=dim(data)[1]*log(mse)+2*length(RRF.data$feaSet)
#   }
#   return(mean(aic))
# }
# ##p.3<-apply(data, 2, function(x)anova(lm(y3~x))[1,5])
# ##q.3<-p.adjust(p.3, "BH")
# opt.3<-optimize(fn.3, interval=c(0,1))
# aw.3<-opt.3$minimum
# RRF.data2.3<-RRF(data, y3, mtry=dim(data)[1], coefReg = (imp.3/sum(imp.3))^aw.3)
# e4.3<-test(RRF.data2.3$feaSet, y3)
# 
# ##known fact
# test(c(1:10),y)
