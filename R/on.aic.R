on.aic <-
function(par, wt, num=1, iter=1) {	
	if(num == 1) {
		return(rrf.opt.1(par, wt, iter)[1]);
#	} else if(num == 2) {
#		return(rrf.opt(par, wt[, 1], wt[, 2], iter)[1]);
	} else {
		return(rrf.opt.m(par, wt, iter)[1]);
	}
}
