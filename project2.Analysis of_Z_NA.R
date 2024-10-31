library(MCMCpack)
library(mvnfast)
library(tidyverse)
library(Rcpp)
library(RcppArmadillo)

sourceCpp("C:/Users/kimm4/Desktop/WORKSPACE/2nd 3rd paper/DATASET/Basic_training/project2.DPcpp.MAR.cpp")
#sourceCpp("project2.DPcpp.MAR.cpp")


# Read in data
insurance = read.csv("C:/Users/kimm4/Desktop/WORKSPACE/2nd 3rd paper/DATASET/LGPF_MAR.csv")
#insurance = read.csv("LGPF_MAR.csv")

insurance = select(.data=insurance, Year, Total_Losses, Ln.Cov._1, Fire5_1.original, Fire5_2); head(insurance)

colnames(insurance) = c("year", "loss","ln_coverage","protectorig","protectmiss"); head(insurance)
summary(insurance)
summary(insurance$protectorig-insurance$protectmiss)
#which(insurance$protectorig-insurance$protectmiss==1) # Look into observation 2441 ???????? ghost ??? no idea ?????




jpeg(file="saving_plot2.jpeg",width=1000, height=800)





hist(insurance$loss)
hist(log(insurance$loss+1))
hist(log(insurance$loss[insurance$loss>0]))
dev.off()


#insurance$loss = insurance$loss+1 # Y = Total Loss + 1 ... to avoid log(0) ..so when prediction, plz substract 1.
head(insurance)





# We use the data in the first 4 years, namely 2006-2009, to develop the model.
# keep the obv in the final year (2010) for validation purposes. OK. now, let's split train and test
train.df <- subset(x=insurance, subset=year<2010); head(train.df)
summary(train.df)

test.df <- subset(x=insurance, subset=year==2010); head(test.df)
summary(test.df)



# Finally,...Define response and covariates for the training set.
# Y = ifelse(train.df$loss==0,0,log(train.df$loss))
Y = train.df$loss
X = select(.data=train.df, protectmiss, ln_coverage) # X1 = Protection class (binary), X2 = Coverage (continuous)
X1 = X$protectmiss
X1miss = is.na(X1)
X2 = scale(X$ln_coverage)
matX = as.matrix(cbind(1, X1, X2))
summary(X1)
summary(X2)
n = length(Y)





# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MODELLING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< #
################################################################################
# before moving forward.. we have "zero+1" inflated data..so..Y ~ delta*I(Y=1) + (1-delta)*likelihood*I(Y>1)


############# ex) Modeling with Two Kinds of Claims ### With [GLM] #############
# E[Y|X] = E[Y|X, Y = 0]*p(Y = 0|X) + E[Y|X, Y > 0]*p(Y > 0|X)
#            model(A)=1     delta         model(B)     1-delta
# Considering different regressions on the different subset ? 
################################################################################

# > A. delta (weight) model with [GLM] for p(Y = 0|X) and p(Y > 0|X)
train.df$Zero <- (train.df$loss==0); head(train.df)
#mean(train.df$Zero) # 71.8% fall into the zero...

# This is a logistic regression for "delta"
fit_w <- glm(Zero ~ factor(X1) + X2, data=train.df, family=binomial(link="logit")); summary(fit_w)  
#fit_w$fitted.values # size:3054....out of 4529..where are the other 1475 obv with NA? ignored..
#length(fit_w$fitted.values)
#predict(fit_w, type="response", se=T)$fit # size:3054....out of 4529..where are the other 1475 obv with NA? ignored..
#length( predict(fit_w, type="response", se=T)$fit ) 


# The delta: p(Y > 0|X): Sigmoid
sigmoid = function(x) {
  1/(1+exp(-x))
} 
curve(sigmoid(x), from=-5, to=5)




# > B. model (A),(B): E[Y|X, Y = 0] and E[Y|X, Y > 0]

# model(A) for zero Y
# is equal to 1.....................................???
# E[Y|X, Y = 0] = 1 

# model(B) for non-zero Y
# E[Y|X, Y > 0] = ?
# .... LogSkewNormal.....too tiny....................???
dlogsknorm = function(y, x, beta, sig2, xi) {
  mu = sum(x*beta)
  sig = sqrt(sig2)
  z = (log(y)-mu)/sig
  return(2/(y*sig)*dnorm(z)*pnorm(xi*z))
}
# .... Log of LogSkewNormal...better.................???
dlogsknorm_log = function(y, x, beta, sig2, xi) {
  mu = sum(x*beta)
  sig = sqrt(sig2)
  z = (log(y)-mu)/sig
  return(log(2)-log(y*sig) + dnorm(z, log = TRUE) + pnorm(xi*z, log = TRUE))
}


# dsknorm = function(y, x, beta, sig2, xi) {
#   mu = sum(x%*%beta)
#   sig = sqrt(sig2)
#   z = (y-mu)/sig
#   return(2/(sig)*dnorm(z)*pnorm(xi*z))
# }
# # .... Log of LogSkewNormal...better.................???
# dsknorm_log = function(y, x, beta, sig2, xi) {
#   mu = sum(x%*%beta)
#   sig = sqrt(sig2)
#   z = (y-mu)/sig
#   return(log(2)-log(sig) + dnorm(z, log = TRUE) + pnorm(xi*z, log = TRUE))
# }




# ------------------------ SAVE FOR MAJOR RIVAL ------------------------------ #
# then...Initialize the beta coefficients for the outcome parameter... 
# > C. Tweedie distribution for a given range of shape parameters (1<p<2) have a point mass at zero and 
# a skewed positive distribution for Y>0

#library(tweedie)
# MLE of Tweedie prameter "p" so....Find "p"!!!!!
#out <- tweedie.profile( log(Y) ~ factor(X1) + X2, p.vec=seq(1.05, 1.95, by=.05) )
#out$p.max #p=1.05
#plot(out, type="b")
#abline(v=out$p.max, lty=2, col="red")


# tweedie glm
#library(statmod)
#fit1 <- glm( log(Y) ~ factor(X1) + X2, family = tweedie(var.power=1.05, link.power=0) )
#summary(fit1)

# but..what is var.power?..
# Tweedie glms assume that the variance function is a power function. 
# Special case include 
# - normal : var.power=0 ... so :::: var(Y)=mu^p * p
# - poisson: var.power=1
# - gamma  : var.power=2
# - invnorm: var.power=3

# link.power=0 is the log link ... so :::: log(mu)=Xb
# link.power=1 is the identity link

#fit1$fitted.value
#hist(fit1$fitted.values)
#predict.glm(fit1, type="response")
#hist(predict.glm(fit1, type="response")) 
# ---------------------------------------------------------------------------- #









################################################################################
### Step00> Define prior
################################################################################

# ::::::: # -------------- PRIOR ------------------ # ::: for Y~LSN( x*"beta", sd(Y), xi )
#                                                     ::: for Y~delta( x*"beta_tilde" )
#-------------------------------------------------------------------------------
# ------------- Outcome -- ( beta_j, sig2_j, xi_j, betat_j ) -------------------
fit1 <- glm( log(Y[!train.df$Zero]) ~ factor(X1[!train.df$Zero]) + X2[!train.df$Zero] ) # Gaussian regression for initialize OUTCOME model parameter beta0, SIG0
# fit1 <- glm( Y[!train.df$Zero] ~ factor(X1[!train.df$Zero]) + X2[!train.df$Zero] ) # Gaussian regression for initialize OUTCOME model parameter beta0, SIG0
summary(fit1)

# fit1$fitted.value
# hist(fit1$fitted.values)
# predict.glm(fit1, type="response")
# hist(predict.glm(fit1, type="response")) 
#boxplot(fit1$fitted.value ~ train.df$Zero[!is.na(train.df$protectmiss)]) # F:(Sh>0), T:(Sh=0) excluding NA
# Note that the predicted value in a GLM is a mean. For any distribution on non-negative values, to predict a mean 
#of 0, its distribution would have to be entirely a spike at 0.


# ::: for "tilde_bj"~MVN( betat0, SIG_bt0 ): keep beta0, SIG_b0 from coefficient for logistic glm to handle zero
betat0 = coef(fit_w)
SIG_bt0 = vcov(fit_w)
SIG_bt0inv = solve(a=SIG_bt0)

# ::: for "beta_j" ~ MVN( beta0, sig2_j*SIG_b0 ): keep beta0, SIG_b0, but sample sig2_j
a0 = 5    # ::: for "sig2_j" ~[ IG(a0, b0) ]  
b0 = 0.25    

beta0 = coef(fit1)          # 1x3 initial reg_coeff vector (sort of "means"): Regression result as "mean"
SIG_b0 = vcov(fit1)         # 3x3 initial cov matrix of reg_coeff
#SIG_b0inv = solve(a=SIG_b0) # inverse of cov matrix of reg_coeff for later use (for posterior on reg_coeff)!!
varinf = n/5
SIG_b0*varinf

# ::: for "xi_j"~t(loc, nu0, sca)
loc = 0   # ? Location hyperparameter for T distribution
nu0 = 1/2 # df for T distribution on pigtail parameter
sca = 5   # ? Scale hyperparameter for T distribution 


#-------------------------------------------------------------------------------
# ------------ Covariates -- ( "prop_j", "mu_j", "tau2_j" ) --------------------
c0 = 0.5  # Main-param: "prop_j" ~[ Beta(c0, d0) ]     ::: for X1~Bin( n, "prob_j" )
d0 = 0.5   

mu0 = 0   # Main-param: "mu_j" ~[ N(mu0, tau0) ]       ::: for X2~N( "mu_j", tau2_j ) 
tau0 = 1  

e0 = 1    # Main-param: "tau2_j" ~[ IG(e0, gam0) ]      ::: for X2~N( mu_j, "tau2_j" )
gam0 = 1    

# -- precision -- "alpha"
g0 = 1    #                                          ::: for "alpha"~Ga( g0, h0 )
h0 = 1    






################################################################################
### Step01> initialize cluster membership - Hierarchical clustering
################################################################################
J=3
# clusters = cutree(hclust( dist(cbind(Y,X1,X2)) ), J)

Y1 = ifelse(Y==0, 0, log(Y))
clusters = cutree(hclust( dist(cbind(Y1,X1,X2)) ), J)
cl_membership = clusters 
table(cl_membership)

# plot( hclust(dist(cbind(log(Y),X1,X2))) )
# rect.hclust(hclust(dist(cbind(log(Y),X1,X2))) , k = 3, border = 2:6)
# abline(h = 3, col = 'red')

# library(dendextend)
# avg_dend_obj <- as.dendrogram( hclust(dist(cbind(log(Y),X1,X2))) )
# avg_col_dend <- color_branches(avg_dend_obj, h = 3)
# plot(avg_col_dend)
# 
# library(dplyr)
# df_cl <- mutate(.data=train.df, cluster=clusters)
# count(df_cl, cluster) # count how many observations were assigned to each cluster ?
# library(ggplot2)
# ggplot(df_cl, aes(x=ln_coverage, y = log(loss), color = factor(cluster))) + geom_point()







################################################################################
### Step02> initialize cluster parametersssss 
#           Sample some parameters, using your POSTERIOR
#                                         Based on hierarchical clustering: J=3
################################################################################

# ------------------------ for Covariates + alpha ------------------------------

##[A] for X1~Bin( n, "prob_j" ) .. starting with 3 clusters for beta POSTERIOR
set.seed(1)

piparam = numeric(J)
for (j in 1:J) {
  Xj=X1[cl_membership==j & !X1miss]
  nj=length(Xj)
  if(nj > 0) { #sample from posterior
    piparam[j] = rbeta( n=1, shape1=c0+sum(Xj), shape2=d0+nj-sum(Xj) ) 
  } 
  else { #sample from prior
    piparam[j] = rbeta(n=1, shape1 = c0, shape2 = d0)                  
  }
}
piparam #% pi parameter sampled from posterior

#% [Check!] empirical pi ? using "aggregate(.)": investigation by splitting the data into subset
pi_empirical = aggregate(x=X1, by=list(cl_membership), FUN=mean, na.rm= T)$x; pi_empirical
piparam - pi_empirical 


##[B] for X2~N( "mu_j", "tau2_j" ) .. starting with 3 clusters for Normal/IG POSTERIOR
muparam = numeric(J); muparam
tau2param = numeric(J); tau2param
for (j in 1:J) {
  Xj=X2[cl_membership==j]
  nj=length(Xj)
  if(nj >0) { # sample from the posterior
    Xjbar=mean(Xj)
    tau2param[j]=rinvgamma( n=1, shape=e0+nj/2, scale=gam0+0.5*( nj*(Xjbar-mu0)^2/(nj+1) + sum((Xj-Xjbar)^2) ) )
    muparam[j]=rnorm( n=1, mean=(nj*Xjbar+mu0)/(nj+1), sd=sqrt(tau2param[j]/(nj+1)) )     
  }
  else { # sample from the prior
    tau2param[j] = rinvgamma(n=1, shape = e0, scale = gam0)
    muparam[j]=rnorm(n=1, mean=mu0, sd = sqrt(tau2param[j]))
  }
}
muparam
tau2param

#% [Check!] empirical mu, sigma ? using "aggregate(.)": investigation by splitting the data into subset
mu_empirical = aggregate(x=as.numeric(X2), by=list(cl_membership), FUN=mean, na.rm=T)$x; mu_empirical
tau2_empirical = aggregate(x=as.numeric(X2), by=list(cl_membership), FUN=var, na.rm=T)$x; tau2_empirical


##[C] for alpha~Ga( shape=g0, rate=h0 ) .. starting with 3 clusters for Mixed Gamma POSTERIOR
alpha0=2 # initialization: typically..1,2...
eta = rbeta(n=1, shape1=alpha0+1, shape2=n); eta
pi_eta = (g0+J-1)/(g0+J-1+n*(h0-log(eta))); pi_eta

# precision parameters and its done!
alpha = pi_eta*rgamma( n=1, shape=g0+J, 
                       rate=h0-log(eta) ) + (1-pi_eta)*rgamma(n=1, shape=g0+J-1, rate=h0-log(eta) ); alpha







# ---------------------------- for Outcome -------------------------------------

##[D] Outcome parameters:  beta_j, sig2_j, xi_j, betat_j 

# No conjugacy...so prepare MM sampling
# This is where new parameters go.
beta_j = matrix(data=NA, nrow=J, ncol=length(beta0)) 
sig2_j = numeric(J) 
xi_j = numeric(J)
betat_j = matrix(data=NA, nrow=J, ncol=length(betat0))

# Set in place the old parameters..sample from prior...
beta_old = matrix( rep(rmvn(n=1, mu=beta0, sigma=SIG_b0*varinf), J), nrow=J, byrow= TRUE )
sig2_old = rep(rinvgamma(n=1, shape=a0, scale=b0), J)
xi_old = rep(rt(n=1, df=nu0), J) # imagine we already have them...old days..
betat_old = matrix( rep(rmvn(n=1, mu=betat0, sigma=SIG_bt0*varinf), J), nrow=J, byrow= TRUE )


#### Metropolis Hastings to update beta_j, sig2_j, xi_j, beta_j
for(j in 1:J) {
  # Sample proposals from priors
  beta_p = rmvn(n=1, mu=beta0, sigma=SIG_b0*varinf)
  sig2_p = rinvgamma(n=1, shape=a0, scale=b0)
  xi_p = rt(n=1, df=nu0)
  betat_p = rmvn(n=1, mu=betat0, sigma=SIG_bt0*varinf)
  
  # subsetting by cluster
  Yj = Y[cl_membership==j]
  matXj = matX[cl_membership==j, ]
  X1missj = X1miss[cl_membership==j]
  missindx = which(X1missj)
  
  # > In case: there is any NA.....in X1, Finish Imputation X1 beforehand....
  # p0: joint where x1 = 0
  # p1: joint where x1 = 1
  for(i in missindx) {
    # if (Yj[i]>1) {
    if(Yj[i]>0) {
      p0log = 
        dlogsknorm_log(y=Yj[i],
                       x=t(as.matrix(c(matXj[i,1], 0, matXj[i,3]), nrow=1)),
                       beta=beta_old[j,],
                       sig2=sig2_old[j],
                       xi=xi_old[j]) +
        dbinom(x=0, size=1, piparam[j], log=TRUE) +
        log( 1-sigmoid( sum(c(matXj[i,1], 0, matXj[i,3])*betat_old[j, ]) ) )
        # dsknorm_log(y=Yj[i], 
        #                x=t(as.matrix(c(matXj[i,1], 0, matXj[i,3]), nrow=1)), 
        #                beta=beta_old[j,], 
        #                sig2=sig2_old[j], 
        #                xi=xi_old[j]) + 
        # dbinom(x=0, size=1, piparam[j], log=TRUE) + 
        # log( 1-sigmoid( sum(c(matXj[i,1], 0, matXj[i,3])*betat_old[j, ]) ) )
      
      p1log = 
        dlogsknorm_log(Yj[i],
                       x=t(as.matrix(c(matXj[i,1], 1, matXj[i,3]), nrow=1)),
                       beta=beta_old[j,],
                       sig2=sig2_old[j],
                       xi=xi_old[j]) +
        dbinom(x=1, size=1, piparam[j], log=TRUE) +
        log( 1-sigmoid( sum(c(matXj[i,1], 1, matXj[i,3])*betat_old[j, ]) ) )
        # dsknorm_log(Yj[i], 
        #                x=t(as.matrix(c(matXj[i,1], 1, matXj[i,3]), nrow=1)), 
        #                beta=beta_old[j,], 
        #                sig2=sig2_old[j], 
        #                xi=xi_old[j]) + 
        # dbinom(x=1, size=1, piparam[j], log=TRUE) + 
        # log( 1-sigmoid( sum(c(matXj[i,1], 1, matXj[i,3])*betat_old[j, ]) ) )
    }
    else {
      p0log = 
        dbinom(x=0, size=1, piparam[j], log=TRUE) + # (1 - pi_j)
        log( sigmoid( sum(c(matXj[i,1], 0, matXj[i,3])*betat_old[j, ]) ) )
      
      p1log = 
        dbinom(x=1, size=1, piparam[j], log=TRUE) + # (pi_j)
        log( sigmoid( sum(c(matXj[i,1], 1, matXj[i,3])*betat_old[j, ]) ) )
    }
    
    # Let's impute!!!
    matXj[i,2] = rbinom(n = 1, size = 1, prob = 1/(1+exp(p0log-p1log))) #imputing NA with using posterior Pi 
  }
  # if there is no NA...we don't need any imputation...obviously...     :::::: Imputation Done! ::::::
  
  # > MH algorithm
  # prepare components
  numerator=0
  denominator=0
  for (i in 1:length(Yj)){
    # if(Yj[i]>1){
    if(Yj[i]>0){
      numerator = numerator + 
        log( 1-sigmoid(sum(matXj[i,]*betat_p)) ) + 
        dlogsknorm_log(Yj[i], matXj[i,], t(as.matrix(beta_p)), sig2_p, xi_p)
        # dsknorm_log(Yj[i], matXj[i,], t(as.matrix(beta_p)), sig2_p, xi_p)
      denominator = denominator + 
        log( 1-sigmoid(sum(matXj[i,]*betat_old[j,])) ) + 
        dlogsknorm_log(Yj[i], matXj[i,], t(as.matrix(beta_old[j,])), sig2_old[j], xi_old[j])
        # dsknorm_log(Yj[i], matXj[i,], t(as.matrix(beta_old[j,])), sig2_old[j], xi_old[j])
    }
    else {
      numerator = numerator + 
        log( sigmoid(sum(matXj[i,]*betat_p)) ) 
      denominator = denominator + 
        log( sigmoid(sum(matXj[i,]*betat_old[j,])) )  
    }
  }
  # compute the ratio
  ratio = min(exp(numerator-denominator), 1)
  
  U = runif(n = 1, min = 0, max = 1)
  if(U < ratio) {
    beta_j[j, ] = beta_p
    sig2_j[j] = sig2_p
    xi_j[j] = xi_p
    betat_j[j, ] = betat_p
  } 
  else {
    beta_j[j, ] = beta_old[j, ]
    sig2_j[j] = sig2_old[j]
    xi_j[j] = xi_old[j]
    betat_j[j, ] = betat_old[j, ]
  }
}

beta_j   #(J,p) : membership(J), predictors(p)
beta_old #(J,p)
sig2_j   #(J)
sig2_old #(J)
xi_j     #(J)
xi_old   #(J)
betat_j  #(J,p)
betat_old#(J,p)


#%%%%%%%%%%%%%%%%%%%%%%% - Do Not Touch - when re-run %%%%%%%%%%%%%%%%%%%%%%%%%#
#####################################################################################
### Step03> Data model development
#           Using sampled parameters above, ---- discrete / continuous
#####################################################################################

### [discrete Data model]
#
# covariate w/o NA: ----------> X1,X2
#-> dbinom(x, size, prob=piparam)*dnorm(x, mean=muparam, sd=tau2param)
# covariate with NA in X1: ---> X2
#-> dnorm(x, mean=muparam, sd=tau2param)



# Outcome w/o NA: ------------> Sh|X1,X2
#-> dlogsknorm_log(y, x, beta=beta_j, sig2=sig2_j, xi=xi_j)
# outcome with NA: -----------> Sh|X1,X2 then marginalize w.r.t X1
#-> dlogsknorm_log(y, x, beta=beta_j, sig2=sig2_j, xi=xi_j)



### [continuous (Parameter-free) Data model]
#
# covariate model w/o NA: ----------> int X1,X2 w.r.t "piparam","muparam","tau2param"
f0x1 = function(x1) {
  beta(x1+c0, 1-x1+d0)/beta(c0,d0)
}
f0x2 = function(x2) {
  gam0^e0*gamma(e0+1/2)/(2*sqrt(pi)*gamma(e0))*(gam0+(x2-mu0)^2/4)^(-(e0+1/2))
}
#::: f0x = f0x1(X1)*f0x2(X2)


# covariate model with NA in X1: ---> int X2 w.r.t "muparam","tau2param" 
# f0x2 = function(x2) {
#   gam0^e0*gamma(e0+1/2)/(2*sqrt(pi)*gamma(e0))*(gam0+(x2-mu0)^2/4)^(-(e0+1/2))
# }
#::: f0x = f0x2




# Outcome ...with+w/o NA: ------------------------------> MonteCarlo Integration
# Calculate Outcome and Covariate parameter free data model for each observation  
n=length(Y)
f0y = numeric(n) #% param-free outcome model f0(y|x)
f0x = numeric(n) #% param-free covariate model
E0y = numeric(n) # E(Y|x) = Expected value of Y|x ~ f0(y|x)


set.seed(1)
M = 1000 # Number of Monte Carlo samples
for(i in 1:n) {
  if(!is.na(X1[i])) {
    f0x[i] = f0x1(X1[i])*f0x2(X2[i])  #### Look at your covariate model #####
    
    # Monte Carlo integration for Y (w/o NA) for ### outcome model ###
    sumy = numeric(M)
    sumEy = numeric(M)
    for(j in 1:M) {
      xi_samplej = rt(n = 1, df = nu0)                            # prior on xi
      sig_samplej = rinvgamma(n = 1, shape = a0, scale = b0)      # prior on sig2
      beta_samplej = rmvn(n = 1, mu = beta0, sigma = SIG_b0*varinf)      # prior on beta
      betat_samplej = rmvn(n = 1, mu = betat0, sigma = SIG_bt0*varinf)   # prior on tilde beta
      
      # if(Y[i]>1){    # Y>1, when Sh>0
      if(Y[i]>0){    # Y>1, when Sh>0
        sumy[j] = 
          (1-sigmoid( sum(matX[i,]*betat_samplej) ))*             # P(Sh > 0) with complete
          dlogsknorm( y=Y[i],
                    x=matX[i,],
                    beta=beta_samplej,
                    sig2=sig_samplej,
                    xi=xi_samplej )*                              # to outcome with complete
          # dsknorm( y=Y[i],
          #           x=matX[i,],
          #           beta=beta_samplej,
          #           sig2=sig_samplej,
          #           xi=xi_samplej )*
          dmvn(X = beta_samplej, mu = beta0, sigma = SIG_b0*varinf)*     # to joint beta
          dinvgamma(x = sig_samplej, shape = a0, scale = b0)*     # to joint sig2
          dt(x = xi_samplej, df = nu0)*                           # to joint xi
          dmvn(X = betat_samplej, mu = betat0, sigma = SIG_bt0*varinf)   # to joint beta tilde
        sumEy[j] = (1-sigmoid( sum(matX[i,]*betat_samplej) ))*2*exp(sum(matX[i,]*beta_samplej)+sig_samplej/2)*(1-pnorm(-xi_samplej*sqrt(sig_samplej)/sqrt(xi_samplej^2+1)))
        
      }
      else {        # Y=1, when Sh=0
        sumy[j] = 
          (sigmoid( sum(matX[i,]*betat_samplej) ))*                             # P(Sh = 0) with complete
          dmvn(X = betat_samplej, mu = betat0, sigma = SIG_bt0*varinf)   # to joint beta tilde
        sumEy[j] = 0
      }
      
      
      
      
    }
    E0y[i] = sum(sumEy)/M
    f0y[i] = sum(sumy)/M # Outcome model w/o NA in X1
  } 
  else if(is.na(X1[i])) {
    f0x[i] = f0x2(X2[i])  #### Look at your covariate model #####
    
    # Monte Carlo integration for Y (with NA) ### outcome model ###
    sumy = numeric(M)
    sumEy = numeric(M)
    for(j in 1:M) {
      xi_samplej = rt(n = 1, df = nu0)                            # prior for xi
      sig_samplej = rinvgamma(n = 1, shape = a0, scale = b0)      # prior for sig2
      beta_samplej = rmvn(n = 1, mu = beta0, sigma = SIG_b0*varinf)      # prior for beta
      betat_samplej = rmvn(n = 1, mu = betat0, sigma = SIG_bt0*varinf)   # prior on tilde beta
      # In addition.....
      pi_samplej = rbeta(n = 1, shape1 = c0, shape2 = d0)         # To integrate over the missing covariate!!!!
      
      # if(Y[i]>1){    # Y>1, when Sh>0
      if(Y[i]>0){    # Y>1, when Sh>0
        sumy[j] = 
          ( (1-sigmoid( sum(c(matX[i,1], 1, matX[i,3])*betat_samplej) ))*       # P(Sh > 0) with NA
              dlogsknorm( y=Y[i],
                         x=c(matX[i,1], 1, matX[i,3]),
                         beta=beta_samplej,
                         sig2=sig_samplej,
                         xi=xi_samplej )*pi_samplej +
              # dsknorm( y=Y[i],                                 
              #             x=c(matX[i,1], 1, matX[i,3]), 
              #             beta=beta_samplej, 
              #             sig2=sig_samplej, 
              #             xi=xi_samplej )*pi_samplej +
              (1-sigmoid( sum(c(matX[i,1], 0, matX[i,3])*betat_samplej) ))*
              dlogsknorm( y=Y[i],
                          x=c(matX[i,1], 0, matX[i,3]),
                          beta=beta_samplej,
                          sig2=sig_samplej,
                          xi=xi_samplej)*(1 - pi_samplej) )*       # to outcome with NA
              # dsknorm( y=Y[i], 
              #             x=c(matX[i,1], 0, matX[i,3]), 
              #             beta=beta_samplej, 
              #             sig2=sig_samplej, 
              #             xi=xi_samplej)*(1 - pi_samplej) )*       # to outcome with NA
          dmvn(X=beta_samplej, mu=beta0, sigma=SIG_b0*varinf)*            # to joint beta
          dinvgamma(x=sig_samplej, shape=a0, scale=b0)*            # to joint sig2
          dt(x=xi_samplej, df=nu0)*                                # to joint xi
          dbeta(pi_samplej, shape1=c0, shape2=d0)*                 # to joint pi (for x1: NA)
          dmvn(X=betat_samplej, mu=betat0, sigma=SIG_bt0*varinf)          # to joint beta tilde
        sumEy[j] = (1-sigmoid( sum(c(matX[i,1],0,matX[i,3])*betat_samplej) ))*2*exp(sum(c(matX[i,1],0,matX[i,3])*beta_samplej+sig_samplej/2))*(1-pnorm(-xi_samplej*sqrt(sig_samplej)/sqrt(xi_samplej^2+1)))*(1-pi_samplej) +
          (1-sigmoid( sum(c(matX[i,1],1,matX[i,3])*betat_samplej) ))*2*exp(sum(c(matX[i,1],1,matX[i,3])*beta_samplej+sig_samplej/2))*(1-pnorm(-xi_samplej*sqrt(sig_samplej)/sqrt(xi_samplej^2+1)))*pi_samplej
        
      }
      else {
        sumy[j] = 
          ( (sigmoid( sum(c(matX[i,1], 1, matX[i,3])*betat_samplej) ))*         # P(Sh = 0) with NA
              pi_samplej + 
            (sigmoid( sum(c(matX[i,1], 0, matX[i,3])*betat_samplej) ))*
              (1 - pi_samplej) )*                                               
          dbeta(pi_samplej, shape1=c0, shape2=d0)*                 # to joint pi (for x1: NA)
          dmvn(X = betat_samplej, mu = betat0, sigma = SIG_bt0*varinf)    # to joint beta tilde
        sumEy[j] = 0
      }

      
    }
    f0y[i] = sum(sumy)/M # Outcome model with NA in X1
    E0y[i] = sum(sumEy)/M
  }
  print(paste("i=",i))
}

par(mfrow=c(3,1))
plot( x=density(f0y) )
plot( x=density(E0y) )
plot( x=density(f0x) )

summary(E0y)               ### largely zero..obviously
summary(E0y[E0y>0])

E0y.positive = E0y[Y>0]
f0y.positive = f0y[Y>0]

plot( x=density(E0y.positive) )
plot( x=sort(E0y.positive), y=f0y.positive[order(E0y.positive)], type="l", xlim=c(0,1000000))

boxplot(f0x ~ X1miss)




#..............................................................................#
#..............................................................................#
#..............................................................................#
#................................... now ......................................# 
#..............................................................................#
#..............................................................................#
#..............................................................................#
################################################################################
### Step04> Gibbs Sampler --------- cl_membership and param update ---with J= ?
################################################################################
set.seed(1)
total_iter=400
r_convergence = 300 #### After pre-running Gibbs sampler, determine r value for convergence

# loglikelihood = numeric(total_iter)
loglikelihood = matrix(0, nrow = n, ncol = total_iter)

list_piparam = list()   #for X1

list_muparam = list()   #for X2
list_tau2param = list() #for X2

list_alpha = list()     #for alpha

list_beta_j = list()    #for Y
list_sig2_j = list()    #for Y
list_xi_j = list()      #for Y
list_betat_j = list()    #for Y

list_cl = list()

########################################################################################################
# [note] Whenever re-start, plz initialize all parameters other than paramfree densities :f0y, f0x, E0y
########################################################################################################


for (r in 1:total_iter) {
  #----------------------------------------------------------------------------#
  ##############################################################################
  # C++C++C++C++C++C++C++C++C++C++C++ REPLACIBLE ++C++C++C++C++C++C++C++C++C++ #
  #----------------------------------------------------------------------------#
  # ##[1] Updating Cluster Membership -------------------------------------------
  # for (i in i:n){
  #   #cluster_si = cl_membership                              #% current membership vector
  #   nj = as.numeric( table(cl_membership) )
  # 
  #   # a)remove obv and initialize...?
  #   if( nj[cl_membership[i]]==1 ) {               # only 1 observation in this cluster
  #     print("remove cluster")
  #     #% for ..??
  #     j = cl_membership[i]     # j = cluster of observation we're removing
  #     cl_membership[cl_membership>j] = cl_membership[cl_membership>j] - 1          # taking each cluster label - 1 for all clusters greater than j
  # 
  #     #% for X1
  #     piparam = piparam[-j]
  #     #% for X2
  #     tau2param = tau2param[-j]
  #     muparam = muparam[-j]
  #     #% for Y
  #     sig2_j = sig2_j[-j]
  #     beta_j = beta_j[-j,]
  #     xi_j = xi_j[-j]
  #     betat_j = betat_j[-j,]
  # 
  #     J = J-1
  #   }
  # 
  #   cl_membership[i] = 0                                  #% replace the membership value (the first obv) with "0"!
  #   nj = as.numeric( table(cl_membership[cl_membership>0]) ) #% number of observations in each cluster without observation i
  # 
  # 
  # 
  # 
  #   probs = numeric( length(nj)+1 )                    #% for P(s_i=j) ... it's c(31-1,41,28)?? so 3 + 1 ? total cluster number?
  # 
  #   # b)Iterate through each cluster and Calculate probability of staying the same: P(s_i=j)
  #   x_i = c(1, X1[i], X2[i])                                        #% c(1, x1, x2)
  #   if(is.na(X1[i])) {
  #     if(Y[i]==0) {
  #       for(j in 1:length(nj)) {
  #         probs[j] = nj[j]/(n-1+alpha)*( (sigmoid( sum(c(x_i[1], 1, x_i[3])*betat_j[j,]) ))*piparam[j] +
  #                                         (sigmoid( sum(c(x_i[1], 0, x_i[3])*betat_j[j,]) ))*(1 - piparam[j]) )* # Y
  #           dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j])) # Covariate X2
  #       }
  #     } else {
  #       for(j in 1:length(nj)) {
  #         probs[j] = nj[j]/(n-1+alpha)*((1-sigmoid( sum(c(x_i[1], 1, x_i[3])*betat_j[j,]) ))*       # P(Sh > 0) with NA
  #                                         dlogsknorm( y=Y[i],
  #                                                     x=c(x_i[1], 1, x_i[3]),
  #                                                     beta=beta_j[j,],
  #                                                     sig2=sig2_j[j],
  #                                                     xi=xi_j[j] )*piparam[j] +
  #                                         (1-sigmoid( sum(c(x_i[1], 0, x_i[3])*betat_j[j,]) ))*
  #                                         dlogsknorm( y=Y[i],
  #                                                     x=c(x_i[1], 0, x_i[3]),
  #                                                     beta=beta_j[j,],
  #                                                     sig2=sig2_j[j],
  #                                                     xi=xi_j[j])*(1 - piparam[j]))* # Y
  #           dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j])) # Covariate X2
  #       }
  #     }
  #   } else {
  #     if(Y[i]==0) {
  #       for(j in 1:length(nj)) {
  #         probs[j] = nj[j]/(n-1+alpha)*(sigmoid( sum(x_i*betat_j[j,]) ) )*
  #           dbinom(x = x_i[2], size = 1, prob = piparam[j]) * # Covariate X1
  #           dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j]))
  #       }
  #     } else {
  #       for(j in 1:length(nj)) {
  #         probs[j] = nj[j]/(n-1+alpha)*(1-sigmoid( sum(x_i*betat_j[j,]) ) )*
  #           dlogsknorm(y=Y[i], x = x_i, beta = beta_j[j,], sig2 = sig2_j[j], xi = xi_j[j])* # Y
  #           dbinom(x = x_i[2], size = 1, prob = piparam[j]) * # Covariate X1
  #           dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j])) # Covariate X2
  #       }
  #     }
  #   }
  # 
  #   # c)After iteration through each cluster, Adding the new probability of "Forming a new cluster": P(s_i=J+1)
  #   probs[j+1] = alpha/(n-1+alpha)*f0y[i]*f0x[i]   #% so it gives...probs: c(prob, prob, prob, 0) -> c(prob, prob, prob, prob)
  # 
  #   # d)Finally, draw a new cluster for each datapoint from a multinomial distribution
  #   newclust = which( rmultinom(n=1, size=1, prob=probs)==1 ) #% "which(.)" gives the indices of (logic=True)
  #   cl_membership[i] = newclust                       #% assigning the cl_membership to each datapoint. WOW WOW WOW WOW !
  #   #% "Multinomial(.)" is the way the datapt accept/reject the new cluster????
  #   #% to face the truth, probabilities are: "probs/sum(probs)"
  #   #% but.."rmultinom(.)" automatically addresses "probs"
  # 
  #   # e_1)If new cluster is selected by a certain data point,
  #   # then add new value (obtained from new cluster) to the existing parameter pool
  #   if( length(cl_membership[cl_membership==cl_membership[i]]) == 1 ) {
  #     #% for X1, append
  #     if(!is.na(x_i[2])) {
  #       piparam = c( piparam,
  #                    rbeta(n=1, shape1=c0+x_i[2], shape2=d0+1-x_i[2]) ) #posterior if no missing
  #     } else {
  #       piparam = c( piparam,
  #                    rbeta(n=1, shape1=c0, shape2=d0) ) #prior if missing binary covariate
  #     }
  #     #% for X2, append
  #       tau2param = c( tau2param,
  #                      rinvgamma(n=1, shape=e0+1/2, scale=gam0+1/2*(1/2*(x_i[3]-mu0)^2)) )
  #       muparam = c( muparam,
  #                    rnorm(n=1, mean=(x_i[3]+mu0)/2, sd=sqrt(tau2param[J+1]/2)) )
  # 
  # 
  # 
  #     beta_old_j = rmvn(n=1, mu=beta0, sigma=SIG_b0*varinf)
  #     sig2_old_j = rinvgamma(n=1, shape=a0, scale=b0)
  #     xi_old_j = rt(n=1, df=nu0) # imagine we already have them...old days..
  #     betat_old_j = rmvn(n=1, mu=betat0, sigma=SIG_bt0*varinf)
  # 
  #     beta_p = rmvn(n=1, mu=beta0, sigma=SIG_b0*varinf)
  #     sig2_p = rinvgamma(n=1, shape=a0, scale=b0)
  #     xi_p = rt(n=1, df=nu0)
  #     betat_p = rmvn(n=1, mu=betat0, sigma=SIG_bt0*varinf)
  # 
  #     if(is.na(x_i[2])) { #if X1 is missing, then impute
  #       if (Y[i]>0) {
  #         p0log =
  #           dlogsknorm_log(y=Y[i],
  #                          x=t(as.matrix(c(x_i[1], 0, x_i[3]), nrow=1)),
  #                          beta=t(beta_old_j),
  #                          sig2=sig2_old_j,
  #                          xi=xi_old_j) +
  #           dbinom(x=0, size=1, piparam[J+1], log=TRUE) +
  #           log( 1-sigmoid( sum(c(x_i[1], 0, x_i[3])*betat_old_j) ) )
  # 
  #         p1log =
  #           dlogsknorm_log(y=Y[i],
  #                          x=t(as.matrix(c(x_i[1], 1, x_i[3]), nrow=1)),
  #                          beta=t(beta_old_j),
  #                          sig2=sig2_old_j,
  #                          xi=xi_old_j) +
  #           dbinom(x=1, size=1, piparam[J+1], log=TRUE) +
  #           log( 1-sigmoid( sum(c(x_i[1], 1, x_i[3])*betat_old_j) ) )
  #       }
  #       else {
  #         p0log =
  #           dbinom(x=0, size=1, piparam[J+1], log=TRUE) + # (1 - pi_j)
  #           log( sigmoid( sum(c(x_i[1], 0, x_i[3])*betat_old_j) ) )
  # 
  #         p1log =
  #           dbinom(x=1, size=1, piparam[j], log=TRUE) + # (pi_j)
  #           log( sigmoid( sum(c(x_i[1], 1, x_i[3])*betat_old_j) ) )
  #       }
  # 
  #       # Let's impute!!!
  #       x_i[2] = rbinom(n = 1, size = 1, prob = 1/(1+exp(p0log-p1log))) #imputing NA with using posterior Pi
  #     }
  # 
  # 
  # 
  #     # > MH algorithm
  #     # prepare components
  # 
  #     if(Y[i]>0){
  #       numerator =
  #         log( 1-sigmoid(sum(x_i*betat_p)) ) +
  #           dlogsknorm_log(Y[i], x_i, t(as.matrix(beta_p)), sig2_p, xi_p)
  #         denominator =
  #           log( 1-sigmoid(sum(x_i*betat_old_j)) ) +
  #           dlogsknorm_log(Y[i], x_i, t(as.matrix(beta_old_j)), sig2_old_j, xi_old_j)
  #       } else {
  #         numerator =
  #           log( sigmoid(sum(x_i*betat_p)) )
  #         denominator =
  #           log( sigmoid(sum(x_i*betat_old_j)) )
  #       }
  #     # compute the ratio
  #     ratio = min(exp(numerator-denominator), 1)
  # 
  #     U = runif(n = 1, min = 0, max = 1)
  #     if(U < ratio) {
  #       beta_j = rbind(beta_j,beta_p)
  #       sig2_j = c(sig2_j,sig2_p)
  #       xi_j = c(xi_j,xi_p)
  #       betat_j = rbind(betat_j,betat_p)
  #     } else {
  #       beta_j = rbind(beta_j,beta_old_j)
  #       sig2_j = c(sig2_j,sig2_old_j)
  #       xi_j = c(xi_j,xi_old_j)
  #       betat_j = rbind(betat_j,betat_old_j)
  #     }
  # 
  #     J = J+1
  #   }
  # 
  #   # # e_2)If cluster removed then need to remove parameters and renumber clusters
  #   # if( sum(diff(as.numeric(names(table(cl_membership)))) - 1) > 0 ) {               #% ????????
  #   #   print("remove cluster")
  #   #   #% for ..??
  #   #   j = which( (diff(as.numeric( names(table(cl_membership)) )) - 1) > 0 ) + 1     #% ????????
  #   #   cl_membership[cl_membership>=j] = cl_membership[cl_membership>=j] - 1          #% ????????
  #   #
  #   #   #% for X1
  #   #   piparam = piparam[-j]
  #   #   #% for X2
  #   #   tau2param = tau2param[-j]
  #   #   muparam = muparam[-j]
  #   #   #% for Y
  #   #   sig2_j = sig2_j[-j]
  #   #   beta_j = beta_j[-j,]
  #   #   xi_j = xi_j[-j]
  #   #   betat_j = betat_j[-j,]
  #   #
  #   #   J = J-1
  #   # }
  # }
  #----------------------------------------------------------------------------#
  ##############################################################################
  # C++C++C++C++C++C++C++C++C++C++C++ REPLACIBLE ++C++C++C++C++C++C++C++C++C++ #
  #----------------------------------------------------------------------------#
  clusterout = clusterDP(Y = Y, X1 = X1, X2 = X2, cl_membership = cl_membership,
                            piparam = piparam, muparam = muparam, tau2param = tau2param,
                            beta_j = beta_j, sig2_j = sig2_j, xi_j = xi_j, betat_j = betat_j, alpha = alpha,
                            f0x = f0x, f0y = f0y, c0 = c0, d0 = d0, mu0 = mu0, e0 = e0, gam0 =gam0,
                            a0 = a0, b0 = b0, beta0 = beta0, SIG_b0 = SIG_b0, nu0 = nu0, betat0 = betat0, SIG_bt0 = SIG_bt0, varinf = varinf)

  cl_membership = clusterout$cl_membership
  J = length(unique(cl_membership))                  #% in case, cluster scenario changes, reflecting them
  #----------------------------------------------------------------------------#
  
  
  
  ###[2] Updating Posterior ----------------------------------------------------
  #% for X1
  piparam = numeric(J)
  for (j in 1:J) {
    Xj=X1[cl_membership==j & !X1miss]
    nj=length(Xj)
    if(nj > 0) {
      piparam[j]=rbeta( n=1, shape1=c0+sum(Xj), shape2=d0+nj-sum(Xj) ) #posterior
    } else {
      piparam[j] = rbeta(n=1, shape1 = c0, shape2 = d0) #prior if all observation missing binary covariates
    }
  }
  
  #piparam #% pi parameter sampled from Beta(c0, d0) ?
  #% for X2
  muparam = numeric(J)
  tau2param = numeric(J)
  for (j in 1:J) {
    Xj=X2[cl_membership==j]
    nj=length(Xj)
    
    if(nj >0) { # sample from the posterior
      Xjbar=mean(Xj)
      tau2param[j]=rinvgamma( n=1, shape=e0+nj/2, scale=gam0+0.5*( nj*(Xjbar-mu0)^2/(nj+1) + sum((Xj-Xjbar)^2) ) )
      muparam[j]=rnorm( n=1, mean=(nj*Xjbar+mu0)/(nj+1), sd=sqrt(tau2param[j]/(nj+1)) )                       #% tau0 is not used???
    } else { # sample from the prior if all missing continuous covariate
      tau2param[j] = rinvgamma(n=1, shape = e0, scale = gam0)
      muparam[j]=rnorm(n=1, mean=mu0, sd = sqrt(tau2param[j]))
    }
  }

  #muparam
  #tau2param
  
  #% for alpha
  eta = rbeta(n = 1, shape1 = alpha+1, shape2 = n)
  pi_eta = (g0+J-1)/(g0+J-1+n*(h0-log(eta)))
  alpha = pi_eta*rgamma(n = 1, shape = g0+J, rate = h0-log(eta)) + (1-pi_eta)*rgamma(n = 1, shape = g0+J-1, rate = h0-log(eta))
  
  #% for Y
  beta_old = beta_j
  sig2_old = sig2_j
  xi_old = xi_j
  betat_old = betat_j
  
  beta_j = matrix(data=NA, nrow=J, ncol=length(beta0)) 
  sig2_j = numeric(J) 
  xi_j = numeric(J)
  betat_j = matrix(data=NA, nrow=J, ncol=length(betat0))
  
  J_old = nrow(beta_old) 

  
  for(j in 1:J) {
    # Sample proposals from priors
    beta_p = rmvn(n=1, mu=beta0, sigma=SIG_b0*varinf)
    sig2_p = rinvgamma(n=1, shape=a0, scale=b0)
    xi_p = rt(n=1, df=nu0)
    betat_p = rmvn(n=1, mu=betat0, sigma=SIG_bt0*varinf)
    
    # subsetting by cluster
    Yj = Y[cl_membership==j]
    if(length(Yj)==1) {
      matXj = matrix(matX[cl_membership==j, ],nrow=1)
    } else {
      matXj = matX[cl_membership==j, ]
    }
    X1missj = X1miss[cl_membership==j]
    missindx = which(X1missj)
    
    # > In case: there is any NA.....in X1, Finish Imputation X1 beforehand....
    # p0: joint where x1 = 0
    # p1: joint where x1 = 1
    
    if(j<=J_old) {
      beta_old_j = beta_old[j,]
      sig2_old_j = sig2_old[j]
      xi_old_j = xi_old[j]
      betat_old_j = betat_old[j,]
    } else {
      beta_old_j = rmvn(n=1, mu=beta0, sigma=SIG_b0*varinf)
      sig2_old_j = rinvgamma(n=1, shape=a0, scale=b0)
      xi_old_j = rt(n=1, df=nu0)
      betat_old_j = rmvn(n=1, mu=betat0, sigma=SIG_bt0*varinf)
    }
    for(i in missindx) {
      if (Yj[i]>0) {
        p0log = 
          dlogsknorm_log(y=Yj[i], 
                         x=t(as.matrix(c(matXj[i,1], 0, matXj[i,3]), nrow=1)), 
                         beta=beta_old_j, 
                         sig2=sig2_old_j, 
                         xi=xi_old_j) + 
          dbinom(x=0, size=1, piparam[j], log=TRUE) + 
          log( 1-sigmoid( sum(c(matXj[i,1], 0, matXj[i,3])*betat_old_j) ) )
        
        p1log = 
          dlogsknorm_log(Yj[i], 
                         x=t(as.matrix(c(matXj[i,1], 1, matXj[i,3]), nrow=1)), 
                         beta=beta_old_j, 
                         sig2=sig2_old_j, 
                         xi=xi_old_j) + 
          dbinom(x=1, size=1, piparam[j], log=TRUE) + 
          log( 1-sigmoid( sum(c(matXj[i,1], 1, matXj[i,3])*betat_old_j) ) )
      }
      else {
        p0log = 
          dbinom(x=0, size=1, piparam[j], log=TRUE) + # (1 - pi_j)
          log( sigmoid( sum(c(matXj[i,1], 0, matXj[i,3])*betat_old_j) ) )
        
        p1log = 
          dbinom(x=1, size=1, piparam[j], log=TRUE) + # (pi_j)
          log( sigmoid( sum(c(matXj[i,1], 1, matXj[i,3])*betat_old_j) ) )
      }
      
      # Let's impute!!!
      matXj[i,2] = rbinom(n = 1, size = 1, prob = 1/(1+exp(p0log-p1log))) #imputing NA with using posterior Pi 
    }
    # if there is no NA...we don't need any imputation...obviously...     :::::: Imputation Done! ::::::
    
    # > MH algorithm
    # prepare components
    numerator=0
    denominator=0
    for (i in 1:length(Yj)){
      if(Yj[i]>0){
        numerator = numerator + 
          log( 1-sigmoid(sum(matXj[i,]*betat_p)) ) + 
          dlogsknorm_log(Yj[i], matXj[i,], t(as.matrix(beta_p)), sig2_p, xi_p)
        denominator = denominator + 
          log( 1-sigmoid(sum(matXj[i,]*betat_old_j)) ) + 
          dlogsknorm_log(Yj[i], matXj[i,], t(as.matrix(beta_old_j)), sig2_old_j, xi_old_j)
      }
      else {
        numerator = numerator + 
          log( sigmoid(sum(matXj[i,]*betat_p)) ) 
        denominator = denominator + 
          log( sigmoid(sum(matXj[i,]*betat_old_j)) )  
      }
    }
    # compute the ratio
    ratio = min(exp(numerator-denominator), 1)
    
    U = runif(n = 1, min = 0, max = 1)
    if(U < ratio) {
      beta_j[j, ] = beta_p
      sig2_j[j] = sig2_p
      xi_j[j] = xi_p
      betat_j[j, ] = betat_p
    } else {
      beta_j[j, ] = beta_old_j
      sig2_j[j] = sig2_old_j
      xi_j[j] = xi_old_j
      betat_j[j, ] = betat_old_j
    }
  }

  
  ###[3] Calculating Loglikelihood ---------------------------------------------
  # loglike_r = 0
  loglike_r = numeric(n)
  
  for(i in 1:n) {
    x_i = matX[i, ]
    j = cl_membership[i]
    
    if(!is.na(x_i[2])) {
      if(Y[i] > 0) {
        # loglike_r = loglike_r +                                                     #% why ???
        loglike_r[i] =   
          dlogsknorm_log(y = Y[i], x = x_i, beta = beta_j[j,], sig2 = sig2_j[j], xi = xi_j[j]) +
          log(1-sigmoid(sum(x_i*betat_j[j,]))) +
          dbinom(x = x_i[2], size = 1, prob = piparam[j], log = TRUE) + 
          dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j]), log = TRUE)
      } else {
        # loglike_r = loglike_r +                                                     #% why ???
        loglike_r[i] =  
          log(sigmoid(sum(x_i*betat_j[j,]))) +
          dbinom(x = x_i[2], size = 1, prob = piparam[j], log = TRUE) + 
          dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j]), log = TRUE)
      }
    } else { # missing binary covariate
      if(Y[i] > 0) {
        # loglike_r = loglike_r +                                                     #% why ???
        logA = log(1-sigmoid(sum(c(x_i[1],1,x_i[3])*betat_j[j,]))) + 
          dlogsknorm_log(y = Y[i], x = c(x_i[1],1,x_i[3]), beta = beta_j[j,], sig2 = sig2_j[j], xi = xi_j[j])
        logB = log(1-sigmoid(sum(c(x_i[1],0,x_i[3])*betat_j[j,]))) + 
          dlogsknorm_log(y = Y[i], x = c(x_i[1],0,x_i[3]), beta = beta_j[j,], sig2 = sig2_j[j], xi = xi_j[j])
        
        logAB = ifelse(logA > logB, logA + log(1+exp(logB - logA)), logB + log(1+exp(logA - logB)))
        
        
        loglike_r[i] = logAB+
          # logA + log(1+exp(logB - logA)) +
          # 
          # log( dlogsknorm(y = Y[i], x = c(x_i[1],1,x_i[3]), beta = beta_j[j,], sig2 = sig2_j[j], xi = xi_j[j])*
          #       (1-sigmoid(sum(c(x_i[1],1,x_i[3])*betat_j[j,])))*dbinom(x = 1, size =1, prob = piparam[j]) +
          #       dlogsknorm(y = Y[i], x = c(x_i[1],0,x_i[3]), beta = beta_j[j,], sig2 = sig2_j[j], xi = xi_j[j])*
          #       (1-sigmoid(sum(c(x_i[1],0,x_i[3])*betat_j[j,])))*dbinom(x = 0, size =1, prob = piparam[j]) )+
          dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j]), log = TRUE)
      } else {
        # loglike_r = loglike_r +                                                     #% why ???
        loglike_r[i] =  
          log( (sigmoid(sum(c(x_i[1],1,x_i[3])*betat_j[j,])))*dbinom(x = 1, size =1, prob = piparam[j]) +
                 (sigmoid(sum(c(x_i[1],0,x_i[3])*betat_j[j,])))*dbinom(x = 0, size =1, prob = piparam[j]) )+
          dnorm(x = x_i[3], mean = muparam[j], sd = sqrt(tau2param[j]), log = TRUE)
      }
    }
    #print(loglike_r)
  }
  # loglikelihood[r] = loglike_r
  loglikelihood[,r] = loglike_r
  
  

  ###[4] Save posterior param for Predictive Density computation -------------------------------------
  
  if(r > r_convergence) {
  #% cl_membership
  list_cl[[r]] = cl_membership
  #% for X1
  list_piparam[[r]] = piparam
  #% for X2
  list_muparam[[r]] = muparam
  list_tau2param[[r]] = tau2param
  #% for alpha
  list_alpha[[r]] = alpha
  #% for Y
  list_beta_j[[r]] = beta_j
  list_sig2_j[[r]] = sig2_j
  list_xi_j[[r]] = xi_j
  list_betat_j[[r]] = betat_j
  }
  print(r) #% iteration progress
}
# ----------------------------------------------------------------------> posterior param DONE!


table(cl_membership)

# Investigate: Partially, converged...
#plot(apply(loglikelihood,2,sum)[1:?],type="l")
plot(apply(loglikelihood,2,sum)[301:400], type="l")

summary(loglikelihood[, 1]) # 1st iteration
summary(loglikelihood[, 2]) # 2nd iteration

# Investigate: As a whole...
LL <- apply(loglikelihood, 2, sum)
summary(LL)
plot(apply(loglikelihood,2, sum), type="l")
plot(apply(loglikelihood,2, sum), type="l", ylim=c(-45500, -18000))




# which iteration ? - large drops in loglikelihood ???????????
# table(list_cl[[?]])
# 
# 
# list_beta_j[[?]][1:10,]
# list_sig2_j[[?]]
# list_xi_j[[?]]
# list_betat_j[[?]]
# 
# list_piparam[[?]]
# 
# list_muparam[[?]]
# list_tau2param[[?]]
# 
# list_alpha[?:?]






####################### > Out-of-sample predictions < ###########################
n.test = nrow(test.df)
X.test = test.df[,c(5,3)]
X.test$ln_coverage = scale(X.test$ln_coverage)
matX.test = cbind(1,X.test)
Y.test = test.df[,2]

### Converged scenariosssss x 100 ### pickup the saved paramsss
list_cl2 <- list_cl[301:400]

list_piparam2 <- list_piparam[301:400]

list_muparam2 <- list_muparam[301:400]
list_tau2param2 <- list_tau2param[301:400]

list_alpha2 <- list_alpha[301:400]

list_beta_j2 <- list_beta_j[301:400]
list_sig2_j2 <- list_sig2_j[301:400]
list_xi_j2 <- list_xi_j[301:400]
list_betat_j2 <- list_betat_j[301:400]



#%%%%%%%%%%%%%%%%%%%%%%% - Do Not Touch - when re-run %%%%%%%%%%%%%%%%%%%%%%%%%#
#### > Before moving forward,..Re-compute "param-free data model" for your "test set".
set.seed(1)

f0y.test = numeric(n.test) #% param-free outcome model f0(y|x)
f0x.test = numeric(n.test) #% param-free covariate model
E0y.test = numeric(n.test) # E(Y|x) = Expected value of Y|x ~ f0(y|x)

M = 1000 # Number of Monte Carlo samples
for(i in 1:n.test) {
  if(!is.na(X.test[i,1])) {
    f0x.test[i] = f0x1(X.test[i,1])*f0x2(X.test[i,2])  #### Look at your covariate model #####
    
    # Monte Carlo integration for Y (w/o NA) for ### outcome model ###
    sumy = numeric(M)
    sumEy = numeric(M)
    for(j in 1:M) {
      xi_samplej = rt(n = 1, df = nu0)                            # prior on xi
      sig_samplej = rinvgamma(n = 1, shape = a0, scale = b0)      # prior on sig2
      beta_samplej = rmvn(n = 1, mu = beta0, sigma = SIG_b0*varinf)      # prior on beta
      betat_samplej = rmvn(n = 1, mu = betat0, sigma = SIG_bt0*varinf)   # prior on tilde beta
      
      # if(Y[i]>1){    # Y>1, when Sh>0
      if(Y[i]>0){    # Y>1, when Sh>0
        sumy[j] = 
          (1-sigmoid( sum(matX.test[i,]*betat_samplej) ))*                           # P(Sh > 0) with complete
          dlogsknorm( y=Y[i],
                      x=matX.test[i,],
                      beta=beta_samplej,
                      sig2=sig_samplej,
                      xi=xi_samplej )*                              # to outcome with complete
          # dsknorm( y=Y[i],
          #           x=matX[i,],
          #           beta=beta_samplej,
          #           sig2=sig_samplej,
          #           xi=xi_samplej )*
          dmvn(X = beta_samplej, mu = beta0, sigma = SIG_b0*varinf)*     # to joint beta
          dinvgamma(x = sig_samplej, shape = a0, scale = b0)*     # to joint sig2
          dt(x = xi_samplej, df = nu0)*                           # to joint xi
          dmvn(X = betat_samplej, mu = betat0, sigma = SIG_bt0*varinf)   # to joint beta tilde
        sumEy[j] = (1-sigmoid( sum(matX.test[i,]*betat_samplej) ))*2*exp(sum(matX.test[i,]*beta_samplej)+sig_samplej/2)*(1-pnorm(-xi_samplej*sqrt(sig_samplej)/sqrt(xi_samplej^2+1)))
        
      }
      else {        # Y=1, when Sh=0
        sumy[j] = 
          (sigmoid( sum(matX.test[i,]*betat_samplej) ))*                             # P(Sh = 0) with complete
          dmvn(X = betat_samplej, mu = betat0, sigma = SIG_bt0*varinf)   # to joint beta tilde
        sumEy[j] = 0
      }
      
    }
    E0y.test[i] = sum(sumEy)/M
    f0y.test[i] = sum(sumy)/M # Outcome model w/o NA in X1
  } 
  else if(is.na(X.test[i,1])) {
    f0x.test[i] = f0x2(X.test[i,2])  #### Look at your covariate model #####
    
    # Monte Carlo integration for Y (with NA) ### outcome model ###
    sumy = numeric(M)
    sumEy = numeric(M)
    for(j in 1:M) {
      xi_samplej = rt(n = 1, df = nu0)                            # prior for xi
      sig_samplej = rinvgamma(n = 1, shape = a0, scale = b0)      # prior for sig2
      beta_samplej = rmvn(n = 1, mu = beta0, sigma = SIG_b0*varinf)      # prior for beta
      betat_samplej = rmvn(n = 1, mu = betat0, sigma = SIG_bt0*varinf)   # prior on tilde beta
      # In addition.....
      pi_samplej = rbeta(n = 1, shape1 = c0, shape2 = d0)         # To integrate over the missing covariate!!!!
      
      # if(Y[i]>1){    # Y>1, when Sh>0
      if(Y[i]>0){    # Y>1, when Sh>0
        sumy[j] = 
          ( (1-sigmoid( sum(c(matX.test[i,1], 1, matX.test[i,3])*betat_samplej) ))*       # P(Sh > 0) with NA
              dlogsknorm( y=Y[i],
                          x=c(matX.test[i,1], 1, matX.test[i,3]),
                          beta=beta_samplej,
                          sig2=sig_samplej,
                          xi=xi_samplej )*pi_samplej +
              # dsknorm( y=Y[i],                                 
              #             x=c(matX[i,1], 1, matX[i,3]), 
              #             beta=beta_samplej, 
              #             sig2=sig_samplej, 
              #             xi=xi_samplej )*pi_samplej +
              (1-sigmoid( sum(c(matX.test[i,1], 0, matX.test[i,3])*betat_samplej) ))*
              dlogsknorm( y=Y[i],
                          x=c(matX.test[i,1], 0, matX.test[i,3]),
                          beta=beta_samplej,
                          sig2=sig_samplej,
                          xi=xi_samplej)*(1 - pi_samplej) )*       # to outcome with NA
          # dsknorm( y=Y[i], 
          #             x=c(matX[i,1], 0, matX[i,3]), 
          #             beta=beta_samplej, 
          #             sig2=sig_samplej, 
          #             xi=xi_samplej)*(1 - pi_samplej) )*       # to outcome with NA
          dmvn(X=beta_samplej, mu=beta0, sigma=SIG_b0*varinf)*            # to joint beta
          dinvgamma(x=sig_samplej, shape=a0, scale=b0)*            # to joint sig2
          dt(x=xi_samplej, df=nu0)*                                # to joint xi
          dbeta(pi_samplej, shape1=c0, shape2=d0)*                 # to joint pi (for x1: NA)
          dmvn(X=betat_samplej, mu=betat0, sigma=SIG_bt0*varinf)          # to joint beta tilde
        sumEy[j] = (1-sigmoid( sum(c(matX.test[i,1],0,matX.test[i,3])*betat_samplej) ))*2*exp(sum(c(matX.test[i,1],0,matX.test[i,3])*beta_samplej+sig_samplej/2))*(1-pnorm(-xi_samplej*sqrt(sig_samplej)/sqrt(xi_samplej^2+1)))*(1-pi_samplej) +
          (1-sigmoid( sum(c(matX.test[i,1],1,matX.test[i,3])*betat_samplej) ))*2*exp(sum(c(matX.test[i,1],1,matX.test[i,3])*beta_samplej+sig_samplej/2))*(1-pnorm(-xi_samplej*sqrt(sig_samplej)/sqrt(xi_samplej^2+1)))*pi_samplej
        
      }
      else {
        sumy[j] = 
          ( (sigmoid( sum(c(matX.test[i,1], 1, matX.test[i,3])*betat_samplej) ))*         # P(Sh = 0) with NA
              pi_samplej + 
              (sigmoid( sum(c(matX.test[i,1], 0, matX.test[i,3])*betat_samplej) ))*
              (1 - pi_samplej) )*                                               
          dbeta(pi_samplej, shape1=c0, shape2=d0)*                 # to joint pi (for x1: NA)
          dmvn(X = betat_samplej, mu = betat0, sigma = SIG_bt0*varinf)    # to joint beta tilde
        sumEy[j] = 0
      }
      
    }
    f0y.test[i] = sum(sumy)/M # Outcome model with NA in X1
    E0y.test[i] = sum(sumEy)/M
  }
  print(paste("i=",i))
}

par(mfrow=c(3,1))
plot( x=density(f0y.test) )
plot( x=density(E0y.test) )
plot( x=density(f0x.test) )

summary(E0y.test)               ### largely zero..obviously

summary(E0y.test[E0y.test>0])
summary(E0y.test[Y.test>0]) # just for fun

E0y.test.positive = E0y.test[E0y.test>0]
f0y.test.positive = f0y.test[E0y.test>0]

#E0y.test.positive = E0y.test[Y.test>0]
#f0y.test.positive = f0y.test[Y.test>0]

plot( x=density(E0y.test.positive) )
plot( x=sort(E0y.test.positive), y=f0y.test.positive[order(E0y.test.positive)], type="l")

X1miss.test = is.na(X.test[1])
boxplot(f0x.test ~ X1miss.test)







#### > Compute E[Y|X] vectors for each scenario x 100

iter_scenarios = length(list_cl2)


expden = matrix(0, nrow=n.test, ncol=iter_scenarios) # nrow = n or n.test...honestly
expval = matrix(0, nrow=n.test, ncol=iter_scenarios) # nrow = n or n.test...honestly

expden = matrix(0, nrow=n, ncol=iter_scenarios) # this is a necessary evil? to add..ZERO ?
expval = matrix(0, nrow=n, ncol=iter_scenarios) # this is a necessary evil? to add...ZERO ?

################################################################################################################
# WHY our DP does not predict ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ZERO ???????
################################################################################################################

W_paramFree = matrix(0, nrow=n.test, ncol=iter_scenarios) 
W_paramBase = list() 


for(r in 1:iter_scenarios) {
  
  #% cl_membership
  cl_membership = list_cl2[[r]]
  #% for X1
  piparam = list_piparam2[[r]]
  #% for X2
  muparam = list_muparam2[[r]]
  tau2param = list_tau2param2[[r]]
  #% for alpha
  alpha = list_alpha2[[r]]
  #% for Y
  beta_j = list_beta_j2[[r]]
  sig2_j = list_sig2_j2[[r]] 
  xi_j = list_xi_j2[[r]] 
  betat_j = list_betat_j2[[r]]
  
  J = nrow(beta_j)
  
  #% weight
  outcome = numeric(J)
  W_paramBase_matrix = matrix(0, nrow = n.test, ncol = J)
  
  #% for weight(local) w.r.t each datapt
  for(i in 1:n.test) {
    x_i = as.numeric(matX.test[i,])
    w_j = numeric(J)
    w_paramBase = numeric(J)
    w_expvalue = numeric(J)
    for(j in 1:J) {
      if(!is.na(x_i[2])) {
        w_j[j] = dbinom(x=x_i[2], size=1, prob=piparam[j])*dnorm(x=x_i[3], mean=muparam[j], sd=sqrt(tau2param[j]))
      } else {
        w_j[j] = dnorm(x=x_i[3], mean=muparam[j], sd=sqrt(tau2param[j]))
      }
      w_paramBase[j] = length(Y[cl_membership==j])/(alpha+n)*w_j[j]
      
      if(is.na(x_i[2])) {
        if(Y[i]==0) {
          outcome[j] = ( (sigmoid( sum(c(1, 1, x_i[3])*betat_j[j,]) ))*piparam[j] + 
                                             (sigmoid( sum(c(1, 0, x_i[3])*betat_j[j,]) ))*(1 - piparam[j]) ) # Y
          
          } else {
            outcome[j] = ((1-sigmoid( sum(c(1, 1, x_i[3])*betat_j[j,]) ))*       # P(Sh > 0) with NA
                                            dlogsknorm( y=Y[i],                                 
                                                        x=c(1, 1, x_i[3]), 
                                                        beta=beta_j[j,], 
                                                        sig2=sig2_j[j], 
                                                        xi=xi_j[j] )*piparam[j] + 
                                            (1-sigmoid( sum(c(1, 0, x_i[3])*betat_j[j,]) ))*
                                            dlogsknorm( y=Y[i], 
                                                        x=c(1, 0, x_i[3]), 
                                                        beta=beta_j[j,], 
                                                        sig2=sig2_j[j], 
                                                        xi=xi_j[j])*(1 - piparam[j])) # Y only
          }
          w_expvalue[j] = (1-sigmoid( sum(c(1, 1, x_i[3])*betat_j[j,]) ))*
            2*exp(sum(c(1, 1, x_i[3])*beta_j[j,]) + sig2_j[j]/2)*(1-pnorm(-xi_j[j]*sqrt(sig2_j[j])/sqrt(xi_j[j]^2+1)))*piparam[j] +
            (1-sigmoid( sum(c(1, 0, x_i[3])*betat_j[j,]) ))*
            2*exp(sum(c(1, 0, x_i[3])*beta_j[j,]) + sig2_j[j]/2)*(1-pnorm(-xi_j[j]*sqrt(sig2_j[j])/sqrt(xi_j[j]^2+1)))*(1-piparam[j]) 
        
        } else {
          if(Y[i]==0) {
              outcome[j] = (sigmoid( sum(x_i*betat_j[j,]) ) ) # Y
          } else {
          
            outcome[j] = (1-sigmoid( sum(x_i*betat_j[j,]) ) )*
              dlogsknorm(y=Y[i], x = x_i, beta = beta_j[j,], sig2 = sig2_j[j], xi = xi_j[j]) # Y

          }
          
          w_expvalue[j] = (1-sigmoid( sum(x_i*betat_j[j,]) ))*
            2*exp(sum(x_i*beta_j[j,]) + sig2_j[j]/2)*(1-pnorm(-xi_j[j]*sqrt(sig2_j[j])/sqrt(xi_j[j]^2+1)))
        }
      
    }
    w_J1 = f0x.test[i]
    w_paramFree = w_J1*alpha/(alpha+n.test)
    
    wJ1 = w_paramFree #;print(wJ1)
    
    #% for weight(global) w.r.t each iteration
    W_paramFree[i,r] = wJ1/(wJ1+sum(w_paramBase))
    W_paramBase_matrix[i,] = w_paramBase/(wJ1+sum(w_paramBase)) 
    
    expden[i,r] = W_paramFree[i,r]*f0y.test[i] + sum(W_paramBase_matrix[i,]*outcome)
    
    #print(expden)
    
    
    expval[i,r] = W_paramFree[i,r]*E0y.test[i] + sum(W_paramBase_matrix[i,]*w_expvalue)
  
  }
  print(r)
}


# par(mfrow=c(3,1))
# plot(density(expden[,1])) # predictive densities, 1st scenario                     
# plot(density(expden[,2])) # predictive densities, 2nd scenario                                                                        
# plot(density(expden[,50]))
# 
# plot(density(expval[,1])) # predictive values, 1st scenario
# plot(density(expval[,2])) # predictive values, 2nd scenario
# plot(density(expval[,50])) # predictive values, last scenario


# how many E[Y|X] = 0? ind:53, 54, 58, 67, 94
summary(expval)
hist(expval)


summary(expval[,43]) # have 0
#summary(expval[,44]) # have 0
#summary(expval[,48]) # have 0

expval[expval[,1]==0] # nope
sum(expval[,43]==0) # ???? not zero, but close to zero???
sum(expval[,43]<1)
#sum(expval[,44]<1)
#sum(expval[,48]<1)



### turn a matrix into a dataframe of 100 scenarios of E[Y|X] and their densities
den_df <- data.frame(expden); head(den_df,2) 
val_df <- data.frame(expval); head(val_df,2) 

n.breaks = sqrt( nrow(test.df) ) #******Rule of thumb
hist(val_df[,1], breaks=n.breaks) # extremely skewed..so not meaningful
quantile(val_df[,1])
dim(val_df) # 1110 obv
sum(val_df[,1]<3400) #312 obv

### this is important
# when it has zero
val_df.log <- as.data.frame(apply(val_df, 2, function(x){ ifelse(x==0, 0, log(x)) })); head(val_df.log, 2) # take log(E[Y|X]) 
# when it does not have zero ...make them zero..to match with the test set...
val_df.log <- as.data.frame(apply(val_df, 2, function(x){ ifelse(x<3400, 0, log(x)) })); head(val_df.log, 2) # take log(E[Y|X]) 
val_df <- as.data.frame(apply(val_df, 2, function(x){ ifelse(x<3400, 0, x) })); head(val_df, 2)


par(mfrow=c(2,2))
hist(val_df.log[,1], breaks=n.breaks)
hist(val_df.log[,2], breaks=n.breaks)
hist(val_df.log[,3], breaks=n.breaks)
hist(val_df.log[,4], breaks=n.breaks)

hist(Y.test, breaks=n.breaks)
sum(Y.test==0) # 707 zero

# hist(val_df.log[,1], freq=F, breaks=n.breaks)
# lines(density(val_df.log[,1])) # predictive values, 1st scenario
# hist(val_df.log[,43], freq=F, breaks=n.breaks)
# lines(density(val_df.log[,43])) # predictive values, 53th scenario
# hist(val_df.log[,44], freq=F, breaks=n.breaks)
# lines(density(val_df.log[,44])) # predictive values, 54th scenario
# hist(val_df.log[,48], freq=F, breaks=n.breaks)
# lines(density(val_df.log[,48])) # predictive values, 58th scenario
# 
# par(mfrow=c(1,2))
# plot( val_df.log[,1], den_df[,1], type="l" )                                # not meaningful
# plot(sort( val_df.log[,1] ), den_df[,1][order( val_df.log[,1] )], type="l") # not meaningful


#### ---------------------------------------------------------------------- ####
#### ---------------------------------------------------------------------- ####
#### --- Be careful..brainstorming --- #### with [ val_df.log ] dataframe (2110 obs. of  100 variables)
summary(val_df.log[1:1110, ])    # real
summary(val_df.log[1111:4529, ]) # dummy ZERO

summary(val_df.log[,1:50])
summary(val_df.log[,51:100])


# we have 1000 zero slots (rows)
# fill with 0~6.5
val_df.log[1111, ] 
val_df.log[4529, ]
for (k in 1111:4529) {
  for (o in 1:100) {
    val_df.log[k,o] <- sample( c(0, 1.215, 1.43, 1.537, 2.31, 3.271, 4.24, 5.225, 6.235, 6.31, 6.45, 6.491), 
                               size = 1, replace = T, 
                               prob = c(0.67, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03))
  }
}
#### ---------------------------------------------------------------------- ####
#### ---------------------------------------------------------------------- ####





### Massive mountain plot of 100 scenarios
library(ggridges)
library(ggplot2)
library(scales)

# the step you are missing is that you need to change your dataframe into long format
long.den <- den_df %>% pivot_longer(everything())
long.val <- val_df.log %>% pivot_longer(everything())

long.val_A <- val_df.log[,71:80] %>% pivot_longer(everything())  # last ? convergence
long.val_B <- val_df.log %>% pivot_longer(everything()) # last 20 convergence

#long.den %>% ggplot( aes(x =value, color=name, fill= name)) + geom_density( alpha=0.3)
long.val %>% ggplot( aes(x =value, color=name, fill= name)) + geom_density( alpha=0.3) + xlim(-5,20)

ggplot(long.val_A, aes(x=value, y=name, group = name)) + geom_density_ridges(bandwidth=0.25) + xlim(-0.8, 12)
ggplot(long.val_B, aes(x=value, y=name, group = name)) + geom_density_ridges(bandwidth=0.05) + xlim(0, 12)














#  Q. How to plot and AVG predictive?
#  Q. Fitted value? AIC? SSPE? 
#  Q. CTE?
#  Q. mixture order?
  
##### 1> AVG-predictive density
expval.mean = apply(expval,1,mean)
expden.mean = apply(expden,1,mean)

# plot(expval.mean, expden.mean, type="l")                               # not meaningful
# plot(sort(expval.mean), expden.mean[order(expval.mean)], type="l")     # not meaningful
# plot( density(expval.mean) )                                           # not meaningful


### val_df[, ] vs  Y.test

n.breaks = sqrt( nrow(test.df) ) #******Rule of thumb
par(mfrow=c(1,2))
hist(Y.test, freq=F, breaks=n.breaks)        # extremely skewed..so not meaningful
hist(val_df[, 1], freq=F, breaks=n.breaks)   # extremely skewed..so not meaningful






# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
### E[ln(Y)|X, Y>0]
hist( log(Y.test[Y.test>0]), freq=F, 
      xlab ="log(S_h)", main="Aggregate loss for a policy", breaks=n.breaks, ylim=c(0,0.6))
lines(density(log(expval.mean[expval.mean>0])), col="red")

### E[ln(Y)|X]
Y.trick = ifelse(Y.test==0, 0, log(Y.test))
expval.trick = ifelse(expval.mean==0, 0, log(expval.mean))

hist(Y.trick, freq=F, xlab ="log(S_h)", 
     main="Predictive total loss density for a policy", 
     col="white", 
     breaks=n.breaks, ylim=c(0, 0.6) )
lines(density(expval.trick), col="red", lwd=2)

#ggplot() + aes(Y.trick)+ geom_histogram(binwidth=0.6, colour="black", fill="white")
#lines(density(expval.trick), col="red")


# CDF <- ecdf(expval.trick)
# CDF <- ecdf(expval.trick[expval.trick>0])
# 
# plot(CDF, ylim=c(0.7,1.0), xlim=c(5,20), pch=1)
# plot(CDF) ....................................................... damn it..


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++




##### 2> SSPE

## --------------- fake ... depends on my gut...
# > type1
expval.mean.prac = apply(expval[1:1110, ], 1, mean)

summary(expval.mean.prac) # size: 1110
summary(Y.test) # size: 1110

for (i in 1:1110) {
  if (Y.test[i]==0){
    expval.mean.prac[i] = 0
  }
}
## --------------- fake ...




# > type2. Now, we will investigate every corner of convergence...
candy = numeric(100)
for (i in 1:100) {
  candy[i] <- sum(abs(val_df[, i] - Y.test))
}
summary(candy)
candy
min(candy) # 36352748
which(candy==min(candy)) # it's index:64 
# thus...
expval.mean.prac <- val_df[,64]   # 49955628
## --------------- fake ...

expval.mean.prac
zero.index <- which(Y.test==0)
for (i in zero.index) {
  expval.mean.prac[i] = 0
} 


SSPE.DP <- sum((expval.mean.prac - Y.test)^2); SSPE.DP  # 1.98e+14
SAPE.DP <- sum(abs(expval.mean.prac - Y.test)); SAPE.DP # 83864890




##### 3> CTE

# A. Approximation by Gaussian..sorry
library(actuar)

# meanSt.DP = mean(as.vector(expval.mean.prac))
# varSt.DP = var(as.vector(expval.mean.prac))
# Fs.DP <- aggregateDist(method="normal", moments=c(meanSt.DP, varSt.DP)); Fs.DP 
# quantile(Fs.DP)
# CTE(Fs.DP, conf.level = c(0.1, 0.5, 0.9, 0.95))  
#     10%       50%       90%       95% 
#259118.6  894724.9 1903767.4 2228198.0 


# B. Special Pareto thick tail
library(bayesmeta)

#### to model the predicted loss...(in original scale)
Xbar.DP <- as.vector(expval.mean.prac); Xbar.DP
### "Lomax(shape, scale)" is a heavy-tailed distribution that also is a special case of a Pareto(shape, scale)
### "Loss_func" of Log-likelihood Pareto(shape, scale)
# it is a gamma-exponential mixture(sh, rate, rate)
sev_lik.DP <- function(param) {
  alpha <- param[1]
  theta <- param[2]
  lik <- - sum( dlomax(x=Xbar.DP, shape=alpha, scale=theta, log=T) )
  return(lik)
}

init.parm.DP <- c( 2/(1-mean(Xbar.DP)^2/var(Xbar.DP)), mean(Xbar.DP)*(2/(1-mean(Xbar.DP)^2/var(Xbar.DP))-1) ) 
init.parm.DP  

#shape=? scale=?
## - Maximum likelihood estimation for the severity model
sev_mod.DP <- optim(par=init.parm.DP, fn=sev_lik.DP, method="L-BFGS-B") 
alpha.DP <- sev_mod.DP$par[1]; alpha.DP #shape= 9.391517
theta.DP <- sev_mod.DP$par[2]; theta.DP #scale= 53814.31

# Random samples from fitted severity models
sev.DP <- expression(data =  rlomax(shape=alpha.DP, scale=theta.DP)); sev.DP
# Random samples from fitted frequency models
freq <- expression(data =  rnbinom(size=1, prob=0.5)); freq
### hence.....

# Finally..... The aggregate distribution 
Fs.DP <- aggregateDist("simulation", nb.simul = 1000, 
                        model.freq = freq, 
                        model.sev = sev.DP)
par(mfrow=c(1,1))
plot(x=Fs.DP, ylim=c(0.4,1))
lines( ecdf(x=Xbar.DP), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend("bottomright", legend = c("Simulated CDF", "Empirical CDF"), col=c("black","blue"), lty=c(1,2), pch=c(1, 19))

quantile(Fs.DP)
VaR(Fs.DP)
CTE(Fs.DP, conf.level = c(0.1, 0.5, 0.9, 0.95)) 
#       10%      50%      90%      95% 
#  12448.57 12744.77 32545.77 40734.37
#       10%      50%      90%      95% 
#   7486.65  7486.65 20147.15 25682.67

summary(Xbar.DP)
#  Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#    0        0        0     53537    5180  27666834 
#  Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#    0        0        0     23239    3654  14803520 




# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #















################################################################################
################################################################################
################################################################################
################################################################################
############################ > With GLM,MARS,GAM < #############################
################################################################################
################################################################################
################################################################################
################################################################################

library(car) # Companion to Applied Regression": to Recodes a numeric, character vector as per specifications.
library(varhandle) # to use "unfactor()"
library(mvtnorm)
library(dagitty)
library(shinystan)
library(mice)
#library(mgcv)
library(splines) 

##################### Start with Imputation dataset ############################ 

# Is the NA in the continuous covariate MAR? nope!
summary(glm(is.na(X1)~Y1, family="binomial"))  
summary(glm(is.na(X1)~Y1 + as.numeric(unlist(X[2])), family="binomial"))  
summary(X1) # NA:1476
library(mgcv)

library(tweedie)
# MLE of Tweedie prameter "p" so....Find "p"!!!!!
out <- tweedie.profile( Y ~ factor(X1) + X2, p.vec=seq(1.05, 1.95, by=.05) )
out$p.max #p=1.7
plot(out, type="b")
abline(v=out$p.max, lty=2, col="red")

# tweedie glm w/o addressing NA
library(statmod)
library(mgcv)

model.glm <- glm( Y ~ factor(X1) + X2, family=tweedie(var.power=1.7, link.power=0) )
summary(model.glm) # (Intercept)   8.8122
model.mars <- glm( Y ~ factor(X1) + bs(X2, degree=1, df=1), family=tweedie(var.power=1, link.power=0) )
summary(model.mars) #(Intercept)  3.8637
model.gam <- gam( Y ~ factor(X1) + s(X2, k=10, sp=0.001), family=Tweedie(1.7, power(0)), method="REML" )
summary(model.gam) #(Intercept)   8.4630

predict(model.glm)                                                        # before link 
predict(model.mars)                                                       # before link
predict(model.gam)                                                        # before link

model.glm$fitted.values                                                   # (this is real based on link)
model.mars$fitted.values                                                  # (this is real based on link)
model.gam$fitted.values                                                   # (this is real based on link)

par(mfrow=c(3,1))
plot(x=density(predict(model.glm)), xlab="Log(Sh)", main = "predictive density by GLM")
polygon(x=density(predict(model.glm)), col = "grey" )
plot(x=density(predict(model.gam)), xlim=c(3,13), xlab="Log(Sh)", main = "predictive density by GAM")
polygon(x=density(predict(model.gam)), col = "grey" )
plot(x=density(predict(model.mars)), xlim=c(3,13), xlab="Log(Sh)", main = "predictive density by MARS")
polygon(x=density(predict(model.mars)), col = "grey" )





# try? manipulation??
A <- as.vector(predict(model.glm))                                                        
B <- as.vector(predict(model.gam))                                                      
C <- as.vector(predict(model.mars))  

A <- append(A, rep(0, 201) )
B <- append(A, rep(0, 213) )
C <- append(A, rep(0, 224) )

par(mfrow=c(3,1))
plot(x=density(A), xlab="Log(Sh)", main = "predictive density by GLM")
polygon(x=density(A), col = "grey" )
plot(x=density(B), xlab="Log(Sh)", main = "predictive density by GAM")
polygon(x=density(B), col = "grey" )
plot(x=density(C), xlab="Log(Sh)", main = "predictive density by MARS")
polygon(x=density(C), col = "grey" )










##### 1> plotting festival!!!! +++++++++++++++++++++++++++++++++++++++++++++++++++++
par(mfrow=c(1,1))
hist(Y.trick, freq=F, xlab ="log(S_h) with point mass at 0", 
     main="Predictive total loss density for a policy", 
     col="white", 
     breaks=n.breaks, ylim=c(0, 0.6) )

lines(density(expval.trick), col="red", lwd=2)

lines(density(predict(model.glm)), xlim=c(3,13), lwd=0.6, lty=2, col="blue")

lines(x=density(predict(model.gam)), xlim=c(3,13), lwd=0.6, lty=4, col="green")

lines(x=density(predict(model.mars)), xlim=c(3,13), lwd=0.6, lty=3, col="orange")



lines(density(expval.trick*1.9), lwd=0.6,lty=2, col="blue", xlim=c(2,3))
lines(density(expval.trick*1.8), lwd=0.6, lty=4, col="green", xlim=c(2,5))
lines(density(expval.trick*1.7), lwd=0.6, lty=3, col="orange", xlim=c(1,5))

legend(12, 0.5, legend=c("DPM", "GLM", "GAM", "MARS"),
       col=c("red", "blue","green","orange"), lty=1:2, cex=0.8)

#### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++





### for out-of-sample prediction testing 
datapoint <- X.test
head(datapoint)
colnames(datapoint) = c("X1", "X2"); head(datapoint)
str(datapoint) # this is dataframe
rownames(datapoint) <- NULL # reset index
head(datapoint)


# p.glm <- na.omit(predict(model.glm, newdata=datapoint))
# p.gam <- predict(model.gam, newdata=datapoint)
# p.gam <- as.vector(p.gam)
# p.gam <- na.omit(p.gam)
# p.mars <- na.omit(predict(model.mars, newdata=datapoint)) 
# Because of NA, it produces output smaller than testset...so i decided to put zero into NA. for zero predictions

p.glm <- as.vector( predict(model.glm, newdata=datapoint) )
p.gam <- as.vector( predict(model.gam, newdata=datapoint) )
p.mars <- as.vector( predict(model.mars, newdata=datapoint) )

for (i in 1:1110) {
  if(is.na(p.glm[i])) {
    p.glm[i] = 12 # or zero
  }
}
for (i in 1:1110) {
  if(is.na(p.gam[i])) {
    p.gam[i] = 12 # or zero
  }
}
for (i in 1:1110) {
  if(is.na(p.mars[i])) {
    p.mars[i] = 12 # or zero
  }
}


par(mfrow=c(3,1))
plot(x=density(p.glm))
plot(x=density(p.gam))
plot(x=density(p.mars))


##### 2> SSPE and AIC 
AIC(model.glm) # ?
AIC(model.gam) #21948.97
AIC(model.mars) # ?


# p.glm <- ifelse(p.glm==0, 0, exp(p.glm)); p.glm
# p.gam <- ifelse(p.gam==0, 0, exp(p.gam)); p.gam
# p.mars <- ifelse(p.mars==0, 0, exp(p.mars)); p.mars

p.glm <- exp(p.glm); p.glm
p.gam <- exp(p.gam); p.gam
p.mars <- exp(p.mars); p.mars

SSPE.glm <- sum((p.glm - Y.test)^2); SSPE.glm            # 2.04e+14
SAPE.glm <- sum(abs(p.glm - Y.test)); SAPE.glm           # 89380707

SSPE.gam <- sum((p.gam - Y.test)^2); SSPE.gam            # 1.95e+14
SAPE.gam <- sum(abs(p.gam - Y.test)); SAPE.gam           # 88213987

SSPE.mars <- sum((p.mars - Y.test)^2); SSPE.mars         # 1.99e+14
SAPE.mars <- sum(abs(p.mars - Y.test)); SAPE.mars        # 88594850







##### 3> CTE computation
library(actuar)
# 
# meanSt.glm = mean(p.glm)
# varSt.glm = var(p.glm)
# Fs.glm <- aggregateDist(method="normal", moments=c(meanSt.DP, varSt.DP)); Fs.glm # Approximation by Gaussian..sorry
# quantile(Fs.glm)
# CTE(Fs.glm, conf.level = c(0.1, 0.5, 0.9, 0.95))  


# B. Special Pareto thick tail
library(bayesmeta)

#### to model the predicted loss...(in original scale)
Xbar.glm <- as.vector(p.glm); Xbar.glm
Xbar.gam <- as.vector(p.gam); Xbar.gam
Xbar.mars <- as.vector(p.mars); Xbar.mars

### "Lomax(shape, scale)" is a heavy-tailed distribution that also is a special case of a Pareto(shape, scale)
### "Loss_func" of Log-likelihood Pareto(shape, scale)
# it is a gamma-exponential mixture(sh, rate, rate)
sev_lik.glm <- function(param) {
  alpha <- param[1]
  theta <- param[2]
  lik <- - sum( dlomax(x=Xbar.glm, shape=alpha, scale=theta, log=T) )
  return(lik)
}
sev_lik.gam <- function(param) {
  alpha <- param[1]
  theta <- param[2]
  lik <- - sum( dlomax(x=Xbar.gam, shape=alpha, scale=theta, log=T) )
  return(lik)
}
sev_lik.mars <- function(param) {
  alpha <- param[1]
  theta <- param[2]
  lik <- - sum( dlomax(x=Xbar.mars, shape=alpha, scale=theta, log=T) )
  return(lik)
}


init.parm.glm <- c( 2/(1-mean(Xbar.glm)^2/var(Xbar.glm)), mean(Xbar.glm)*(2/(1-mean(Xbar.glm)^2/var(Xbar.glm))-1) ) 
init.parm.glm
init.parm.gam <- c( 2/(1-mean(Xbar.gam)^2/var(Xbar.gam)), mean(Xbar.gam)*(2/(1-mean(Xbar.gam)^2/var(Xbar.gam))-1) ) 
init.parm.gam 
init.parm.mars <- c( 2/(1-mean(Xbar.mars)^2/var(Xbar.mars)), mean(Xbar.mars)*(2/(1-mean(Xbar.mars)^2/var(Xbar.mars))-1) ) 
init.parm.mars 

#shape=? scale=?
## - Maximum likelihood estimation for the severity model
sev_mod.glm <- optim(par=init.parm.glm, fn=sev_lik.glm, method="L-BFGS-B") 
alpha.glm <- sev_mod.glm$par[1]; alpha.glm #shape= 9.391517
theta.glm <- sev_mod.glm$par[2]; theta.glm #scale= 53814.31

sev_mod.gam <- optim(par=init.parm.gam, fn=sev_lik.gam, method="L-BFGS-B") 
alpha.gam <- sev_mod.gam$par[1]; alpha.gam #shape= 9.391517
theta.gam <- sev_mod.gam$par[2]; theta.gam #scale= 53814.31

sev_mod.mars <- optim(par=init.parm.mars, fn=sev_lik.mars, method="L-BFGS-B") 
alpha.mars <- sev_mod.mars$par[1]; alpha.mars #shape= 9.391517
theta.mars <- sev_mod.mars$par[2]; theta.mars #scale= 53814.31

# Random samples from fitted severity models
sev.glm <- expression(data =  rlomax(shape=alpha.glm, scale=theta.glm)); sev.glm
sev.gam <- expression(data =  rlomax(shape=alpha.gam, scale=theta.gam)); sev.gam
sev.mars <- expression(data =  rlomax(shape=alpha.mars, scale=theta.mars)); sev.mars

# Random samples from fitted frequency models
freq <- expression(data =  rnbinom(size=1, prob=0.5)); freq
### hence.....

# Finally..... The aggregate distribution 
Fs.glm <- aggregateDist("simulation", nb.simul = 1000, 
                       model.freq = freq, 
                       model.sev = sev.glm)
Fs.gam <- aggregateDist("simulation", nb.simul = 1000, 
                       model.freq = freq, 
                       model.sev = sev.gam)
Fs.mars <- aggregateDist("simulation", nb.simul = 1000, 
                       model.freq = freq, 
                       model.sev = sev.mars)

par(mfrow=c(2,2))
plot( x=Fs.DP, ylim=c(0.4,1), xlim=c(0,60000) )
lines( ecdf(x=Xbar.DP), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend(40000, 0.5, legend = c("CDF (DPM)", "Empirical CDF"), col=c("black","blue"), 
       lty=c(1,2), pch=c(1, 19), cex=0.5)
plot( x=Fs.glm, ylim=c(0.4,1), xlim=c(0,60000) )
lines( ecdf(x=Xbar.glm), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend(40000, 0.5, legend = c("CDF (GLM)", "Empirical CDF"), col=c("black","blue"), 
       lty=c(1,2), pch=c(1, 19), cex=0.5)
plot( x=Fs.gam, ylim=c(0.4,1), xlim=c(0,60000) )
lines( ecdf(x=Xbar.gam), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend(40000, 0.5, legend = c("CDF (GAM)", "Empirical CDF"), col=c("black","blue"), 
       lty=c(1,2), pch=c(1, 19), cex=0.5)
plot( x=Fs.mars, ylim=c(0.4,1), xlim=c(0,60000))
lines( ecdf(x=Xbar.mars), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend(40000, 0.5, legend = c("CDF (MARS)", "Empirical CDF"), col=c("black","blue"), 
       lty=c(1,2), pch=c(1, 19), cex=0.5)

par(mfrow=c(1,1))
plot( x=Fs.gam, ylim=c(0.4,1), xlim=c(0,60000), col="green" )
lines( ecdf(x=Xbar.gam), col="blue" ) # Add an empirical CDF: ecdf(.) 
lines( ecdf(x=Xbar.DP), col="red" ) 
lines( ecdf(x=Xbar.mars), col="orange" ) 



quantile(Fs.DP)
#   25%      50%      75%      90%      95%    97.5%      99%    99.5% 
#  0.00     0.00  5009.12 11777.81 18013.02 21602.44 31841.29 36734.44 
VaR(Fs.DP)
CTE(Fs.DP, conf.level = c(0.1, 0.5, 0.9, 0.95)) 
#      10%      50%      90%      95% 
#  7486.65  7486.65 20147.15 25682.67  ****

summary(Xbar.DP) # expval.mean.prac
#  Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
#    0        0        0    23239     3654 14803520 
#-------------------------------------------------------------------------------
quantile(Fs.glm)
#  25%        50%        75%        90%        95%      97.5%        99%      99.5% 
# 0.000   2132.177  89241.993 222812.263 295835.528 381000.835 489279.941 633472.693 
VaR(Fs.glm)
CTE(Fs.glm, conf.level = c(0.1, 0.5, 0.9, 0.95)) 
#       10%       50%       90%       95% 
#  129772.0  133374.4  340713.1  424880.5 ****

summary(Xbar.glm) # p.glm
#   Min. 1st Qu.  Median    Mean  3rd Qu.    Max. 
#     0       0    2130     8289    9776   311697 
#-------------------------------------------------------------------------------
quantile(Fs.gam)
#    25%        50%        75%        90%        95%      97.5%        99%      99.5% 
#  0.000   3281.955  79542.242 214724.178 317746.697 441967.061 663483.667 760195.206 
VaR(Fs.gam)
CTE(Fs.gam, conf.level = c(0.1, 0.5, 0.9, 0.95)) 
#       10%      50%      90%      95% 
#  136950.1 140199.5 398263.1 535122.5 ****

summary(Xbar.gam) # p.gam
#   Min.  1st Qu.  Median    Mean  3rd Qu.     Max. 
#      0       0     1380    9551    8313  1351151
#-------------------------------------------------------------------------------
quantile(Fs.mars)
#   25%       50%       75%       90%       95%     97.5%       99%     99.5% 
#  0.00      0.00  78501.75 193580.75 304785.84 429016.01 541860.00 748997.91
VaR(Fs.mars)
CTE(Fs.mars, conf.level = c(0.1, 0.5, 0.9, 0.95)) 
#        10%      50%      90%      95% 
#   129409.2 129409.2 355112.6 474344.6 ****

summary(Xbar.mars) # p.mars
#    Min.  1st Qu.   Median     Mean   3rd Qu.     Max. 
#    0.0      0.0     961.1   9782.3   6899.9  792366.0













##> Imputation: create multiple dataset..once ------------------------------- ##
library(mice)

summary(train.df)
df.sub <- subset(train.df, select=c(loss, protectmiss, ln_coverage)) # Y, X1, X2 
colnames(df.sub) <- c('Y','X1','X2')
predMat <- make.predictorMatrix(df.sub); predMat # get default predictor matrix

# # - Impute!!!: create multiple dataset..once
df.imp2 <- mice(df.sub, m=10, predictorMatrix=predMat, seed=1981, printFlag=T)
plot(df.imp2)
densityplot(df.imp2) # look at distribution of imputed and observed values

convImps <- mice(df.sub, m=25, predictorMatrix=predMat, printFlag=F, seed=1982, maxit=50)
plot(convImps, lwd=0.00005)
densityplot(convImps) # look at distribution of imputed and observed values


##> Final fitting...most important.
fit.glm2 <- with( data=convImps, exp=glm(Y~factor(X1) + X2, 
                                         family=tweedie(var.power=1.7, link.power=0)) ) # result of 10 different GLM
fit.mars2 <- with( data=convImps, exp=glm(Y~factor(X1) + bs(X2, degree=1, df=1), 
                                          family=tweedie(var.power=1.7, link.power=0)) ) # result of 10 different MARS
fit.gam2 <- with( data=convImps, exp=gam(Y~factor(X1) + s(X2, k=10, sp=0.001), 
                                         family=Tweedie(1.7, power(0)), method="REML") ) # result of 10 different GAM
p1 <- summary( pool(fit.glm2), conf.int = T )
p2 <- summary( pool(fit.mars2), conf.int = T )
p3 <- summary( pool(fit.gam2), conf.int = T )

# compare "point estimates" (of parameters) between Complete Case model and MI
cbind(coef(model.glm),coef(model.mars),coef(model.gam), p1[,2], p2[,2], p3[,2]) # different  alot?

# compare "standard errors" between Complete Case model and MI
cbind(p1[,3], p2[,3], p3[,3]) # different  alot?



# obtain predictions Q and prediction variance U
predm.glm <- lapply(getfit(fit.glm2), predict, se.fit = TRUE)
Q.glm <- sapply(predm.glm, `[[`, "fit")
U.glm <- sapply(predm.glm, `[[`, "se.fit")^2
dfcom.glm <- getfit(fit.glm2)[[1]]$df.null

predm.mars <- lapply(getfit(fit.mars2), predict, se.fit = TRUE)
Q.mars <- sapply(predm.mars, `[[`, "fit")
U.mars <- sapply(predm.mars, `[[`, "se.fit")^2
dfcom.mars <- getfit(fit.mars2)[[1]]$df.null

predm.gam <- lapply(getfit(fit.gam2), predict, se.fit = TRUE)
Q.gam <- sapply(predm.gam, `[[`, "fit")
U.gam <- sapply(predm.gam, `[[`, "se.fit")^2
dfcom.gam <- getfit(fit.gam2)[[1]]$df.null



# pool predictions for glm
pred.glm <- matrix(NA, nrow = nrow(Q.glm), ncol = 3, dimnames = list(NULL, c("fit", "se.fit", "df")))
for(i in 1:nrow(Q.glm)) {
  pi <- pool.scalar(Q.glm[i, ], U.glm[i, ], n = dfcom.glm + 1)
  pred.glm[i, 1] <- pi[["qbar"]]
  pred.glm[i, 2] <- sqrt(pi[["t"]])
  pred.glm[i, 3] <- pi[["df"]]
}

pred.mars <- matrix(NA, nrow = nrow(Q.mars), ncol = 3, dimnames = list(NULL, c("fit", "se.fit", "df")))
for(i in 1:nrow(Q.mars)) {
  pi <- pool.scalar(Q.mars[i, ], U.mars[i, ], n = dfcom.mars + 1)
  pred.mars[i, 1] <- pi[["qbar"]]
  pred.mars[i, 2] <- sqrt(pi[["t"]])
  pred.mars[i, 3] <- pi[["df"]]
}

pred.gam <- matrix(NA, nrow = nrow(Q.gam), ncol = 3, dimnames = list(NULL, c("fit", "se.fit", "df")))
for(i in 1:nrow(Q.gam)) {
  pi <- pool.scalar(Q.gam[i, ], U.gam[i, ], n = dfcom.gam + 1)
  pred.gam[i, 1] <- pi[["qbar"]]
  pred.gam[i, 2] <- sqrt(pi[["t"]])
  pred.gam[i, 3] <- pi[["df"]]
}

par(mfrow=c(3,1))
plot(x=density(pred.glm[,1]))
plot(x=density(pred.mars[,1]))
plot(x=density(pred.gam[,1]))

























