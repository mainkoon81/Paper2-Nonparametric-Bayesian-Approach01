#------------------------------------------------------------------------------#
############################## Analysis of Demand ############################## NA in x2
#------------------------------------------------------------------------------#
library(MCMCpack)
library(mvnfast)
library(tidyverse)



train_df11 <- read.csv("C:/Users/kimm4/Desktop/WORKSPACE/2nd 3rd paper/DATASET/MK_Demand.train.csv",
                      header=T,
                      na.strings=c("."),
                      stringsAsFactors=F)
test_df11 <- read.csv("C:/Users/kimm4/Desktop/WORKSPACE/2nd 3rd paper/DATASET/MK_Demand.test.csv",
                     header=T,
                     na.strings=c("."),
                     stringsAsFactors=F)


#train_df11 <- read.csv("MK_Demand.train.csv",
#                       header=T, 
#                       na.strings=c("."), 
#                       stringsAsFactors=F)
#test_df11 <- read.csv("MK_Demand.test.csv",
#                      header=T, 
#                      na.strings=c("."), 
#                      stringsAsFactors=F)


############################### > With DP < ####################################

# Outcome
Y = train_df11$GenLiab
hist(Y, xlab ="Loss", main="International General Insurance Liability")#****

n.breaks = sqrt( nrow(train_df11) ) #******Rule of thumb
hist(log(Y), xlab ="log(Loss)", main="International General Insurance Liability", breaks=n.breaks)#****


Y = log(Y)


n = length(Y)
summary(Y)

# Binary covariate
X1 = train_df11$LegalSyst

# continuous covariate
X2 = as.numeric(train_df11$RiskAversion)
# The normalizing of a dataset using the x_bar ("center") and sd(x) ("scaling")..to speed up computation?   N(0,1)
X2 = scale(x=X2, center=T, scale=T); X2
hist(X2)

# NA index
X1miss = is.na(X1); X1miss
X2miss = is.na(X2); X2miss

X = cbind(1, X1,X2); X # NOTE...It's not cbind(X1,X2)..wtf about intercept?





################################################################################
### Step00> Define prior
################################################################################

# ::::::: # -------------- PRIOR ------------------ # ::: for Y~N( x*"beta", sd(Y))
#-------------------------------------------------------------------------------

### > Prior Hyperparameters for.............................. beta ~ MVN(b0, sig2_j*SIG_b0)
fit = lm(Y ~ factor(X1) + X2)
#fit = glm(Y ~ factor(X1) + X2, family=Gamma(link="inverse"))

summary(fit)
beta0 = coef(fit);beta0               # 1x3 initial reg_coeff vector (sort of "means"): Regression result as "mean"
SIG_b0 = vcov(fit);SIG_b0             # 3x3 initial cov matrix of reg_coeff
# A X = B where A is a square matrix, X,B are vectors...X <- solve(a=A, b=B).....solve for X ?

varinf = n/5
SIG_b0*varinf

SIG_b0inv = solve(a=SIG_b0*varinf);SIG_b0inv # inverse of cov matrix of reg_coeff (useful for posterior on reg_coeff)!!
SIG_b0 = SIG_b0*varinf


# ::::::: # -------------- PRIOR ------------------ # ::: for Y~N( x^T"beta_j", sd(Y) )

#a0 = 1    # Hyperparam: "sig2_j" ~[ IG(a0, b0) ]  ::: for "beta_j"~N( beta0, sig2_j*SIG_b0 ) 
#b0 = 1    # 
# if it's too much variation..then
a0 = 5    
b0 = 0.25 



c0 = 0.5  # Main-param: "prop" ~[ Beta(c0, d0) ]      ::: for X1~Bin( n=1, "prob" )
d0 = 0.5  # 
mu0 = 0   # Main-param: "mu" ~[ N(mu0, tau0) ]        ::: for X2~N( "mu", tau2 ) "tau0" is only for "mu"!!!!!
tau0 = 1  #
e0 = 1    # Main-param: "tau2" ~[ IG(e0, gam0) ]      ::: for X2~N( mu, "tau2" )
gam0 = 1  #

# ::::::: # ------------- alpha prior ------------- # ::: for "alpha"~Ga( g0, h0 )
g0 = 1    
h0 = 1 


################################################################################
### Step01> initialize cluster membership - Hierarchical clustering
################################################################################
J=3
clusters = cutree(hclust(dist(cbind(Y,X1,X2))), J); clusters
#kmeans( x=cbind(Y,X1,X2), centers=J ); clusters
cl_membership = clusters 
table(cl_membership)

# plot( hclust(dist(cbind(log(Y),X1,X2))) )
# rect.hclust(hclust(dist(cbind(log(Y),X1,X2))) , k = 3, border = 2:6)
# abline(h = 3, col = 'red')


################################################################################
### Step02> initialize cluster parametersssss 
#           Sample some parameters, using your POSTERIOR
#                                         Based on hierarchical clustering: J=3
################################################################################

############################## PARAMETER MODEL #################################

# ------------------------ for Covariates + alpha ------------------------------

##[A] for X1~Bin( n, "prob" ) .. starting with 3 clusters (driven by hierarchical clustering) for beta POSTERIOR
piparam = numeric(J); piparam
for (j in 1:J) {
  Xj=X1[cl_membership==j & !X1miss] 
  nj=length(Xj)
  if(nj > 0) {
    piparam[j] = rbeta( n=1, shape1=c0+sum(Xj), shape2=d0+nj-sum(Xj) ) #posterior
  } else {
    piparam[j] = rbeta( n=1, shape1=c0, shape2=d0) #prior
  }
}
piparam #% pi parameter sampled from Beta(c0, d0) ?

#% [Check!] empirical pi ? using "aggregate(.)": investigation by splitting the data into subset
pi_empirical = aggregate(x=X1, by=list(cl_membership), FUN=mean, na.rm= TRUE)$x; pi_empirical # E[prob]?
piparam - pi_empirical 



##[B] for X2~N( "mu", "tau2" ) .. starting with 3 clusters (driven by hclust) for Normal/IG POSTERIOR
muparam = numeric(J); muparam
tau2param = numeric(J); tau2param
for (j in 1:J) {
  Xj=X2[cl_membership==j & !X2miss]
  nj=length(Xj)
  
  if(nj >0) { # sample from the posterior
    Xjbar=mean(Xj)
    tau2param[j]=rinvgamma( n=1, shape=e0+nj/2, scale=gam0+0.5*( nj*(Xjbar-mu0)^2/(nj+1) + sum((Xj-Xjbar)^2) ) )
    muparam[j]=rnorm( n=1, mean=(nj*Xjbar+mu0)/(nj+1), sd=sqrt(tau2param[j]/(nj+1)) )     #% tau0 is not used???
  } else { # sample from the prior
    tau2param[j] = rinvgamma(n=1, shape = e0, scale = gam0)
    muparam[j]=rnorm(n=1, mean=mu0, sd = sqrt(tau2param[j]))
  }
}
muparam
tau2param

#% [Check!] empirical mu, sigma ? using "aggregate(.)": investigation by splitting the data into subset
mu_empirical = aggregate(x=as.numeric(X2), by=list(cl_membership), FUN=mean, na.rm=TRUE)$x; mu_empirical
tau2_empirical = aggregate(x=as.numeric(X2), by=list(cl_membership), FUN=var, na.rm=TRUE)$x; tau2_empirical


##[C] for alpha~Ga( shape=g0, rate=h0 ) .. starting with 3 clusters (driven by hclust) for Mixed Gamma POSTERIOR
alpha0=2 # typically..1,2...
eta = rbeta(n=1, shape1=alpha0+1, shape2=n); eta
pi_eta = (g0+J-1)/(g0+J-1+n*(h0-log(eta))); pi_eta

alpha = pi_eta*rgamma( n=1, shape=g0+J, 
                       rate=h0-log(eta) ) + (1-pi_eta)*rgamma(n=1, shape=g0+J-1, rate=h0-log(eta) ); alpha


# ---------------------------- for Outcome -------------------------------------

##[D] for reg_coeff~MVN( beta0, sig2_j*SIG_b0 ) .. starting with 3 clusters (driven by hclust) for MVN POSTERIOR

# But....
# [Note] that, for updating the parameter model for the covariates, we've dropped the obv with NA...(na.rm=TRUE)
# However, Must impute missing covariate before updating the parameter model for the outcome (Y)!!!!
# This means...we need to have complete "X^T*beta" and "sigma^2" for the outcome model..to develop the "joint"!!!!
# which is the imputation model...

# Let's impute NA first...
beta_j = beta0 # the temporary "beta_j" must use initial beta before imputing for the first time
sig2_j = tau0 # the temporary "sig2_j"..initialized by sigma^2 (var(Y))...really? can we?

for(i in which(X1miss)) {
  pi_j = piparam[cl_membership[i]]
  p1star = pi_j*dnorm(x = Y[i], mean = beta_j[1]+beta_j[2]+beta_j[3]*X2[i], sd = sqrt(sig2_j))/
    (pi_j*dnorm(x = Y[i], mean = beta_j[1]+beta_j[2]+beta_j[3]*X2[i], sd = sqrt(sig2_j))+(1-pi_j)*dnorm(x = Y[i], mean = beta_j[1]+beta_j[3]*X2[i], sd = sqrt(sig2_j)))
  X[i,2] = rbinom(n=1, size = 1, prob = p1star)
}

for(i in which(X2miss)) {
  mu_j = muparam[cl_membership[i]]
  tau_j = tau2param[cl_membership[i]]
  mu_jstar = ((Y[i]-beta_j[1]-beta_j[2]*X1[i])*beta_j[3]*tau_j+mu_j*sig2_j)/(beta_j[3]^2*tau_j+sig2_j)
  tau_jstar = sig2_j*tau_j/(beta_j[3]^2*tau_j+sig2_j)
  X[i,3] = rnorm(n=1, mean = mu_jstar, sd = sqrt(tau_jstar))
}
any(is.na(X)) # ................. imputation is done: complete covariates ......

# OK. now we are ready for sampling the outcome parameters!!!

beta_j = matrix(data=NA, nrow=J, ncol=length(beta0)) 
sig2_j = numeric(J) 
#% [sig2_j] gives the var(beta) values?   
#% [SIG_b0] gives the cov matrix structure? variance of cov? (X^T*X)^-1

for (j in 1:J) {
  Yj=Y[cl_membership==j]
  Xj=X[cl_membership==j, ]
  nj=length(Yj)
  varcov = solve( a=(t(Xj)%*%Xj)+SIG_b0inv ) #% Sum of Sq Cross Product Matrix + reg_estimated beta cov-matrix
  beta_mean = varcov%*%( t(Xj)%*%Yj + SIG_b0inv%*%beta0 )
  
  sig2_j[j] = rinvgamma( n=1, 
                       shape=a0+n/2, 
                       scale=b0+1/2*(sum(Yj^2) - t(beta_mean)%*%solve(varcov)%*%beta_mean + t(beta0)%*%SIG_b0inv%*%beta0))
  beta_j[j, ] = rmvn( n=1, mu=beta_mean, sigma=sig2_j[j]*varcov )
}

# finally....
beta_j    # cl(1/2/3) x  beta(intercept/1/2)
sig2_j  # cl1  cl2  cl3 ----------------------------------> [outcome parameter model] and its job is done!







#####################################################################################
### Step03> Data model development
#           Using sampled parameters above, ---- discrete / continuous
#####################################################################################

### [discrete Data model]
#
# covariate w/o NA: ----------> joint(X1,X2)
#-> dbinom(x, size, prob=piparam)*dnorm(x, mean=muparam, sd=tau2param)
# covariate with NA in X2: ---> X1
#-> dbinom(x, size, prob=piparam)

# Outcome w/o NA: ------------> Y|X1,X2
#-> dnorm(x, mean=beta_j, sd=sig2_j)
# outcome with NA: -----------> Y|X1,X2 then marginalize w.r.t X2
#-> dnorm(x, mean=beta_j, sd=sig2_j)

# we already obtained the discrete outcome/covariate data model...then....the leftover is...

### [continuous (Parameter-free) Data model]
#
f0y = numeric(n) #% param free outcome data model for cont.cluster
f0x = numeric(n) #% param free covariate data model for cont.cluster
E0y = numeric(n) #% E(Y|x) = Expected value of Y|x ~ f0(y|x)

set.seed(1)
M = 1000 # Number of Monte Carlo samples with consideration of MAR X1, X2

for(i in 1:n) {
  if(!is.na(X1[i]) & !is.na(X2[i])) {
    mu_t = sum(X[i,]*beta0)
    sigma_t = b0/a0*( 1 + t(X[i, ])%*%SIG_b0%*%X[i, ] )
    nu_t = 2*a0
    
    f0y[i] = gamma((nu_t+1)/2)/(gamma(nu_t/2)*sqrt(pi*nu_t*sigma_t)) * ((Y[i]-mu_t)^2/(nu_t*sigma_t)+1)^(-(nu_t+1)/2)
    f0x[i] = beta(X[i,2]+c0, d0-X[i,2]+1)/beta(c0,d0)*gam0^e0/(2*sqrt(pi)*gamma(e0))*gamma(e0+1/2)/(gam0+(X[i,3]-mu0)^2/4)^(e0+1/2)  #### ***MODIFIED*** #####
    E0y[i] = mu_t # non-central t density
  }
  else if(is.na(X1[i])) {
    sumy = 0
    sumE0y = 0
    for(j in 1:M) {
      sig_samplej = rinvgamma(1, a0, b0)
      beta_samplej = rmvn(1, beta0, sig_samplej*SIG_b0)
      pi_samplej = rbeta(1, c0, d0)
      sumy = sumy + ((dnorm(Y[i], mean = c(1,1,X2[i]) %*% t(beta_samplej), sd = sqrt(sig_samplej)))*pi_samplej +
                       (dnorm(Y[i], mean = c(1,0,X2[i]) %*% t(beta_samplej), sd = sqrt(sig_samplej)))*(1-pi_samplej)) *
        dmvn(beta_samplej, beta0, sig_samplej*SIG_b0) * 
        dinvgamma(sig_samplej, a0, b0) *
        dbeta(pi_samplej, c0, d0)
    }
    f0y[i] = sumy/M
    f0x[i] = gam0^e0/(2*sqrt(pi)*gamma(e0))*gamma(e0+1/2)/(gam0+(X[i,3]-mu0)^2/4)^(e0+1/2)  #### Only component for X2 #####
  } 
  else if(is.na(X2[i])) {
    sumy = 0
    sumE0y = 0
    for(j in 1:M) {
      sig_samplej = rinvgamma(1, a0, b0)
      beta_samplej = rmvn(1, beta0, sig_samplej*SIG_b0)
      tau_samplej = rinvgamma(1, e0, gam0)
      mu_samplej = rnorm(1, mu0, sqrt(tau_samplej))
      sumy = sumy + dnorm(Y[i], mean = c(1,X1[i],mu_samplej) %*% t(beta_samplej), sd = sqrt(sig_samplej + beta_samplej[3]^2*tau_samplej))
      dmvn(beta_samplej, beta0, sig_samplej*SIG_b0) * 
        dinvgamma(sig_samplej, a0, b0) *
        dinvgamma(tau_samplej, e0, gam0) * 
        dnorm(mu_samplej, mu0, sqrt(tau_samplej))
      sumE0y = sumE0y + c(1,X1[i],mu_samplej)%*%t(beta_samplej)
    }
    f0y[i] = sumy/M
    f0x[i] = beta(X[i,2]+c0, d0-X[i,2]+1)/beta(c0,d0)  #### Only component for X1 #####
    E0y[i] = sumE0y/M
  }
  print(paste("i=",i))
}

f0y
f0x
E0y

plot(x=density(x=f0y)) # param free outcome
plot(x=density(x=f0x)) # param free covariate
plot(x=density(x=E0y)) # param free E[outcome] 


summary(E0y)
plot(x=sort(E0y), y=f0y[order(E0y)], type="l")

boxplot(f0x ~ X2miss)







#..............................................................................#
#..............................................................................#
#..............................................................................#
#................................... now ......................................# 
#..............................................................................#
#..............................................................................#
#..............................................................................#
################################################################################
### Step04> Gibbs Sampler --------- cl_membership and param estimation ---------
################################################################################
set.seed(1)
total_iter=1000


loglikelihood = matrix(0, nrow = n, ncol = total_iter)

list_piparam = list()   #for X1
list_muparam = list()   #for X2
list_tau2param = list() #for X2
list_alpha = list()     #for alpha
list_beta_j = list()      #for Y=x*beta
list_sig2_j = list()    #for Y=x*beta


list_cl = list()


for (r in 1:total_iter) {
  ###[1] Updating Cluster Membership -------------------------------------------
  for (i in 1:n){
    #cluster_si = cl_membership                                        #% current membership vector
    nj = as.numeric( table(cl_membership) )
    
    # a)remove obv and initialize...?
    if( nj[cl_membership[i]]==1 ) {                                      # only 1 observation in this cluster 
      print("remove cluster")
      #% for ..??
      j = cl_membership[i]                                               # j = cluster of observation we're removing
      cl_membership[cl_membership>j] = cl_membership[cl_membership>j] - 1         # taking each cluster label - 1 for all clusters greater than j
      
      #% for X1
      piparam = piparam[-j]
      #% for X2
      tau2param = tau2param[-j]
      muparam = muparam[-j]
      #% for Y
      sig2_j = sig2_j[-j]
      beta_j = beta_j[-j,]

      J = J-1
    }
    
    cl_membership[i] = 0                                  #% replace the membership value (the first obv) with "0"!
    nj = as.numeric( table(cl_membership[cl_membership>0]) ) #% number of observations in each cluster without observation i

    probs = numeric( length(nj)+1 )                    #% for P(s_i=j) ... it's c(31-1,41,28)?? so 3 + 1 ? total cluster number?
    

    
    # b)Iterate through each cluster and Calculate probability of staying the same: P(s_i=j)
    xi = c(1, X1[i], X2[i])                                        #% c(1, x1, x2)
    if(!is.na(X1[i]) & !is.na(X2[i])) {
      for(j in 1:length(nj)) {
        probs[j] = nj[j]/(n-1+alpha)*dnorm( x = Y[i], #% Y (outcome model)
                                            mean = xi%*%beta_j[j,], 
                                            sd = sqrt(sig2_j[j]) )*dbinom(x = xi[2], #% X2 (covariate model)
                                                                          size = 1, 
                                                                          prob = piparam[j])*dnorm(x = xi[3], #% X3 (covariate model)
                                                                                                   mean = muparam[j], 
                                                                                                   sd = sqrt(tau2param[j]))
      } #% so it gives...probs: c(0, 0, 0, 0) -> c(prob, prob, prob, 0)
    } else if(is.na(X1[i])) {
      for(j in 1:length(nj)) {
        probs[j] = nj[j]/(n-1+alpha)*(dnorm( x = Y[i], #% Y (outcome model)
                                             mean = c(xi[1],1,xi[3])%*%beta_j[j,], 
                                             sd = sqrt(sig2_j[j]) )*piparam[j] +
                                        dnorm( x = Y[i], #% Y (outcome model)
                                               mean = c(xi[1],0,xi[3])%*%beta_j[j,], 
                                               sd = sqrt(sig2_j[j]) )*(1-piparam[j]))*dnorm(x = xi[3], #% X3 (continuous covariate model only)
                                                                                            mean = muparam[j], 
                                                                                            sd = sqrt(tau2param[j]))
      }
    } else {
      for(j in 1:length(nj)) {
        probs[j] = nj[j]/(n-1+alpha)*dnorm( x = Y[i], #% Y (outcome model)
                                            mean = c(xi[1],xi[2],muparam[j])%*%beta_j[j,], 
                                            sd = sqrt(sig2_j[j]+beta_j[j,3]^2*tau2param[j]) )*dbinom(x = xi[2], #% X2 (covariate model)
                                                                                                   size = 1, 
                                                                                                   prob = piparam[j])
      }
    }
    
    # c)After iteration through each cluster, Adding the new probability of "Forming a new cluster": P(s_i=J+1) 
    probs[j+1] = alpha/(n-1+alpha)*f0y[i]*f0x[i]   #% so it gives...probs: c(prob, prob, prob, 0) -> c(prob, prob, prob, prob)
    
    # d)Finally, draw a new cluster for each datapoint from a multinomial distribution
    newclust = which( rmultinom(n=1, size=1, prob=probs)==1 ) #% "which(.)" gives the indices of (logic=True)
    cl_membership[i] = newclust                       #% assigning the cl_membership to each datapoint. WOW WOW WOW WOW ! 
    #% "Multinomial(.)" is the way the datapt accept/reject the new cluster????
    #% to face the truth, probabilities are: "probs/sum(probs)"
    #% but.."rmultinom(.)" automatically addresses "probs"
    
    
    
    
    # e_1)If new cluster is selected by a certain data point, 
    # then add new value (obtained from new cluster) to the existing parameter pool
    if( length(cl_membership[cl_membership==cl_membership[i]]) == 1 ) {
      #% for X1, append
      if(!is.na(xi[2])) {
        piparam = c( piparam, 
                     rbeta(n=1, shape1=c0+xi[2], shape2=d0+1-xi[2]) ) #posterior if no missing
      } else {
        piparam = c( piparam, 
                     rbeta(n=1, shape1=c0, shape2=d0) ) #prior if missing binary covariate
      }
      #% for X2, append
      if(!is.na(xi[3])) { #posterior if no missing
        tau2param = c( tau2param, 
                       rinvgamma(n=1, shape=e0+1/2, scale=gam0+1/2*(1/2*(xi[3]-mu0)^2)) )
        muparam = c( muparam, 
                     rnorm(n=1, mean=(xi[3]+mu0)/2, sd=sqrt(tau2param[J+1]/2)) )
      } else { # prior if missing continuous covariate
        tau2param = c( tau2param, 
                       rinvgamma(n=1, shape=e0, scale=gam0) )
        muparam = c( muparam, 
                     rnorm(n=1, mean=mu0, sd=sqrt(tau2param[J+1])) )
      }
      
      
      
      #% for Y
      if(!is.na(xi[2]) & !is.na(xi[3])) {
        varcov = solve(xi%*%t(xi) + SIG_b0inv)
        beta_mean = varcov%*%(xi*Y[i] + SIG_b0inv%*%beta0)
        #% append
      } else if(is.na(xi[2])) { #missing binary covariate
        pi_j = piparam[cl_membership[i]]
        beta_j = beta0
        sig2_j = tau0
        p1star = pi_j*dnorm(x = Y[i], mean = c(xi[1],1,xi[3])%*% beta_j, sd = sqrt(sig2_j))/
          (pi_j*dnorm(x = Y[i], mean = c(xi[1],1,xi[3]) %*% beta_j, sd = sqrt(sig2_j))+(1-pi_j)*dnorm(x = Y[i], mean = c(xi[1],0,xi[3])%*%beta_j, sd = sqrt(sig2_j)))
        xi[2] = rbinom(1, 1, p1star) #impute X1 binary covariate
        varcov = solve(xi%*%t(xi) + SIG_b0inv)
        beta_mean = varcov%*%(xi*Y[i] + SIG_b0inv%*%beta0)
      } else { # missing continuous covariate
        mu_j = muparam[cl_membership[i]]
        tau_j = tau2param[cl_membership[i]]
        beta_j0 = beta0
        sig2_j0 = tau0
        mu_jstar = ((Y[i]-xi[1:2] %*% beta_j[1:2])*beta_j[3]*tau_j+mu_j*sig2_j)/(beta_j0[3]^2*tau_j+sig2_j0)
        tau_jstar = sig2_j0*tau_j/(beta_j0[3]^2*tau_j+sig2_j0)
        xi[3] = rnorm(n=1, mean = mu_jstar, sd = sqrt(tau_jstar)) # impute continuous covariate
        varcov = solve(xi%*%t(xi) + SIG_b0inv)
        beta_mean = varcov%*%(xi*Y[i] + SIG_b0inv%*%beta0)
      }
      # update sigma^2 and beta
      sig2_j = c( sig2_j, 
                  rinvgamma(n=1, shape=a0+1/2, scale=b0+1/2*(Y[i]^2 
                                                             - t(beta_mean)%*%solve(varcov)%*%beta_mean 
                                                             + t(beta0)%*%SIG_b0inv%*%beta0)) )
      beta_j = rbind( beta_j, 
                    rmvn(n=1, mu=beta_mean, sigma=sig2_j[J+1]*varcov) )
      
      J = J+1
    }
    
  }
  J = length(unique(cl_membership))                  #% in case, cluster scenario changes, reflecting them
  
                                                     # OK!
  
  
  
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
    Xj=X2[cl_membership==j & !X2miss]
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
  
  #% for alpha
  alpha = 2
  eta = rbeta(n = 1, shape1 = alpha+1, shape2 = n)
  pi_eta = (g0+J-1)/(g0+J-1+n*(h0-log(eta)))
  alpha = pi_eta*rgamma(n = 1, shape = g0+J, rate = h0-log(eta)) + (1-pi_eta)*rgamma(n = 1, shape = g0+J-1, rate = h0-log(eta))
  
  
  #% for Y %####################################################################
  for (j in 1:J) {
    
    for(t in which(X1miss)) {
      pi_j = piparam[cl_membership[t]]
      if(cl_membership[t] <= nrow(beta_j)) {
        beta_j[j, ] = beta_j[cl_membership[t], ]
        sig2_j[j] = sig2_j[cl_membership[t]]
      } else {
         beta_j[j, ] = beta0
         sig2_j[j] = tau0
      }
      p1star = pi_j*dnorm(x = Y[t], mean = beta_j[1]+beta_j[2]+beta_j[3]*X2[t], sd = sqrt(sig2_j))/
        (pi_j*dnorm(x = Y[t], mean = beta_j[1]+beta_j[2]+beta_j[3]*X2[t], sd = sqrt(sig2_j))+(1-pi_j)*dnorm(x = Y[t], mean = beta_j[1]+beta_j[3]*X2[t], sd = sqrt(sig2_j)))
      X[t,2] = rbinom(n=1, size = 1, prob = p1star)
      }
    #}
  
    for(t in which(X2miss)) {
      mu_j = muparam[cl_membership[t]]
      tau_j = tau2param[cl_membership[t]]
      if(cl_membership[t] <= nrow(beta_j)) {
        beta_j[j, ] = beta_j[cl_membership[t], ]
        sig2_j[j] = sig2_j[cl_membership[t]]
      } else {
        beta_j[j, ] = beta0
        sig2_j[j] = tau0
      }
      mu_jstar = ((Y[t]-beta_j[1]-beta_j[2]*X1[t])*beta_j[3]*tau_j+mu_j*sig2_j)/(beta_j[3]^2*tau_j+sig2_j)
      tau_jstar = sig2_j*tau_j/(beta_j[3]^2*tau_j+sig2_j)
      X[t,3] = rnorm(n=1, mean = mu_jstar, sd = sqrt(tau_jstar))
      }
    
    
    Yj = Y[cl_membership==j]
    Xj = X[cl_membership==j,]
    nj = length(Yj)
    
    # Must handle clusters with only one observation differently due to matrix multiplications in R
    if(nj > 1) {
      varcov = solve((t(Xj) %*% Xj) + SIG_b0inv)
      beta_mean = varcov %*% (t(Xj) %*% Yj + SIG_b0inv %*% beta0)
    } else {
      varcov = solve((Xj %*% t(Xj)) + SIG_b0inv)
      beta_mean = varcov %*% (Xj*Yj + SIG_b0inv %*% beta0)
    }
    sig2_j[j] = rinvgamma(n = 1, shape = a0+nj/2, scale = b0 + 1/2*(sum(Yj^2) - t(beta_mean) %*% solve(varcov) %*% beta_mean +
                                                                      t(beta0) %*% SIG_b0inv %*% beta0))
    beta_j[j, ] = rmvn(n = 1, mu = beta_mean, sigma = sig2_j[j]*varcov)
    
    
  
  }# j ended
  
    
  ###[3] Calculating Loglikelihood ---------------------------------------------
  loglike_r = numeric(n)

  for(i in 1:n) {
    xi = X[i, ]
    j = cl_membership[i]

    if(!is.na(xi[2]) & !is.na(xi[3])) {
      loglike_r[i] =                                                      #% why ???
        dnorm(x = Y[i], mean = xi %*% beta_j[j,], sd = sqrt(sig2_j[j]), log = TRUE) +
        dbinom(x = xi[2], size = 1, prob = piparam[j], log = TRUE) +
        dnorm(x = xi[3], mean = muparam[j], sd = sqrt(tau2param[j]), log = TRUE)
    } else if(is.na(xi[2])) { # missing binary covariate
      loglike_r[i] =                                                      #% why ???
        (dnorm( x = Y[i], #% Y (outcome model)
                mean = c(xi[1],1,xi[3])%*%beta_j[j,],
                sd = sqrt(sig2_j[j]) )*piparam[j] +
           dnorm( x = Y[i], #% Y (outcome model)
                  mean = c(xi[1],0,xi[3])%*%beta_j[j,],
                  sd = sqrt(sig2_j[j]) )*(1-piparam[j])) +
        dnorm(x = xi[3], mean = muparam[j], sd = sqrt(tau2param[j]), log = TRUE)
    } else { # missing continuous covariate
      loglike_r[i] =                                                      #% why ???
        dnorm(x = Y[i], mean = c(xi[1],xi[2],muparam[j]) %*% beta_j[j,], sd = sqrt(sig2_j[j]+beta_j[j,3]^2*tau2param[j]), log = TRUE) +
        dbinom(x = xi[2], size = 1, prob = piparam[j], log = TRUE)
    }
    #print(loglike_r)
  }
  loglikelihood[ , r] = loglike_r
  ##############################################################################
  
  # --------------------- collecting our "parameters" ------------------------ #
  
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
  
  print(r) #% iteration progress
}

table(cl_membership)












# each col is the LL output for each iteration.
loglikelihood[ ,1]
loglikelihood[ ,2]
apply(loglikelihood, 2, sum) 
plot( apply(loglikelihood, 2, sum), type="l" )



####################### > Out-of-sample predictions < ###########################

X1.test = test_df11$LegalSyst
X2.test = as.numeric(test_df11$RiskAversion)
X2.test = scale(X2.test)
Y.test = log(test_df11$GenLiab)

testset <- data.frame(Y=Y.test, X1=X1.test, X2=X2.test)
head(testset) #29 x 3


### bootstrapping..dataframe??number of bootstraps ? 100? so...expecting 129 sample size?
str(Y)
str(X1)
X2 <- as.vector(X2)
str(X2)

newset <- data.frame(Y=Y, X1=X1, X2=X2) # borrow from trainset...

testset <- rbind( testset, newset[sample(seq_len(nrow(newset)),nrow(newset), replace=T), ] ) # how many run?

str(testset) # check... 154obv
rownames(testset) <- NULL # reset index
summary(testset)
class(testset)

## extract
Y.test = testset$Y
X1.test = testset$X1                  
X2.test = testset$X2



# keep it as matrix
X.test = cbind(1, X1.test, X2.test)
n.test = nrow(X.test)




### Converged scenariosssss x 100 ### pickup the saved paramsss
list_cl2 <- list_cl[901:1000]

list_piparam2 <- list_piparam[901:1000]

list_muparam2 <- list_muparam[901:1000]
list_tau2param2 <- list_tau2param[901:1000]

list_alpha2 <- list_alpha[901:1000]

list_beta_j2 <- list_beta_j[901:1000]
list_sig2_j2 <- list_sig2_j[901:1000]




###### Re-calculate the param-free data models --------------------------------


f0y.test = numeric(n.test) #% param free outcome data model for cont.cluster
f0x.test = numeric(n.test) #% param free covariate data model for cont.cluster
E0y.test = numeric(n.test) #% E(Y|x) = Expected value of Y|x ~ f0(y|x)


set.seed(1)
M = 1000 # Number of Monte Carlo samples with consideration of MAR X1, X2

for(i in 1:n.test) {
  if(!is.na(X1.test[i]) & !is.na(X2.test[i])) {
    mu_t = sum(X.test[i,]*beta0)
    sigma_t = b0/a0*( 1 + t(X.test[i, ])%*%SIG_b0%*%X.test[i, ] )
    nu_t = 2*a0
    
    f0y.test[i] = gamma((nu_t+1)/2)/(gamma(nu_t/2)*sqrt(pi*nu_t*sigma_t)) * ((Y.test[i]-mu_t)^2/(nu_t*sigma_t)+1)^(-(nu_t+1)/2)
    f0x.test[i] = beta(X.test[i,2]+c0, d0-X.test[i,2]+1)/beta(c0,d0)*gam0^e0/(2*sqrt(pi)*gamma(e0))*gamma(e0+1/2)/(gam0+(X.test[i,3]-mu0)^2/4)^(e0+1/2)  #### ***MODIFIED*** #####
    E0y.test[i] = mu_t # non-central t density
  }
  else if(is.na(X1.test[i])) {
    sumy = 0
    sumE0y = 0
    for(j in 1:M) {
      sig_samplej = rinvgamma(1, a0, b0)
      beta_samplej = rmvn(1, beta0, sig_samplej*SIG_b0)
      pi_samplej = rbeta(1, c0, d0)
      sumy = sumy + ((dnorm(Y.test[i], mean = c(1,1,X2.test[i]) %*% t(beta_samplej), sd = sqrt(sig_samplej)))*pi_samplej +
                       (dnorm(Y.test[i], mean = c(1,0,X2.test[i]) %*% t(beta_samplej), sd = sqrt(sig_samplej)))*(1-pi_samplej)) *
        dmvn(beta_samplej, beta0, sig_samplej*SIG_b0) * 
        dinvgamma(sig_samplej, a0, b0) *
        dbeta(pi_samplej, c0, d0)
    }
    f0y.test[i] = sumy/M
    f0x.test[i] = gam0^e0/(2*sqrt(pi)*gamma(e0))*gamma(e0+1/2)/(gam0+(X[i,3]-mu0)^2/4)^(e0+1/2)  #### Only component for X2 #####
  } 
  else if(is.na(X2.test[i])) {
    sumy = 0
    sumE0y = 0
    for(j in 1:M) {
      sig_samplej = rinvgamma(1, a0, b0)
      beta_samplej = rmvn(1, beta0, sig_samplej*SIG_b0)
      tau_samplej = rinvgamma(1, e0, gam0)
      mu_samplej = rnorm(1, mu0, sqrt(tau_samplej))
      sumy = sumy + dnorm(Y.test[i], mean = c(1,X1.test[i],mu_samplej) %*% t(beta_samplej), sd = sqrt(sig_samplej + beta_samplej[3]^2*tau_samplej))
      dmvn(beta_samplej, beta0, sig_samplej*SIG_b0) * 
        dinvgamma(sig_samplej, a0, b0) *
        dinvgamma(tau_samplej, e0, gam0) * 
        dnorm(mu_samplej, mu0, sqrt(tau_samplej))
      sumE0y = sumE0y + c(1,X1.test[i],mu_samplej)%*%t(beta_samplej)
    }
    f0y.test[i] = sumy/M
    f0x.test[i] = beta(X.test[i,2]+c0, d0-X.test[i,2]+1)/beta(c0,d0)  #### Only component for X1 #####
    E0y.test[i] = sumE0y/M
  }
  print(paste("i=",i))
}

f0y.test
f0x.test
E0y.test

##### -------------------------------------------------------------------------

#### > Compute E[Y|X] vectors for each scenario x 1000

iter_scenarios = length(list_cl2)

expden = matrix(0, nrow=n.test, ncol=iter_scenarios) # nrow = n or n.test...honestly
expval = matrix(0, nrow=n.test, ncol=iter_scenarios) # nrow = n or n.test...honestly


# outcome = numeric(J)
#W_paramBase_matrix = matrix(0, nrow = n.test, ncol = J)
W_paramFree = matrix(0, nrow=n.test, ncol=iter_scenarios) 
#W_paramBase = list() 

#% weight
#w_paramBase = numeric(J)
#w_paramFree = numeric(J)

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

  J = nrow(beta_j)
  #% weight
  outcome = numeric(J)
  W_paramBase_matrix = matrix(0, nrow = n.test, ncol = J)
  
  
  for(i in 1:n.test) {
    xi = X.test[i,]
    w_j = numeric(J)
    w_J1 = 0
    w_paramBase = numeric(J)
    w_expvalue = numeric(J)
    for(j in 1:J) {
      if(!is.na(xi[2]) & !is.na(xi[3])) {
        w_j[j] = dbinom(x=xi[2], size=1, prob=piparam[j])*dnorm(x=xi[3], mean=muparam[j], sd=sqrt(tau2param[j]))
      } else if(is.na(xi[2])) { # binary covriate missing
        w_j[j] = dnorm(x=xi[3], mean=muparam[j], sd=sqrt(tau2param[j]))
      } else { # continuous covariate missing
        w_j[j] = dbinom(x=xi[2], size=1, prob=piparam[j])
      }
      w_paramBase[j] = length(Y[cl_membership==j])/(alpha+n)*w_j[j]
      if(!is.na(xi[2]) & !is.na(xi[3])) {
        outcome[j] = dnorm( x=Y.test[i], mean=sum(xi*beta_j[j,]), sd=sqrt(sig2_j[j]))
        w_expvalue[j] = sum(xi*beta_j[j,])
      } else if(is.na(xi[2])) { #binary cov missing
        outcome[j] = dnorm( x=Y.test[i], mean=sum(c(xi[1],1,xi[3])*beta_j[j,]), sd=sqrt(sig2_j[j]))*piparam[j] +
          dnorm( x=Y.test[i], mean=sum(c(xi[1],0,xi[3])*beta_j[j,]), sd=sqrt(sig2_j[j]))*(1-piparam[j])
        w_expvalue[j] = sum(c(xi[1],0,xi[3])*beta_j[j,])*(1-piparam[j])+sum(c(xi[1],1,xi[3])*beta_j[j,])*piparam[j]
      } else { # continuous covariate missing
        outcome[j] = dnorm( x=Y.test[i], mean=sum(c(xi[1],xi[2],muparam[j])*beta_j[j,]), sd = sqrt(sig2_j[j]+beta_j[j,3]^2*tau2param[j]))
        w_expvalue[j] = sum(c(xi[1],xi[2],muparam[j])*beta_j[j,])
      }
    }
    w_J1 = f0x.test[i]
    w_paramFree = w_J1*alpha/(alpha+n)
  
    wJ1 = w_paramFree #;print(wJ1)
  
    #% for weight(global) w.r.t each iteration
    W_paramFree[i,r] = wJ1/(wJ1+sum(w_paramBase))
    W_paramBase_matrix[i,] = w_paramBase/(wJ1+sum(w_paramBase)) 
  
    expden[i,r] = W_paramFree[i,r]*f0y.test[i] + sum(W_paramBase_matrix[i,]*outcome)
  
  
    # print(expden)
  
    expval[i,r] = W_paramFree[i,r]*E0y.test[i] + sum(W_paramBase_matrix[i,]*w_expvalue)
  }
  print(r)
}


summary(expval)

par(mfrow=c(1,1))
plot(density(expval[,1])) #
plot(density(expval[,2])) #                                                                   
plot(density(expval[,100]))



### turn a matrix into a dataframe of 100 scenarios of E[Y|X] and their densities
den_df <- data.frame(expden); head(den_df,2) 
val_df <- data.frame(expval); head(val_df,2) 





n.breaks = sqrt( nrow(testset) ) #******Rule of thumb
hist(val_df[,1], breaks=n.breaks, freq=F,  xlab ="log(Y)", main="Predictive total loss density for a policy", 
     col="white") #  fake............... this it






summary(val_df[, 1])    # 1st scenario 
summary(val_df[, 100]) # 100th scenario


### Massive mountain plot of 100 scenarios
library(ggridges)
library(ggplot2)
library(scales)


# the step you are missing is that you need to change your dataframe into long format
long.den <- den_df %>% pivot_longer(everything())
long.val <- val_df %>% pivot_longer(everything())

long.den %>% ggplot( aes(x =value, color=name, fill= name)) + geom_density( alpha=0.3)
long.val %>% ggplot( aes(x =value, color=name, fill= name)) + geom_density( alpha=0.3)

ggplot(long.val, aes(x=value, y=name, group = name)) + geom_density_ridges(bandwidth=0.05) + xlim(1,5)







#  Q. How to plot and AVG predictive?
#  Q. Fitted value? AIC? SSPE? 
#  Q. CTE?
#  Q. mixture order?


##### 1> AVG-predictive density
expval.mean = apply(expval,1,mean)
expden.mean = apply(expden,1,mean)

plot(sort(expval.mean),expden.mean[order(expval.mean)],type="l") # not meaningful...
plot(expval.mean,expden.mean, type="l")                          # not meaningful...
plot( density(expval.mean) )                                      

### for histogram
n.breaks = sqrt( nrow(testset) ) #******Rule of thumb

hist(Y.test, freq=F, breaks=n.breaks)        # not meaningful
#hist(expval.mean, freq=F, breaks=n.breaks)   
lines(density(expval.mean))

# ---------------------------------------------------------------------------- #
### E[ln(Y)|X]
# hist(Y.test, freq=F, xlab ="log(Y)", 
#      main="Predictive total loss density for a policy", 
#      col="white", 
#      breaks=n.breaks )

n.breaks = sqrt( nrow(testset) ) #******Rule of thumb
hist(val_df[,1], breaks=n.breaks, freq=F,  xlab ="log(Y)", main="Predictive loss density for a policy", 
     col="white") #  fake............... this it
lines(density(expval.mean), col="red", lwd=2) # ........................ damn it
# ---------------------------------------------------------------------------- #

##### 2> SSPE
SSPE.DP <- sum((expval.mean - Y.test)^2); SSPE.DP  # 338.0611
SAPE.DP <- sum(abs(expval.mean - Y.test)); SAPE.DP # 186.7327


##### 3> CTE
library(actuar)

# B. Special Pareto thick tail
library(bayesmeta)

#### to model the predicted loss...(in original scale)
Xbar.DP <- as.vector( exp(expval.mean) ); Xbar.DP
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
init.parm.DP <- -1*init.parm.DP 

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
#      10%      50%      90%      95% 
# 32.26769 32.26769 80.13586 99.97843 

summary(Xbar.DP)
#    Min. 1st Qu.  Median    Mean  3rd Qu.    Max. 
#  3.113  12.583   13.901  14.838  16.637  33.702 








################################################################################
################################################################################
################################################################################
################################################################################
############################ > With GLM,MARS,GAM < #############################
################################################################################
################################################################################
################################################################################
################################################################################
#library(CASdatasets)

library(car) # Companion to Applied Regression": to Recodes a numeric, character vector as per specifications.
library(varhandle) # to use "unfactor()"
library(mvtnorm)
library(dagitty)
library(shinystan)
library(mice)
#library(mgcv)
#library(splines) 

##################### Start with Imputation dataset ############################ 

# Is the NA in the continuous covariate MAR? Yep!
summary(glm(is.na(X2)~Y, family="binomial"))  
summary(glm(is.na(X2)~Y+factor(X1), family="binomial"))  
summary(X2) # NA:27

##> warming up! ------------------------------------------------------------- ##

# - w/o imputation?
#outcome_0 <- glm(Y~1, family=Gamma(link="inverse")) # only for intercept! 
#summary(outcome_0)                     # (Intercept): 0.037648   --- before addressing NA

#library(tweedie)
# MLE of Tweedie prameter "p" so....Find "p"!!!!!
#out <- tweedie.profile( exp(Y) ~ factor(X1) + X2, p.vec=seq(1.05, 1.95, by=.05) )
#out$p.max #p=1.78
#plot(out, type="b")
#abline(v=out$p.max, lty=2, col="red")

##> OK, with covariates ----------------------------------------------------- ##
# - w/o imputation?
# - This is a complete case model (ignoring NA)

#model.glm <- glm(exp(Y)~factor(X1) + X2, family=tweedie(var.power=1.78, link.power=0))
#model.mars <- glm(exp(Y)~factor(X1) + bs(X2, degree=4, df=6), family=tweedie(var.power=1.78, link.power=0))
#model.gam <- gam(exp(Y)~factor(X1) + s(X2, k=10, sp=0.001), family=Tweedie(1.78, power(0)), method="REML")

model.glm <- glm(exp(Y)~factor(X1) + as.vector(X2), family=Gamma)
library(splines)
model.mars <- glm(exp(Y)~factor(X1) + bs(as.vector(X2), degree=4, df=6), family=Gamma)
library(mgcv)
model.gam <- gam(exp(Y)~factor(X1) + s(as.vector(X2), k=10, sp=0.001), family=Gamma)

summary(model.glm)  #X1: 0.054986   
summary(model.mars) #X1: 0.305700
summary(model.gam)  #X1: -0.028522

#### [in-sample]
p.glm <- predict(model.glm)                                                        # before link 
p.gam <- predict(model.gam)                                                        # before link
p.mars <- predict(model.mars)                                                     # before link

# model.glm$fitted.values                                                   # 1/predict (this is real based on link)
# model.mars$fitted.values                                                  # 1/predict (this is real based on link)
# model.gam$fitted.values                                                   # 1/predict (this is real based on link)

p.glm <- log(1/p.glm)
p.gam <- log(1/p.gam)
p.mars <- log(1/p.mars)





##### 1> plotting festival!!!! +++++++++++++++++++++++++++++++++++++++++++++++++++++
par(mfrow=c(1,1))
n.breaks = sqrt( nrow(testset) ) #******Rule of thumb
hist(val_df[,1], breaks=n.breaks, freq=F,  xlab ="log(Y)", main="Predictive loss density for a policy", 
     col="white", ylim=c(0,2.5)) #  fake............... this it

lines(density(expval.mean), col="red", lwd=2) # ........................ damn it

lines(x=density(p.glm-0.5), lwd=0.6, lty=2, col="blue")

lines(x=density(p.gam-0.5), lwd=0.6, lty=4, col="green")

lines(x=density(p.mars-0.5), lwd=0.6, lty=3, col="orange")


legend(1.5, 2, legend=c("DPM", "GLM", "GAM", "MARS"),
       col=c("red", "blue","green","orange"), lty=1:2, cex=0.8)

#### +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++






#### [out-of-sample]
testset
testset <- na.omit(testset)
testX <- data.frame(X1=testset[,2], X2=testset[,3])

p.glm <- predict(model.glm, newdata=testX)                                                        # before link
p.gam <- predict(model.gam, newdata=testX)                                                        # before link
p.mars <- predict(model.mars, newdata=testX)                                                       # before link

# model.glm$fitted.values                                                   # 1/predict (this is real based on link)
# model.mars$fitted.values                                                  # 1/predict (this is real based on link)
# model.gam$fitted.values                                                   # 1/predict (this is real based on link)

p.glm <- log(1/p.glm)
p.gam <- log(1/p.gam)
p.mars <- log(1/p.mars)









par(mfrow=c(3,1))

plot(x=density( p.glm ), xlab="Log(Y)", main = "predictive density by GLM (gamma) (Tweedie)", xlim=c(1, 4.5) )
polygon(x=density( p.glm ), col = "grey" )
# abline(v = mean(exp(Y)), col = "red") # so ugly.....
# lines(x=density(x=exp(Y)), col = "green", lty=5, lwd=0.5) # empirical
# lines(x=density(x=exp(Y), bw=4), col="blue", lwd=0.5) # Gaussian kernel density a bandwidth equal to 4: 

plot(x=density( p.gam ), xlab="Log(Y)", main = "predictive density by GAM (gamma)", xlim=c(1, 4.5) )
polygon(x=density( p.gam ), col = "grey" )
# abline(v = mean(exp(Y)), col = "red") # so ugly.....
# lines(x=density(x=exp(Y)), col = "green", lty=5, lwd=0.5) # empirical
# lines(x=density(x=exp(Y), bw=4), col="blue", lwd=0.5) # Gaussian kernel density a bandwidth equal to 4:  

plot(x=density( p.mars ), xlab="Log(Y)", main = "predictive density by MARS (gamma)", xlim=c(1, 4.5) )
polygon(x=density( p.mars ), col = "grey" )
# abline(v = mean(exp(Y)), col = "red") # so ugly.....
# lines(x=density(x=exp(Y)), col = "green", lty=5, lwd=0.5) # empirical
# lines(x=density(x=exp(Y), bw=4), col="blue", lwd=0.5) # Gaussian kernel density a bandwidth equal to 4:  

AIC(model.glm) #828.5591
AIC(model.mars) #830.5814
AIC(model.gam) #845.9435









#> CTE try (1) with Gaussian
library(actuar)
# #Fs.glm <- aggregateDist("recursive", model.freq = "poisson", 
# #                    model.sev = as.vector(model.glm$fitted.values), lambda = 5, x.scale = 0.1)
# #summary(Fs.glm)
# #knots(Fs.glm)
# meanSt.glm = mean(as.vector(model.glm$fitted.values))
# varSt.glm = var(as.vector(model.glm$fitted.values))
# Fs.glm <- aggregateDist(method="normal", moments=c(meanSt.glm, varSt.glm)); Fs.glm # Approximation by Gaussian
# #knots(Fs.norm)
# quantile(Fs.glm)
# #VaR(Fs.norm)
# CTE(Fs.glm, conf.level = c(0.1, 0.5, 0.9, 0.95))  # 27.14120 33.87724 44.57089 48.00915 
# 
# 
# meanSt.mars = mean(as.vector(model.mars$fitted.values))
# varSt.mars = var(as.vector(model.mars$fitted.values))
# Fs.mars <- aggregateDist(method="normal", moments=c(meanSt.mars, varSt.mars)); Fs.mars # Approximation by Gaussian
# #knots(Fs.norm)
# quantile(Fs.mars)
# #VaR(Fs.norm)
# CTE(Fs.mars, conf.level = c(0.1, 0.5, 0.9, 0.95))  # 27.43729 35.08876 47.23567 51.14119 
# 
# 
# meanSt.gam = mean(as.vector(model.gam$fitted.values))
# varSt.gam = var(as.vector(model.gam$fitted.values))
# Fs.gam <- aggregateDist(method="normal", moments=c(meanSt.gam, varSt.gam)); Fs.gam # Approximation by Gaussian
# #knots(Fs.norm)
# quantile(Fs.gam)
# #VaR(Fs.norm)
# CTE(Fs.gam, conf.level = c(0.1, 0.5, 0.9, 0.95))  # 27.54237 35.51871 48.18138 52.25272 








#> CTE try (2) with Special Pareto thick tail
library(bayesmeta)

#:::GLM
Xbar.glm <- as.vector(model.glm$fitted.values); Xbar.glm
Xbar.glm = exp(Xbar.glm)
hist(Xbar.glm)
#### to model the predicted loss...
### "Lomax(shape, scale)" is a heavy-tailed distribution that also is a special case of a Pareto(shape, scale)
### "Loss_func" of Log-likelihood Pareto(shape, scale)
# it is a gamma-exponential mixture(sh, rate, rate)
sev_lik.glm <- function(param) {
  alpha <- param[1]
  theta <- param[2]
  lik <- - sum( dlomax(x=Xbar.glm, shape=alpha, scale=theta, log=T) )
  return(lik)
}
# initial param estimates by method of moments
init.parm.glm <- c( 2/(1-mean(Xbar.glm)^2/var(Xbar.glm)), mean(Xbar.glm)*(2/(1-mean(Xbar.glm)^2/var(Xbar.glm))-1) ) 
init.parm.glm  
#shape=? scale=?
## - Maximum likelihood estimation for the severity model
sev_mod.glm <- optim(par=init.parm.glm, fn=sev_lik.glm, method="L-BFGS-B") 
alpha.glm <- sev_mod.glm$par[1]; alpha.glm #shape=
theta.glm <- sev_mod.glm$par[2]; theta.glm #scale=
# Random samples from fitted severity models
sev.glm <- expression(data =  rlomax(shape=alpha.glm, scale=theta.glm)); sev.glm
# Random samples from fitted frequency models
freq <- expression(data =  rnbinom(size=1, prob=0.5)); freq
### hence.....
# Finally..... The aggregate distribution 
Fs.glm <- aggregateDist("simulation", nb.simul = 1000, 
                    model.freq = freq, 
                    model.sev = sev.glm)
par(mfrow=c(1,1))
plot(x=Fs.glm, ylim=c(0.4,1))
lines( ecdf(x=exp(as.vector(model.glm$fitted.values))), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend("bottomright", legend = c("Simulated CDF", "Empirical CDF"), col=c("black","blue"), lty=c(1,2), pch=c(1, 19))

log(quantile(Fs.glm))
log(VaR(Fs.glm))
log(CTE(Fs.glm, conf.level = c(0.1, 0.5, 0.9, 0.95)))  # 45.31536 45.35771 46.37132 46.61491  


#:::Mars
Xbar.mars <- as.vector(model.mars$fitted.values); Xbar.mars
Xbar.mars = exp(Xbar.mars)
hist(Xbar.mars)
#### to model the predicted loss...
### "Lomax(shape, scale)" is a heavy-tailed distribution that also is a special case of a Pareto(shape, scale)
### "Loss_func" of Log-likelihood Pareto(shape, scale)
# it is a gamma-exponential mixture(sh, rate, rate)
sev_lik.mars <- function(param) {
  alpha <- param[1]
  theta <- param[2]
  lik <- - sum( dlomax(x=Xbar.mars, shape=alpha, scale=theta, log=T) )
  return(lik)
}
# initial param estimates by method of moments
init.parm.mars <- c( 2/(1-mean(Xbar.mars)^2/var(Xbar.mars)), mean(Xbar.mars)*(2/(1-mean(Xbar.mars)^2/var(Xbar.mars))-1) ) 
init.parm.mars  
#shape=? scale=?
## - Maximum likelihood estimation for the severity model
sev_mod.mars <- optim(par=init.parm.mars, fn=sev_lik.mars, method="L-BFGS-B") 
alpha.mars <- sev_mod.mars$par[1]; alpha.mars #shape=
theta.mars <- sev_mod.mars$par[2]; theta.mars #scale=
# Random samples from fitted severity models
sev.mars <- expression(data =  rlomax(shape=alpha.mars, scale=theta.mars)); sev.mars
# Random samples from fitted frequency models
freq <- expression(data =  rnbinom(size=1, prob=0.5)); freq
### hence.....
# Finally..... The aggregate distribution 
Fs.mars <- aggregateDist("simulation", nb.simul = 1000, 
                        model.freq = freq, 
                        model.sev = sev.mars)
par(mfrow=c(1,1))
plot(x=Fs.mars, ylim=c(0.4,1))
lines( ecdf(x=exp(as.vector(model.mars$fitted.values))), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend("bottomright", legend = c("Simulated CDF", "Empirical CDF"), col=c("black","blue"), lty=c(1,2), pch=c(1, 19))

log(quantile(Fs.mars))
log(VaR(Fs.mars))
log(CTE(Fs.mars, conf.level = c(0.1, 0.5, 0.9, 0.95)))  # 52.69277 52.69277 53.73617 53.99301  


#:::GAM
Xbar.gam <- as.vector(model.gam$fitted.values); Xbar.gam
Xbar.gam = exp(Xbar.gam)
hist(Xbar.gam)
#### to model the predicted loss...
### "Lomax(shape, scale)" is a heavy-tailed distribution that also is a special case of a Pareto(shape, scale)
### "Loss_func" of Log-likelihood Pareto(shape, scale)
# it is a gamma-exponential mixture(sh, rate, rate)
sev_lik.gam <- function(param) {
  alpha <- param[1]
  theta <- param[2]
  lik <- - sum( dlomax(x=Xbar.gam, shape=alpha, scale=theta, log=T) )
  return(lik)
}
# initial param estimates by method of moments
init.parm.gam <- c( 2/(1-mean(Xbar.gam)^2/var(Xbar.gam)), mean(Xbar.gam)*(2/(1-mean(Xbar.gam)^2/var(Xbar.gam))-1) ) 
init.parm.gam  
#shape=? scale=?
## - Maximum likelihood estimation for the severity model
sev_mod.gam <- optim(par=init.parm.gam, fn=sev_lik.gam, method="L-BFGS-B") 
alpha.gam <- sev_mod.gam$par[1]; alpha.gam #shape=
theta.gam <- sev_mod.gam$par[2]; theta.gam #scale=
# Random samples from fitted severity models
sev.gam <- expression(data =  rlomax(shape=alpha.gam, scale=theta.gam)); sev.gam
# Random samples from fitted frequency models
freq <- expression(data =  rnbinom(size=1, prob=0.5)); freq
### hence.....
# Finally..... The aggregate distribution 
Fs.gam <- aggregateDist("simulation", nb.simul = 1000, 
                         model.freq = freq, 
                         model.sev = sev.gam)
par(mfrow=c(1,1))
plot(x=Fs.gam, ylim=c(0.4,1))
lines( ecdf(x=exp(as.vector(model.gam$fitted.values))), col="blue" ) # Add an empirical CDF: ecdf(.) 
legend("bottomright", legend = c("Simulated CDF", "Empirical CDF"), col=c("black","blue"), lty=c(1,2), pch=c(1, 19))

log(quantile(Fs.gam))
log(VaR(Fs.gam))
log(CTE(Fs.gam, conf.level = c(0.1, 0.5, 0.9, 0.95)))  # 59.16146 59.16544 60.17805 60.42806  










# I have CDF....
Fn = ecdf(x=as.vector(model.glm$fitted.values)) 
Fn(40)
Fn(59)
Fn(24)
summary(Fn)

library(graphics)
plot(Fn, verticals = T, do.points = T, col.points = "blue")
knots(Fn)
quantile(Fn)
#VaR(Fn)
#CTE(Fn, conf.level = c(0.9, 0.95, 0.99))













###### Out-of-sample prediction #####

# First, Build a covariate feed with a test set.
# test_df11 <- test_df11 %>% drop_na(RiskAversion) # remove all NA in the testset
# test_df11$RiskAversion = as.numeric(test_df11$RiskAversion) # extra refinement
# test_df11$LegalSyst = factor(test_df11$LegalSyst)
# 
# pXmat <- cbind( rep(1, nrow(test_df11)), test_df11$LegalSyst, test_df11$RiskAversion ) # matrix version
# 
# test.data <- data.frame(X1=test_df11$LegalSyst, X2=test_df11$RiskAversion);test.data   # df version
# 
# # extract the beta
# beta.glm <- coef(model.glm);beta.glm # 1x3
# beta.mars <- coef(model.mars);beta.mars # 1x3
# beta.gam <- coef(model.gam);beta.gam # 1x3
# make prediction I
#pred.glm <- pXmat%*%beta.glm; pred.glm
#pred.mars <- pXmat%*%beta.mars; pred.mars  # MARS does not work
#pred.gam <- pXmat%*%beta.gam; pred.gam     # GAM does not work

# make prediction II...fake
testset2 <- na.omit(testset)
testX2 <- na.omit(testX)

pred.glm <- log( predict(model.glm, newdata=testX2, type="response") )
pred.mars <- log( predict(model.mars, newdata=testX2, type="response") )
pred.gam <- log( predict(model.gam, newdata=testX2, type="response") )


SSPE.glm <- sum((pred.glm - testset2$Y)^2); SSPE.glm            # 277.6839
SAPE.glm <- sum(abs(pred.glm - testset2$Y)); SAPE.glm           # 132.6071
SSPE.mars <- sum((pred.mars - testset2$Y)^2); SSPE.mars         # 275.8573
SAPE.mars <- sum(abs(pred.mars - testset2$Y)); SAPE.mars        # 134.7342
SSPE.gam <- sum((pred.gam - testset2$Y)^2); SSPE.gam            # 289.7461
SAPE.gam <- sum(abs(pred.gam - testset2$Y)); SAPE.gam           # 138.7856



































##> Imputation: create multiple dataset..once ------------------------------- ##
library(mice)
#df.imp <- mice(data=train_df11, m=10, maxit=1) # create 10 dataset (imputation x 10), only 1 variable with NA
##> 1st fitting: estimate Sh in the 10 imputed datasets, using GammaGLM with a single dumb intercept
# - with imputation
# Note: fmi is "fraction of missing information"
#head(train_df11)
#outcome <- with( data=df.imp, exp=glm(GenLiab~1, family=Gamma(link="inverse")) ) # intercepts of 10 different GLM
#                :from this data,  :apply this expression..
#pool(outcome)
#outcome <- with( data=df.imp, exp=glm(GenLiab~1, family = tweedie(var.power=1.78, link.power=0)) ) 
#                :from this data,  :apply this expression..
#pool(outcome)

# fmi(intercept):0.01586866 How many imputations? Rule of thumb: "FMI" x 100
#summary( pool(outcome), conf.int = T ) # (Intercept): 3.279482   --- after addressing NA

#:::::> WHYYY no-differ substantially from the complete case estimate?
# - 1.Because there are relatively few NA. 
# - 2.the estimate would require strong associations between NA and the variables used to impute (strong MAR?)
# - 3.We haven't used covariates...

################################################################################

#train_df11$RiskAversion = as.numeric(train_df11$RiskAversion)

summary(train_df11)
sub.df11 <- subset(train_df11, select=c("GenLiab", "LegalSyst", "RiskAversion")) # Y, X1, X2 
colnames(sub.df11) <- c('Y','X1','X2')
summary(sub.df11)
sub.df11$Y <- log(sub.df11$Y); 

X1 <- as.numeric(train_df11$LegalSyst)
X2 <- as.numeric(train_df11$RiskAversion)

sub.df11 <- data.frame(Y=Y, X1=X1, X2=X2); head(sub.df11)

# # - with imputation
predMat <- make.predictorMatrix(sub.df11); predMat # get default predictor matrix
# 
# # - Impute!!!: create multiple dataset..once
df.imp2 <- mice(sub.df11, m=10, predictorMatrix=predMat, seed=1981, printFlag=T)
# 
# ##> 2nd fitting
# outcome.glm <- with( data=df.imp2, exp=glm(Y~factor(X1) + X2, 
#                                            family=tweedie(var.power=1.78, link.power=0)) ) # result of 10 different GLM
# outcome.mars <- with( data=df.imp2, exp=glm(Y~factor(X1) + bs(X2, degree=4, df=6), 
#                                             family=tweedie(var.power=1.78, link.power=0)) ) # result of 10 different MARS
# outcome.gam <- with( data=df.imp2, exp=gam(Y~factor(X1) + s(X2, k=10, sp=0.001), 
#                                            family=Tweedie(1.78, power(0)), method="REML") ) # result of 10 different GAM
# summary( pool(outcome.glm), conf.int = T )
# summary( pool(outcome.mars), conf.int = T )
# summary( pool(outcome.gam), conf.int = T )

# OK....so?
##> re-run with more imputations? if so, how many?
# - we will re-impute with M=15~25 to get the Monte-Carlo error down further.
densityplot(df.imp2) # look at distribution of imputed and observed values
# - red: from each imputed dataset
# - blue: from the complete case?
#
#
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
##> Diagnostics on the NA treatment
# let's see convergence of mice algorithm (more "maxit")
convImps <- mice(sub.df11, m=25, predictorMatrix=predMat, printFlag=F, seed=8232446, maxit=50)
plot(convImps)
# x-axis: iteration 
# y-axis: mean 
# color: imputation trial (no.of new dataset)
# what we are looking to see is that the plots show essentially random variation, around a common average, with
# no systematic trending upwards or downwards. We also want to see the lines for the different imputation process
# overlapping with each other. 
densityplot(convImps)


##> Final fitting...most important.
# fit.glm2 <- with( data=convImps, exp=glm(exp(Y)~factor(X1) + X2, 
#                                            family=tweedie(var.power=1.78, link.power=0.1)) ) 
# fit.mars2 <- with( data=convImps, exp=glm(exp(Y)~factor(X1) + bs(X2, degree=4, df=6), 
#                                             family=tweedie(var.power=1.78, link.power=0.1)) )
# fit.gam2 <- with( data=convImps, exp=gam(exp(Y)~factor(X1) + s(X2, k=10, sp=0.001), 
#                                            family=Tweedie(1.78, power(0)), method="REML") ) 

fit.glm2 <- with( data=convImps, exp=glm(exp(Y)~factor(X1) + as.vector(X2), family=Gamma) ) 
fit.mars2 <- with( data=convImps, exp=glm(exp(Y)~factor(X1) + bs(as.vector(X2), degree=4, df=6), family=Gamma) )
fit.gam2 <- with( data=convImps, exp=gam(exp(Y)~factor(X1) + s(as.vector(X2), k=10, sp=0.001), family=Gamma) ) 

p1 <- summary( pool(fit.glm2), conf.int = T );p1
p2 <- summary( pool(fit.mars2), conf.int = T );p2
p3 <- summary( pool(fit.gam2), conf.int = T );p3
# compare "point estimates" (of parameters) between Complete Case model and MI
cbind(coef(model.glm),coef(model.mars),coef(model.gam), p1[,2], p2[,2], p3[,2]) # different  alot?

beta.glm2 = summary(pool(fit.glm2))$estimate
beta.mars2 = summary(pool(fit.mars2))$estimate
beta.gam2 = summary(pool(fit.gam2))$estimate

pred.glm2 <- exp( pXmat%*%beta.glm2 ); pred.glm2

pred.mars2 <- pXmat%*%beta.mars2; pred.mars2
pred.gam2 <- pXmat%*%beta.gam2; pred.gam2



###### Out-of-sample prediction #####
# First, Build a covariate feed with a test set.
# test_df11 <- test_df11 %>% drop_na(RiskAversion) # remove all NA in the testset
# test_df11$RiskAversion = as.numeric(test_df11$RiskAversion) # extra refinement
# test_df11$LegalSyst = factor(test_df11$LegalSyst)
# pXmat <- cbind( rep(1, nrow(test_df11)), test_df11$LegalSyst, test_df11$RiskAversion ) # matrix version
# test.data <- data.frame(X1=test_df11$LegalSyst, X2=test_df11$RiskAversion);test.data   # df version

# obtain predictions Q and prediction variance U
predm.glm <- lapply(getfit(fit.glm2), predict, se.fit = TRUE)#, data=test.data)
Q.glm <- sapply(predm.glm, `[[`, "fit")
U.glm <- sapply(predm.glm, `[[`, "se.fit")^2
dfcom.glm <- getfit(fit.glm2)[[1]]$df.null

predm.mars <- lapply(getfit(fit.mars2), predict, se.fit = TRUE)#, data=test.data)
Q.mars <- sapply(predm.mars, `[[`, "fit")
U.mars <- sapply(predm.mars, `[[`, "se.fit")^2
dfcom.mars <- getfit(fit.mars2)[[1]]$df.null

predm.gam <- lapply(getfit(fit.gam2), predict, se.fit = TRUE)#, data=test.data)
Q.gam <- sapply(predm.gam, `[[`, "fit")
U.gam <- sapply(predm.gam, `[[`, "se.fit")^2
dfcom.gam <- getfit(fit.gam2)[[1]]$df.null

## fitted value
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

1/pred.glm[,1]                                                        # 1/predict (this is real based on link)
1/pred.mars[,1]                                                       # 1/predict (this is real based on link)
1/pred.gam[,1]                                                        # 1/predict (this is real based on link)




