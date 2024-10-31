#------------------------------------------------------------------------------#
################################ Analysis of NA ################################
#------------------------------------------------------------------------------#
library(CASdatasets)
library(rstan)
library(coda)
library(mvtnorm)
library(devtools)
library(dagitty)
library(shinystan)
library(rethinking)
library(mice)
library(mgcv)
library(splines) 
library(hglm)

#quantile(Insample.df$Ln.Cov._1)
#dim( subset(Insample.df, subset=Ln.Cov._1>3.9) ) # 1091 obv (top 25%)
#dim( subset(Insample.df, subset=is.na(Fire5_2)==T) ) #1849 obv 
#....the story is that: 
# among some expensive properties, there has been NA (mostly "0") on the protection level report...coz..heavy taxes 
# are imposed on that properties with low protection+high coverage ? 

Insample.df <- read.csv("C:/Users/kimm4/Desktop/WORKSPACE/2nd 3rd paper/DATASET/LGPF_MAR.csv", 
                     header=T, na.strings=c("."), stringsAsFactors=F)
head(Insample.df)
str(Insample.df)
table(Insample.df$Year)
# 2006 2007 2008 2009 2010 
# 1154 1138 1125 1112 1110 ???  : policies

# Is the NA in the binary covariate MAR? Yep!
summary(glm(is.na(Fire5_2)~Total_Losses, data=Insample.df, family="binomial"))  
summary(glm(is.na(Fire5_2)~Total_Losses+Ln.Cov._1, data=Insample.df, family="binomial"))  
summary(Insample.df$Fire5_2) # NA:1849

Sh = Insample.df$Total_Losses/1000000; Sh
Insample.df$Sh = Sh 
head(Insample.df)

df.sub <- subset(Insample.df, 
                 select=c(Sh, Fire5_2, Ln.Cov._1, Claim_Counts, PolicyNum)) # Y, X1, X2, X3, X4 
df.sub <- df.sub[df.sub$Sh>0, ] # to use Gamma GLM...sorry..
head(df.sub)



##> OK,...warming up!
# - w/o imputation?
outcome_0 <- glm(Sh~1, data=df.sub, family=Gamma(link="log")) # only for intercept! 
summary(outcome_0)                     # (Intercept): -2.8463   --- before addressing NA

##> Imputation: create multiple dataset..once
library(mice)
df.imp <- mice(data=df.sub, m=10, maxit=1) # create 10 dataset (imputation x 10), only 1 variable with NA
##> 1st fitting: estimate Sh in the 10 imputed datasets, using GammaGLM with a single dumb intercept
# - with imputation
# Note: fmi is "fraction of missing information"
outcome <- with( data=df.imp, exp=glm(Sh~1, family=Gamma(link="log")) ) # intercepts of 10 different GLM
#                :from this data,  :apply this expression..
pool(outcome)
# fmi(intercept):0.001191304 How many imputations? Rule of thumb: "FMI" x 100
summary( pool(outcome), conf.int = T ) # (Intercept): -2.846275   --- after addressing NA

#:::::> WHYYY no-differ substantially from the complete case estimate?
# - Because there are relatively few NA. 
# - the estimate would require strong associations between NA and the variables used to impute (strong MAR?)
# - We haven't used covariates...



##> OK, with covariates
# - w/o imputation?
# - This is a complete case model (ignoring NA)
model.glm <- glm(Sh~factor(Fire5_2) + Ln.Cov._1 + Claim_Counts, 
                 family=Gamma(link="log"), data=df.sub)
model.mars <- glm(Sh~factor(Fire5_2) + bs(Ln.Cov._1, degree=4, df=6) + Claim_Counts, 
                  family=Gamma(link="log"), data=df.sub) 
model.gam <- gam(Sh~factor(Fire5_2) + s(Ln.Cov._1, k=10, sp=0.001) + Claim_Counts, 
                 family=Gamma(link="log"), data=df.sub)
#model.RE <- hglm( fixed=Sh~factor(Fire5_2) + bs(Ln.Cov._1, degree=4, df=6) + Claim_Counts, 
#                  random = ~1|PolicyNum, 
#                  familly=Gamma(link="log"), data=df.sub )   

summary(model.glm)
summary(model.mars)
summary(model.gam)
#summary(model.RE)

# - with imputation
head(df.sub)
predMat <- make.predictorMatrix(df.sub); predMat # get default predictor matrix
# Impute!!!: create multiple dataset..once
df.imp2 <- mice(df.sub, m=10, predictorMatrix=predMat, seed=8232446, printFlag=T)
##> 2nd fitting
outcome.glm <- with( data=df.imp2, exp=glm(Sh~factor(Fire5_2)+Ln.Cov._1+Claim_Counts, 
                                      family=Gamma(link="log")) ) # result of 10 different GLM
outcome.mars <- with( data=df.imp2, exp=glm(Sh~factor(Fire5_2)+bs(Ln.Cov._1, degree=4, df=6)+Claim_Counts, 
                                          family=Gamma(link="log")) ) # result of 10 different MARS
outcome.gam <- with( data=df.imp2, exp=gam(Sh~factor(Fire5_2)+s(Ln.Cov._1, k=10, sp=0.001)+Claim_Counts, 
                                          family=Gamma(link="log")) ) # result of 10 different GAM
#outcome.RE <- with( data=df.imp2, exp=hglm( fixed=Sh~factor(Fire5_2)+bs(Ln.Cov._1, degree=4, df=6)+Claim_Counts, 
#                                            random = ~1|PolicyNum, familly=Gamma(link="log")) )   

summary( pool(outcome.glm), conf.int = T )
summary( pool(outcome.mars), conf.int = T )
summary( pool(outcome.gam), conf.int = T )


# OK....so?
##> re-run with more imputations? if so, how many?
# - we will re-impute with M=15~25 to get the Monte-Carlo error down further.
densityplot(df.imp2) # look at distribution of imputed and observed values
# - red: from each imputed dataset
# - blue: from the complete case?

# Q. should we expect the distribution of imputed values to look the same as the distribution of the observed values?
# - If data were MCAR, the distribution of the imputed and observed should be (approximately) the same.
# - Under MAR, the true distribution of the NA differs from the distribution of the observed values, due to the other
#   variables which affect NA and are associated with the variable in question. 

##> Diagnostics on the NA treatment
# let's see convergence of mice algorithm (more "maxit")
convImps <- mice(df.sub, m=25, predictorMatrix=predMat, printFlag=F, seed=8232446, maxit=50)
plot(convImps)
# x-axis: iteration 
# y-axis: mean 
# color: imputation trial (no.of new dataset)
# what we are looking to see is that the plots show essentially random variation, around a common average, with
# no systematic trending upwards or downwards. We also want to see the lines for the different imputation process
# overlapping with each other. 

##> Final fitting.....
outcome.glm2 <- with( data=convImps, exp=glm(Sh~factor(Fire5_2)+Ln.Cov._1+Claim_Counts, 
                                           family=Gamma(link="log")) ) # result of 25 different GLM
outcome.mars2 <- with( data=convImps, exp=glm(Sh~factor(Fire5_2)+bs(Ln.Cov._1, degree=4, df=6)+Claim_Counts, 
                                            family=Gamma(link="log")) ) # result of 25 different MARS
outcome.gam2 <- with( data=convImps, exp=gam(Sh~factor(Fire5_2)+s(Ln.Cov._1, k=10, sp=0.001)+Claim_Counts, 
                                           family=Gamma(link="log")) ) # result of 25 different GAM
p1 <- summary( pool(outcome.glm2), conf.int = T )
p2 <- summary( pool(outcome.mars2), conf.int = T )
p3 <- summary( pool(outcome.gam2), conf.int = T )


# compare "point estimates" (of parameters) between Complete Case model and MI
cbind(coef(model), p1[,2], p2[,2], p3[,2]) # different  alot?
# the coeff of "Fire5" gives the difference in the predicted value of Sh 
# between the category for which Fire5=1, and the category for which Fire5=0

# compare "standard errors" between Complete Case model and MI
cbind(diag(vcov(model))^0.5, p1[,3], p2[,3], p3[,3]) # different  alot?
# MI is able to obtain more precise estimates by extracting information from the incomplete cases.
# Multiple imputation standard errors are typically smaller than complete case standard errors. This is because MI
# is able to make use of the information that is observed in the incomplete cases to help estimate the parameters
# of interest. This observed information is completely discarded in a complete case analysis. 
# How much the standard errors reduce is very much situation dependent.

