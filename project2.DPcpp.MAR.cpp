#include <RcppArmadillo.h>
#include <vector>
#include <cmath>
#include <Rmath.h>
#include <iostream>
#include <algorithm>
#include <math.h> 
#include <random>
//#include "utilFunctions.h"
  
  // [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;
using namespace Rcpp;
using namespace std;
using namespace stats;

void set_seed(unsigned int seed) {
  Rcpp::Environment base_env("package:base");
  Rcpp::Function set_seed_r = base_env["set.seed"];
  set_seed_r(seed);  
}

double sigmoid(double x) {
  return( ( 1 / (1 + exp(- x) ) ) );
}


uvec histC(uvec x) {
  int xmax = max(x);
  return(hist( x, xmax ));
}

vec removeelement(vec x, int n) {
  uvec n1(1);
  n1(0) = n;
  mat y(x.size(),1);
  y.col(0) = x;
  y.shed_rows(n1);
  return(y.col(0));
}

// [[Rcpp::export]]
vec appendelement(vec x, double a) {
  vec a1(1);
  a1(0) = a;
  mat y(x.size(),1);
  y.col(0) = x;
  y.insert_rows(x.size(),a1);
  return(y.col(0));
}

double dlogsknorm(double y, vec x, rowvec beta, double sig2, double xi) {
  double mu = dot(x,beta);
  double sig = sqrt(sig2);
  double z = (log(y)-mu)/sig;
  return(2/(y*sig)*normpdf(z)*normcdf(xi*z));
}

double dlogsknorm_log(double y, vec x, rowvec beta, double sig2, double xi) {
  double mu = dot(x,beta);
  double sig = sqrt(sig2);
  double z = (log(y)-mu)/sig;
  return(log(2)-log(y*sig) + log_normpdf(z) + log(normcdf(xi*z)));
}



// [[Rcpp::export]]
int rmultinomF(vec const& p) {
  vec csp = cumsum(p/sum(p));
  double rnd = runif(1)[0];
  int res = 0;
  int psize = p.size();
  
  for(int i = 0; i < psize; i++) {
    if(rnd>csp(i)) res = res+1;
  }
  
  return(res+1);
}

// [[Rcpp::export]]
int rmultinomFlog(vec const& logp) {
  double logptemp = logp(0);
  for(int i=1; i<logp.size(); i++) {
    if(logp(i)>logptemp) {
      logptemp = logp(i)+log(1+exp(logptemp-logp(i)));
    } else {
      logptemp = logptemp+log(1+exp(logp(i)-logptemp));
    }
  }
  
  vec p = exp(logp - logptemp);
  vec csp = cumsum(p);
  double rnd = runif(1)[0];
  int res = 0;
  int psize = p.size();

  for(int i = 0; i < psize; i++) {
    if(rnd>csp(i)) res = res+1;
  }

  return(res+1);
  // return(csp);
}


double rinvgamma(double a0, double b0) {
  double x = R::rgamma(a0, 1/b0);
  return(1/x);
}


// [[Rcpp::export]]
List clusterDP(vec Y, vec X1, vec X2, uvec cl_membership,
               vec piparam, vec muparam, vec tau2param, 
               mat beta_j, vec sig2_j, vec xi_j, mat betat_j, double alpha,
               vec f0x, vec f0y, double c0, double d0, double mu0, double e0, double gam0,
               double a0, double b0, vec beta0, mat SIG_b0, double nu0, vec betat0, mat SIG_bt0, double varinf) {
  int n = Y.size();
  //Rprintf("n=%d,",n);
  //uvec cluster_si;
  uvec nj;
  int j;
  int J; // total number of clusters
  vec x_i(3);
  vec x_i0(3);
  vec x_i1(3);
  int newclust;
  
  vec beta_old_j;
  double sig2_old_j;
  double xi_old_j;
  vec betat_old_j;
    
  vec beta_p;
  double sig2_p;
  double xi_p;
  vec betat_p;
  
  double p0log;
  double p1log;
  double numerator;
  double denominator;
  double ratio;
  vec compare(2);
  double U;
  double logLSNna;
  double logA;
  double logB;
  
  J = max(cl_membership);
  for(int i = 0; i < n; i++) {
    //Rprintf("i=%d,",i);
    //cluster_si = cl_membership;
    nj = histC(cl_membership); // counts of each cluster
    if(nj(cl_membership(i)-1)==1) {
      j = cl_membership(i);
      //Rprintf("j=%d,",j);
      for(int k = 0; k < n; k++) {
        if(cl_membership(k)>j) {
          cl_membership(k) = cl_membership(k)-1;
        }
      }
      uvec j1(1);
      j1(0) = j-1;
      //Rprintf("j1=%d,",j1(0));
      piparam = removeelement(piparam,j-1);
      //Rprintf("length(piparm)=%d,",piparam.size());
      muparam = removeelement(muparam,j-1);
      //Rprintf("length(muparm)=%d,",muparam.size());
      tau2param = removeelement(tau2param,j-1);
      //Rprintf("length(tau2parm)=%d,",tau2param.size());
      sig2_j = removeelement(sig2_j,j-1);
      //Rprintf("length(sig2_j)=%d,",sig2_j.size());
      beta_j.shed_rows(j1);
      //Rprintf("length(beta_j)=%d,",beta_j.size());
      betat_j.shed_rows(j1);
      xi_j = removeelement(xi_j,j-1);
      J = J-1;
    }
    
    cl_membership(i) = 0;
    nj = histC(cl_membership(find(cl_membership>0)));
    vec probs(nj.size()+1, fill::zeros); // vector of length nj.size()+1 
    vec log_probs(nj.size()+1, fill::zeros); // vector of length nj.size()+1 
    
    x_i(0) = 1;
    x_i(1) = X1(i);
    x_i(2) = X2(i);
    //Rprintf("x_i(0)=%d,x_i(1)=%d,x_i(2)=%.4f,",x_i(0),x_i(1),x_i(2));
    if(isnan(x_i(1))) {
      x_i0 = x_i;
      x_i1 = x_i;
      x_i0(1) = 0;
      x_i1(1) = 1;
      if(Y(i)==0) {
        for(int j=0; j<nj.size(); j++) {
          probs(j) = nj(j)/(n-1+alpha)*( (sigmoid( dot(x_i1,betat_j.row(j)) ))*piparam(j) + 
            (sigmoid( dot(x_i0,betat_j.row(j)) ))*(1 - piparam(j) ))* 
          normpdf(x_i(2), muparam(j), sqrt(tau2param(j)));
        } 
      } else {
        for(int j=0; j<nj.size(); j++) {
          logA = log(1-sigmoid( dot(x_i1,betat_j.row(j)))) + dlogsknorm_log(Y(i), x_i1, beta_j.row(j), sig2_j(j),xi_j(j) ) + log(piparam(j));
          logB = log(1-sigmoid( dot(x_i0,betat_j.row(j)))) + dlogsknorm_log(Y(i), x_i0, beta_j.row(j), sig2_j(j),xi_j(j) ) + log(1-piparam(j));
          if(logA > logB) {
            logLSNna = logA + log(1+exp(logB-logA));
          } else {
            logLSNna = logB + log(1+exp(logA-logB));
          }
          probs(j) = nj(j)/(n-1+alpha)*((1-sigmoid( dot(x_i1,betat_j.row(j))  ))*      
                                          dlogsknorm(Y(i), x_i1, beta_j.row(j), sig2_j(j),xi_j(j) )*
                                          piparam(j) + 
                                          (1-sigmoid( dot(x_i0,betat_j.row(j))  ))*      
                                          dlogsknorm(Y(i), x_i0, beta_j.row(j), sig2_j(j),xi_j(j) )*
                                          (1-piparam(j)))* 
                                          normpdf(x_i(2), muparam(j), sqrt(tau2param(j)));
          log_probs(j) = log(nj(j)) - log(n-1+alpha) + logLSNna + log_normpdf(x_i(2), muparam(j), sqrt(tau2param(j)));
        }
      }
    } else {
      if(Y(i)==0) { 
        for(int j=0; j<nj.size(); j++) {
          probs(j) = nj(j)/(n-1+alpha)*(sigmoid( dot(x_i,betat_j.row(j)) ))*
            (x_i(1)*piparam(j)+(1-x_i(1))*(1-piparam(j))) * 
            normpdf(x_i(2), muparam(j), sqrt(tau2param(j)));
        } 
      } else {
        for(int j=0; j<nj.size(); j++) {
          probs(j) = nj(j)/(n-1+alpha)*((1-sigmoid( dot(x_i,betat_j.row(j))  ))*      
            dlogsknorm(Y(i), x_i, beta_j.row(j), sig2_j(j),xi_j(j) ))*
            (x_i(1)*piparam(j)+(1-x_i(1))*(1-piparam(j))) * 
            normpdf(x_i(2), muparam(j), sqrt(tau2param(j)));
          log_probs(j) = log(nj(j))-log(n-1+alpha) + log(1-sigmoid( dot(x_i,betat_j.row(j)))) +
            dlogsknorm_log(Y(i), x_i, beta_j.row(j), sig2_j(j),xi_j(j) ) +
            (x_i(1)*log(piparam(j))+(1-x_i(1))*log(1-piparam(j))) +
            log_normpdf(x_i(2), muparam(j), sqrt(tau2param(j)));
          //Rprintf("lognj=%.8f,log(n-1+alpha)=%.8f,log(1-sigmoid( dot(x_i,betat_j.row(j))))=%.8f,dlogsklog=%.8f,dbinomlog=%.8f,lognorm=%.8f,"
          //          ,log(nj(j)),log(n-1+alpha),log(1-sigmoid( dot(x_i,betat_j.row(j)))),dlogsknorm_log(Y(i), x_i, beta_j.row(j), sig2_j(j),xi_j(j) ),
          //          (x_i(1)*log(piparam(j))+(1-x_i(1))*log(1-piparam(j))),log_normpdf(x_i(2), muparam(j), sqrt(tau2param(j))));
          //Rprintf("log_probs(j)=%.8f,",log_probs(j));
        }
      }
    }
    
    probs(nj.size()) = alpha/(n-1+alpha)*f0y(i)*f0x(i); 
    log_probs(nj.size()) = log(alpha)-log(n-1+alpha)+log(f0y(i))+log(f0x(i));
    // Rprintf("probs=");
    // for(int u = 0; u < probs.size(); u++) {
    //   Rprintf("%.8f,",probs(u));
    // }
    // Rprintf("log_probs=");
    // if(Y(i)>0) {
    //   for(int u = 0; u < probs.size(); u++) {
    //     Rprintf("%.8f,",log_probs(u));
    //   }
    // }
    if(Y(i)==0) {
      newclust = rmultinomF(probs);
    } else {
      newclust = rmultinomFlog(log_probs);
    }
    //Rprintf("newclust=%d,",newclust);
    cl_membership(i) = newclust;
    nj = histC(cl_membership);
    //Rprintf("nj.size=%d,",nj.size());
    
    if(nj(cl_membership(i)-1)==1) { // add new parameters for new (continuous) cluster
      //binary covariate
      //vec piparam_temp(piparam.size()+1);
      //piparam_temp(span(0,piparam.size()-1)) = piparam;
      //Rprintf("newcluster,");
      if(isnan(x_i(1))) {
        piparam = appendelement(piparam, R::rbeta(c0, d0));
      } else {
        piparam = appendelement(piparam, R::rbeta(c0+x_i(1), d0-x_i(1)+1));
      }
      //Rprintf("piparam.size=%d,",piparam.size());
      //vec muparam_temp(muparam.size()+1);
      //vec tau2param_temp(tau2param.size()+1);
      //muparam_temp(span(0,muparam.size()-1)) = muparam;
      //tau2param_temp(span(0,tau2param.size()-1)) = tau2param;
      
      //tau2param_temp(tau2param.size()) = rinvgamma(e0+1/2, gam0+1/2*(1/2*pow(x_i(2)-mu0,2)) );
      tau2param = appendelement(tau2param, rinvgamma(e0+1/2, gam0+1/2*(1/2*pow(x_i(2)-mu0,2)) ));
      // pow(3,4) = 3^4 
      //Rprintf("tau2param.size=%d,",tau2param.size());
      //muparam_temp(muparam.size()) = R::rnorm((x_i(2)+mu0)/2,sqrt(tau2param_temp(J)/2)); 
      muparam = appendelement(muparam, R::rnorm((x_i(2)+mu0)/2,sqrt(tau2param(J)/2)));
      //Rprintf("muparam.size=%d,",muparam.size());
      
      beta_old_j = mvnrnd(beta0, SIG_b0*varinf);
      sig2_old_j = rinvgamma(a0, b0);
      xi_old_j = R::rt(nu0); 
      betat_old_j = mvnrnd(betat0, SIG_bt0*varinf);
        
      beta_p = mvnrnd(beta0, SIG_b0*varinf);
      sig2_p = rinvgamma(a0, b0);
      xi_p = R::rt(nu0); 
      betat_p = mvnrnd(betat0, SIG_bt0*varinf);
      
      //Rprintf("sig2_old_j=%.4f,sig2_p=%.4f,",sig2_old_j,sig2_p);
      //Rprintf("x_i(1)=%.4f,",x_i(1));
      if(isnan(x_i(1))) {
        x_i0 = x_i;
        x_i1 = x_i;
        x_i0(1) = 0;
        x_i1(1) = 1;
        if (Y(i)>0) {
          p0log = 
            dlogsknorm_log(Y(i), x_i0, beta_old_j.t(),sig2_old_j, xi_old_j) + 
                             log(1-piparam(J)) + 
                             log( 1-sigmoid( dot(x_i0,betat_old_j) ) );
          
          p1log = 
            dlogsknorm_log(Y(i), x_i1, beta_old_j.t(),sig2_old_j, xi_old_j) + 
            log(piparam(J)) + 
            log( 1-sigmoid( dot(x_i1,betat_old_j) ) );
        } else {
          p0log = 
            log(1-piparam(J)) + 
            log( sigmoid( dot(x_i0,betat_old_j) ) );
          
          p1log = 
            log(piparam(J)) + 
            log( sigmoid( dot(x_i1,betat_old_j) ) );
        }
        x_i(1) = R::rbinom(1, 1/(1+exp(p0log-p1log))); //imputation
      }
      
      //Rprintf("Y(i)=%.4f,",Y(i));
      if(Y(i)>0){
        //Rprintf("Y(i)>0,");
        //Rprintf("dot(x_i,betat_p)=%.4f,",dot(x_i,betat_p));
        //Rprintf("logsk=%.4f,",dlogsknorm_log(Y(i), x_i, beta_p.t(), sig2_p, xi_p));
        numerator = log( 1-sigmoid(dot(x_i,betat_p)) ) + 
          dlogsknorm_log(Y(i), x_i, beta_p.t(), sig2_p, xi_p);
        //Rprintf("numerator=%.4f,",numerator);
        denominator = 
          log( 1-sigmoid(dot(x_i,betat_old_j)) ) + 
          dlogsknorm_log(Y(i), x_i, beta_old_j.t(), sig2_old_j, xi_old_j);
        //Rprintf("denominator=%.4f,",denominator);
      } else {
        numerator = 
          log( sigmoid(dot(x_i,betat_p)) ); 
        denominator = 
          log( sigmoid(dot(x_i,betat_old_j)) );  
      }
      //Rprintf("numerator=%.4f,denominator=%.4f,",numerator,denominator);
      // compute the ratio
      compare(0) = exp(numerator-denominator);
      compare(1) = 1;
      ratio = min(compare);
      //Rprintf("ratio=%.4f,",ratio);
        
        U = randu();
        if(U < ratio) {
          beta_j.insert_rows(beta_j.n_rows, beta_p.t());
          sig2_j = appendelement(sig2_j,sig2_p);
          xi_j = appendelement(xi_j,xi_p);
          betat_j.insert_rows(betat_j.n_rows, betat_p.t());
        } else {
          beta_j.insert_rows(beta_j.n_rows, beta_old_j.t());
          sig2_j = appendelement(sig2_j,sig2_old_j);
          xi_j = appendelement(xi_j,xi_old_j);
          betat_j.insert_rows(betat_j.n_rows, betat_old_j.t());
        }
        
    }
    
    
    // cl_membership = cluster_si;
  }
  
  return List::create(_["cl_membership"]=cl_membership,
                      _["piparam"]=piparam,
                      _["muparam"]=muparam,
                      _["tau2param"]=tau2param,
                      _["beta_j"]=beta_j,
                      _["sig2_j"]=sig2_j,
                      _["xi_j"]=xi_j,
                      _["betat_j"]=betat_j);
}