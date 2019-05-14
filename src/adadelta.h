#ifndef VSGD_H_
#define VSGD_H_

#define _USE_MATH_DEFINES

#include <cmath>

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;

/*
* vSGD optimizer
*/
template <class T>
class Optim{
private:
  // moving averages
  T meanSqGrad;
  T meanSqDiff;

public:
  // Constructor
  Optim(T p, T msg, T msd);

  // returns list of variables
  List invert();
  // update via vSGD
  T update(T grad, double rho, double eps);

  T param;
};

template <class T>
Optim<T>::Optim(T p, T msg, T msd){
  this->param = p;
  this->meanSqGrad = msg;
  this->meanSqDiff = msd;
}

template <class T>
List Optim<T>::invert(){
  return List::create(
    Named("param") = wrap(this->param),
    Named("mean.sq.grad") = wrap(this->meanSqGrad),
    Named("mean.sq.diff") = wrap(this->meanSqDiff)
  );
}

template <class T>
T Optim<T>::update(T grad, double rho, double eps){
  this->meanSqGrad = rho * this->meanSqGrad + (1 - rho) * arma::pow(grad, 2);
  T step = arma::sqrt(this->meanSqDiff + eps) / arma::sqrt(this->meanSqGrad + eps);
  T diffParam = step % grad;
  this->meanSqDiff = rho * this->meanSqDiff + (1 - rho) * arma::pow(diffParam, 2);
  
  this->param += diffParam;
  
  return step;
}

#endif



