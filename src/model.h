#ifndef MODEL_H_
#define MODEL_H_

#define _USE_MATH_DEFINES

#include "adadelta.h"
#include "latent.h"

#include <cmath>

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;

/*
 * Definition and procedure of NeSAGe Model.
 */
class NesageModel{
protected:
  // Constructor
  NesageModel(List m);

  int iter;
  // parameters
  arma::rowvec alpha;
  arma::mat topicCounts;
  Optim<arma::mat>* muEta;
  Optim<arma::mat>* sgmEta;
  Optim<arma::vec>* dispersion;

  // config
  arma::vec meanCnt;
  double rho;
  double eps;

  double calcElbo(CalcGrad &calcGrad, arma::mat &cnt, arma::vec &librarySizes, arma::mat &mu, arma::mat &sgm, arma::rowvec &alpha, arma::vec &dispInv);
  void updateEta(CalcGrad &calcGrad, arma::mat &mu, arma::mat &sgm, arma::mat &omg);
  void updateDispersion(CalcGrad &calcGrad, arma::mat &cnt, unsigned int batchSize, arma::vec &disp, arma::vec &dispInv);
  void updateMeanCnt(CalcGrad &calcGrad);

public:
  // configuration
  int nGene;
  int nTopic;

  // returns list of variables
  static NesageModel* invert(List model);
  // convert list to NesageModel object
  List convert();
  // update posterior distribution of eta
  double update(arma::mat cnt, arma::vec librarySizes, bool updateAlpha);
  // estimate expression
  NumericMatrix estimateExpression(NumericMatrix cnt, NumericVector librarySizes);

  // destructor
  ~NesageModel();
};


#endif
