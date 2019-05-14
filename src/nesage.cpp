//' @useDynLib nesage
#include "model.h"

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
List nesage(
    NumericMatrix dataMat,  // count data
    NumericVector library,
    bool updateAlpha, 
    List modelList, 
    int iter
){
  Rcpp::Rcout << "nesage\n";
  NesageModel* model = NesageModel::invert(modelList);
  NumericVector elbo(iter);

  arma::mat cnt = as<arma::mat>(dataMat);
  arma::vec librarySizes = as<arma::vec>(library);

  int iterOffset = modelList["iter"];
  
  /* update*/
  for(int i=0; i<iter; i++){
    Rcpp::Rcout << "iteration:\t" << iterOffset+i+1;
    
    elbo[i] = model->update(cnt, librarySizes, updateAlpha);

    Rcpp::Rcout << "\tapproximated ELBO: " << elbo[i] << "\n";
  }

  List modelListRet = model->convert();
  delete model;
  modelListRet.push_back(elbo, "elbo");

  return modelListRet;
}

// [[Rcpp::export]]
NumericMatrix estimateExpression(
    NumericMatrix cnt, 
    NumericVector library, 
    List modelList
){
  NesageModel* model = NesageModel::invert(modelList);
  NumericMatrix ex = model->estimateExpression(cnt, library);
  delete model;
  
  return ex;
}