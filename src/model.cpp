#include "model.h"

#include "adadelta.h"

#include <cmath>

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::plugins(cpp11)]]

using namespace Rcpp;

/*
 * constructor
 */
NesageModel::NesageModel(List m){
  this->iter = m["iter"];
  
  this->topicCounts = as<arma::mat>(m["topic.counts"]);
  
  List muEta = m["mu.eta"];
  this->muEta = new Optim<arma::mat>(
    as<arma::mat>(muEta["param"]),
    as<arma::mat>(muEta["mean.sq.grad"]),
    as<arma::mat>(muEta["mean.sq.diff"])
  );
  List sgmEta = m["sgm.eta"];
  this->sgmEta = new Optim<arma::mat>(
    as<arma::mat>(sgmEta["param"]),
    as<arma::mat>(sgmEta["mean.sq.grad"]), 
    as<arma::mat>(sgmEta["mean.sq.diff"])
  );
  List disp = m["dispersion"];
  this->dispersion = new Optim<arma::vec>(
    as<arma::vec>(disp["param"]),
    as<arma::vec>(disp["mean.sq.grad"]),
    as<arma::vec>(disp["mean.sq.diff"])
  );
  
  // config
  this->meanCnt = as<arma::vec>(m["mean.cnt"]);
  this->rho = m["rho"];
  this->eps = m["eps"];
  
  this->nGene = this->muEta->param.n_rows;
  this->nTopic = this->muEta->param.n_cols;
  
  this->alpha = as<arma::rowvec>(m["alpha"]);
}

/*
 * convert into a list
 */
List NesageModel::convert(){
  return List::create(
    Named("topic.counts") = wrap(this->topicCounts),
    Named("mu.eta") = this->muEta->invert(),
    Named("sgm.eta") = this->sgmEta->invert(), 
    Named("dispersion") = this->dispersion->invert(),
    Named("alpha") = wrap(this->alpha),
    Named("mean.cnt") = wrap(this->meanCnt),     
    Named("rho") = rho, 
    Named("eps") = eps, 
    Named("iter") = this->iter
  );
}

/*
 * invert the model object from list
 */
NesageModel* NesageModel::invert(List m){
  NesageModel* model = new NesageModel(m);
  return model;
}

double NesageModel::calcElbo(CalcGrad &calcGrad, 
                             arma::mat &cnt, arma::vec &librarySizes, 
                             arma::mat &mu, arma::mat &sgm, 
                             arma::rowvec &alpha, arma::vec &dispInv){
  unsigned int batchSize = cnt.n_cols;

  double elbo = calcGrad.grad.elbo;
  elbo -= (double)batchSize * arma::accu(arma::lgamma(dispInv));
  elbo += arma::accu(arma::lgamma(arma::sum(alpha,1)) - arma::lgamma(this->nGene + arma::sum(alpha,1)));
  elbo -= (double)batchSize * arma::accu(arma::lgamma(alpha));
  elbo += arma::accu(arma::log(sgm) - 0.5*arma::log(arma::pow(sgm, 2) + arma::pow(mu, 2)));
  
  for(unsigned int i=0; i<batchSize; ++i){
    elbo += arma::accu(arma::lgamma(this->topicCounts.row(i) + alpha));
    elbo += arma::accu(lgamma(cnt.col(i) + dispInv) + cnt.col(i) % this->meanCnt
                         + cnt.col(i) * log(librarySizes(i)) - cnt.col(i) % arma::log(dispInv));
  }
  
  return elbo;
}

/*
 * update parameters of posterior of eta
 */
void NesageModel::updateEta(CalcGrad &calcGrad, arma::mat &mu, arma::mat &sgm, arma::mat &omg){
  arma::mat gradEta = arma::reshape(arma::vec(calcGrad.grad.eta), this->nGene, this->nTopic);
  arma::mat sqMean = arma::pow(mu, 2) + arma::pow(sgm, 2);
  
  /* update mu */
  arma::mat gradMuEta = gradEta - mu / sqMean;
  this->muEta->update(gradMuEta, this->rho, this->eps);
  
  /* update sgm */
  arma::mat gradSgmEta = (gradEta % omg + 1.0/sgm - sgm / sqMean) / (1.0 + arma::exp(-this->sgmEta->param));
  this->sgmEta->update(gradSgmEta, this->rho, this->eps);
}

/* 
 * update dispersion
 */
void NesageModel::updateDispersion(
    CalcGrad &calcGrad, 
    arma::mat &cnt, unsigned int batchSize, 
    arma::vec &disp, 
    arma::vec &dispInv
){
  arma::vec sqDispInv = arma::pow(dispInv, 2);
  
  arma::mat gradDispersion = arma::vec(calcGrad.grad.dispersion);
  arma::vec logTerm = arma::vec(calcGrad.grad.dispersionLogTerm);
  
  arma::vec digammaTerm = dispInv;
  digammaTerm.for_each([batchSize] (arma::vec::elem_type& val){val = -((double)batchSize) * R::digamma(val);});
  
  for(unsigned int i=0; i<batchSize; ++i){
    gradDispersion += cnt.col(i) % dispInv;
    arma::vec tmpVec = dispInv + cnt.col(i);
    tmpVec.for_each([] (arma::vec::elem_type& val){val = R::digamma(val);});
    digammaTerm += tmpVec;
  }
  gradDispersion -= sqDispInv % (digammaTerm - logTerm); 
gradDispersion -= disp;
  gradDispersion = gradDispersion / (1.0 + arma::exp(-this->dispersion->param));
  this->dispersion->update(gradDispersion, this->rho, this->eps);
}

/*
 * update parameters
 */
double NesageModel::update(arma::mat cnt, arma::vec librarySizes, bool updateAlpha){
  this->iter++;
  
  unsigned int batchSize = cnt.n_cols;
  
  /* prepare */
  arma::mat omg = arma::randn<arma::mat>(this->nGene, this->nTopic);
  arma::mat mu = this->muEta->param;
  arma::mat sgm = arma::log(1.0 + arma::exp(this->sgmEta->param));
  arma::mat eta = sgm % omg + mu;
  
  arma::vec disp = arma::log(1.0 + arma::exp(this->dispersion->param));
  arma::vec dispInv = 1.0 / disp;
  
  arma::mat topicTerm = arma::exp(eta.each_col() + this->meanCnt);
  arma::mat topicTermDispersionSample = topicTerm.each_col() % disp;
  arma::mat topicTermDispersionExp = arma::exp(mu.each_col() + this->meanCnt).eval().each_col() % disp;
  
  /* latent variables */
  CalcGrad calcGrad(wrap(cnt), wrap(librarySizes),
                    wrap(eta), wrap(mu), 
                    wrap(this->meanCnt), wrap(dispInv),
                    wrap(topicTermDispersionSample), 
                    wrap(topicTermDispersionExp), 
                    wrap(this->alpha));
  RcppParallel::parallelReduce(0, batchSize, calcGrad);
  
  /* expected counts of topics */
  this->topicCounts = arma::reshape(arma::vec(calcGrad.grad.topicCounts), this->nTopic, batchSize).t().eval();
  
  /* evidence lower bound */
  double elbo = calcElbo(calcGrad, cnt, librarySizes, mu, sgm, this->alpha, dispInv);
  
  /* update eta */
  updateEta(calcGrad, mu, sgm, omg);

  /* update dispersion */
  updateDispersion(calcGrad, cnt, batchSize, disp, dispInv);
  
  /* update alpha */
  if(updateAlpha){
    arma::rowvec newAlpha = this->alpha;
    arma::mat digammaAlpha = this->topicCounts.each_row() + this->alpha;
    digammaAlpha.for_each([](arma::mat::elem_type& val){val = R::digamma(val);});
    arma::rowvec digammaAlpha2 = newAlpha;
    digammaAlpha2.for_each([](arma::rowvec::elem_type& val){val = R::digamma(val);});
    newAlpha %= (arma::sum(digammaAlpha, 0) - (double)batchSize * digammaAlpha2)
                / (R::digamma(this->nGene+arma::accu(newAlpha)) - R::digamma(arma::accu(newAlpha)))
                / (double)batchSize;
    newAlpha.elem(arma::find(newAlpha < 1e-10)).fill(1e-10);
    this->alpha = newAlpha;
  }
  
  return elbo;
}

NumericMatrix NesageModel::estimateExpression(NumericMatrix cnt, NumericVector librarySizes){
  /* prepare */
  arma::mat mu = this->muEta->param;
  arma::mat sgm = arma::log(1.0 + arma::exp(this->sgmEta->param));
  arma::vec disp = arma::log(1.0 + arma::exp(this->dispersion->param));
  arma::vec dispInv = 1.0 / disp;
  
  arma::mat topicTerm = arma::exp(mu.each_col() + this->meanCnt);
  arma::mat topicTermDispersion = topicTerm.each_col() % disp;

  NumericMatrix ex(cnt.nrow(), cnt.ncol());
  EstExp estExp(cnt, librarySizes,
                wrap(this->muEta->param), wrap(this->meanCnt), 
                wrap(dispInv), wrap(topicTerm), wrap(topicTermDispersion), 
                wrap(this->alpha.t()), 
                ex);
  RcppParallel::parallelFor(0, cnt.ncol(), estExp);
  
  return ex;
}

NesageModel::~NesageModel(){
  delete this->muEta;
  delete this->sgmEta;
  delete this->dispersion;
}

