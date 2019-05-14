#ifndef LATENT_H_
#define LATENT_H_

// [[Rcpp::depends(BH)]]
#include <boost/math/special_functions/digamma.hpp>

#include <RcppParallel.h>
// [[Rcpp::depends(RcppParallel)]]

using namespace Rcpp;

#define MX_ITER 200
#define EPS 1.0
/*
* soft-max
*/
inline void softmax(std::vector<double> &x){
  std::vector<double> tmp(x.size(), 0);
  double mx = x[0];
  for(auto iter=x.begin()+1; iter!=x.end(); ++iter){
    if(*iter > mx) mx = *iter;
  }
  auto t = tmp.begin();
  double s = 0.0;
  for(auto iter=x.begin(); iter!= x.end(); ++iter){
    *t = std::exp(*iter - mx);
    s += *t;
    ++t;
  }
  
  auto xi = x.begin();
  for(auto iter=tmp.begin(); iter!=tmp.end(); ++iter){
    *xi = *iter / s;
    ++xi;
  }
}

/*
 * initialize posterior of topic probability
 */
inline void initTopicProb(
  std::vector<double> &digammaAlpha, 
  std::vector<double> &logLikelihoods,
  const RcppParallel::RVector<double> &alpha, 
  std::size_t nGene, 
  std::size_t nTopic, 
  std::mt19937 gen
){
  std::uniform_int_distribution<> dis(0, nGene-1);
  
  std::vector<double> qAlpha(nTopic);
  double p = (double)nGene / (double)nTopic;
  for(std::size_t t=0; t<nTopic; ++t) qAlpha[t] = p + alpha[t];
  
  std::vector<double> expCounts(nTopic);
  std::vector<double> tmp(nTopic);
  for(std::size_t i=0; i<nGene; i++){
    std::fill(expCounts.begin(), expCounts.end(), 0.0);
    for(std::size_t j=0; j<10; j++){
      int idx = dis(gen);
      auto ll = logLikelihoods.begin() + idx*nTopic;
      auto qa = qAlpha.begin();
      for(auto x = tmp.begin(); x!= tmp.end(); ++x){
        *x = *ll + boost::math::digamma(*qa);
        ++ll; ++qa;
      }
      softmax(tmp);
      auto n = expCounts.begin();
      for(auto x = tmp.begin(); x!= tmp.end(); ++x){
        *n += *x;
        ++n;
      }
    }
    auto n = expCounts.begin();
    auto a = alpha.begin();
    double s = 1.0 / (1.0 + (double)i/10000.0);
    for(auto qa = qAlpha.begin(); qa!=qAlpha.end(); ++qa){
      *qa = (1-s)*(*qa) + s*(*n * (double)nGene / 10.0 + *a);
      ++n; ++a;
    }
  }
  auto da = digammaAlpha.begin();
  for(auto qa = qAlpha.begin(); qa!=qAlpha.end(); ++qa){
    *da = boost::math::digamma(*qa);
    ++da;
  }
}

/*
* calculate latent probability
*/
inline void calcLatentProb(
  const RcppParallel::RMatrix<double>::Column &cnt, 
  double libSize, 
  const RcppParallel::RMatrix<double> &eta, 
  const RcppParallel::RMatrix<double> &topicTermDispersion, 
  const RcppParallel::RVector<double> &alpha, 
  const RcppParallel::RVector<double> &dispersionInv, 
  std::vector<double> &latentProb, 
  std::size_t nGene, 
  std::size_t nTopic, 
  std::mt19937 gen
){
  std::vector<double> logLikelihoods(nTopic*nGene, 0.0);
  std::vector<double> topicCount(nTopic, 0.0);
  std::vector<double> topicCountOld(nTopic, (double)nGene / (double)nTopic);
  std::vector<double> digammaAlpha(nTopic);
  std::vector<double> tmp(nTopic, 0.0);
  
  // constant term
  auto ll = logLikelihoods.begin();
  auto c = cnt.begin(); auto d = dispersionInv.begin();
  for(std::size_t g=0; g<nGene; ++g){
    for(std::size_t t=0; t<nTopic; ++t){
      *ll = eta(g,t) * *c
          - (*c + *d) * log(1 + libSize * topicTermDispersion(g, t));
      ++ll;
    }
    ++c; ++d;
  }
  
  // initialize 
  initTopicProb(digammaAlpha, logLikelihoods, alpha, nGene, nTopic, gen);
  
  // update latent variable
  for(unsigned int i=0; i<MX_ITER;++i){//50; ++i){
    auto z = latentProb.begin();
    auto ll = logLikelihoods.begin();
    for(std::size_t g=0; g<nGene; ++g){
      
      auto iter = tmp.begin();
      auto da = digammaAlpha.begin();
      for(std::size_t t=0; t<nTopic; ++t){
        *iter = *ll + *da;
        ++iter; ++ll; ++da;
      }
      
      softmax(tmp);
    
      iter = tmp.begin();
      auto tc = topicCount.begin();
      for(std::size_t t=0; t<nTopic; ++t){
        *(z + t*nGene) = *iter;
        *tc += *iter;
        ++iter; ++tc;
      }
      ++z;
    }
    auto da = digammaAlpha.begin(); auto a = alpha.begin(); auto tco = topicCountOld.begin();
    auto x = tmp.begin();
    for(auto tc = topicCount.begin(); tc != topicCount.end(); ++tc){
      *x = abs(*tc - *tco);
      *da = boost::math::digamma(*tc + *a);
      *tco = *tc; *tc = 0.0;
      ++da; ++a; ++ tco; ++x;
    }
    if(*(std::max_element(tmp.begin(), tmp.end())) < EPS) break;
  }
}

/*
* Gradients
*/
struct Grad{
  std::vector<double> eta;
  std::vector<double> dispersion;
  std::vector<double> dispersionLogTerm;
  std::vector<double> meanCnt;
  std::vector<double> topicCounts;
  double elbo;
  
  Grad(int nGene, int nTopic, int nSample)
    : eta(std::vector<double>(nGene*nTopic, 0.0)),
      dispersion(std::vector<double>(nGene, 0.0)),
      dispersionLogTerm(std::vector<double>(nGene, 0.0)), 
      meanCnt(std::vector<double>(nGene, 0.0)), 
      topicCounts(std::vector<double>(nSample*nTopic, 0.0)),
      elbo(0.0){}
};

/*
* Worker for calculating gradients
*/
struct CalcGrad : public RcppParallel::Worker{
  const RcppParallel::RMatrix<double> cnt;
  const RcppParallel::RVector<double> libSize;
  const RcppParallel::RMatrix<double> eta;
  const RcppParallel::RMatrix<double> etaExp;
  const RcppParallel::RVector<double> meanCnt;
  const RcppParallel::RVector<double> dispersionInv;
  const RcppParallel::RMatrix<double> topicTermDispersionSample;
  const RcppParallel::RMatrix<double> topicTermDispersionExp;
  const RcppParallel::RVector<double> alpha;
  
  Grad grad;
  
  // constructors
  CalcGrad(const NumericMatrix cnt,
           const NumericVector libSize,
           const NumericMatrix eta, 
           const NumericMatrix etaExp, 
           const NumericVector meanCnt,
           const NumericVector dispersionInv,
           const NumericMatrix topicTermDispersionSample, 
           const NumericMatrix topicTermDispersionExp, 
           const NumericVector alpha)
    : cnt(cnt), libSize(libSize),
      eta(eta), etaExp(etaExp), meanCnt(meanCnt), 
      dispersionInv(dispersionInv), 
      topicTermDispersionSample(topicTermDispersionSample), 
      topicTermDispersionExp(topicTermDispersionExp), 
      alpha(alpha), 
      grad(Grad(cnt.nrow(), eta.ncol(), cnt.ncol())){}
  
  CalcGrad(const CalcGrad& c, RcppParallel::Split)
    : cnt(c.cnt), libSize(c.libSize),
      eta(c.eta), etaExp(c.etaExp), meanCnt(c.meanCnt), 
      dispersionInv(c.dispersionInv), 
      topicTermDispersionSample(c.topicTermDispersionSample), 
      topicTermDispersionExp(c.topicTermDispersionExp), 
      alpha(c.alpha), 
      grad(Grad(cnt.nrow(), eta.ncol(), cnt.ncol())){}
  
  // accumulate
  void operator()(std::size_t begin, std::size_t end){
    // latent probabilities
    size_t nGene = eta.nrow();
    size_t nTopic = eta.ncol();
    size_t N = nGene * nTopic;
    
    std::random_device rd;
    
    for(std::size_t i=begin; i<end; i++){
      std::mt19937 gen(rd() * (i+1));
      std::vector<double> latentProb(N);
      calcLatentProb(cnt.column(i), libSize[i], 
                     etaExp, topicTermDispersionExp, 
                     alpha, dispersionInv, 
                     latentProb, nGene, nTopic, gen);
      // calculate gradients
      auto ge = grad.eta.begin();
      auto tc = grad.topicCounts.begin() + (nTopic*i);
      auto z = latentProb.begin();
      auto e = eta.begin();
      auto ttd = topicTermDispersionSample.begin();
      for(std::size_t j=0; j<nTopic; ++j){
        auto gd = grad.dispersion.begin();
        auto gdl = grad.dispersionLogTerm.begin();
        auto gm = grad.meanCnt.begin();
        auto c = cnt.column(i).begin();
        auto mc = meanCnt.begin();
        auto di = dispersionInv.begin();
        for(std::size_t k=0; k<nGene; ++k){
          // expected count
          *tc += *z;
          // gradients of eta
          *ge += *z * (*c - (*c + *di) / (1.0 + 1.0 / ((double)libSize[i] * *ttd)));
          // gradients of dispersion
          *gd -= (*c + *di) * *di * *z / (1.0 + 1.0 / ((double)libSize[i] * *ttd));
          *gdl += *z * log(1.0 + (double)libSize[i] * *ttd);
          // gradients of mean Count
          *gm += *c - (*c + *di) / (1.0 + 1.0 / ((double)libSize[i] * *ttd));

          // evidence lower bound
          grad.elbo += *z * (*c * *e - (*c + *di) * std::log(1.0 + (double)libSize[i] * *ttd));
          if(*z >0) grad.elbo -= *z * std::log(*z);
          
          ++ge; ++gd; ++gdl; ++gm;
          ++z; ++e; ++ttd; ++c; ++di; ++mc;
        }
        ++tc;
      }
    }
  }
  
  void join(const CalcGrad& rhs){
    auto rge = rhs.grad.eta.begin();
    for(auto ge = grad.eta.begin(); ge != grad.eta.end(); ++ge){
      *ge += *rge;
      ++rge;
    }
    auto rgd = rhs.grad.dispersion.begin();
    auto rgdl = rhs.grad.dispersionLogTerm.begin();
    auto gdl = grad.dispersionLogTerm.begin();
    auto rgm = rhs.grad.meanCnt.begin();
    auto gm = grad.meanCnt.begin();
    for(auto gd = grad.dispersion.begin(); gd != grad.dispersion.end(); ++gd){
      *gd += *rgd;
      *gdl += *rgdl;
      *gm += *rgm;
      ++rgd; ++rgdl; ++gdl;
      ++rgm; ++gm;
    }
    auto rtc = rhs.grad.topicCounts.begin();
    for(auto tc = grad.topicCounts.begin(); tc != grad.topicCounts.end(); ++tc){
      *tc += *rtc;
      ++rtc;
    }
    
    grad.elbo += rhs.grad.elbo;
  }
};

/*
* Worker for estimation of expression
*/
struct EstExp : public RcppParallel::Worker{
  const RcppParallel::RMatrix<double> cnt;
  const RcppParallel::RVector<double> libSize;
  const RcppParallel::RMatrix<double> eta;
  const RcppParallel::RVector<double> meanCnt;
  const RcppParallel::RVector<double> dispersionInv;
  const RcppParallel::RMatrix<double> topicTerm; 
  const RcppParallel::RMatrix<double> topicTermDispersion;
  const RcppParallel::RVector<double> alpha;

  RcppParallel::RMatrix<double> ex;
  
  // constructors
  EstExp(const NumericMatrix cnt,
         const NumericVector libSize,
         const NumericMatrix eta,
         const NumericVector meanCnt,
         const NumericVector dispersionInv,
         const NumericMatrix topicTerm, 
         const NumericMatrix topicTermDispersion, 
         const NumericVector alpha, 
         NumericMatrix ex
  )
    : cnt(cnt), libSize(libSize),
      eta(eta), meanCnt(meanCnt), dispersionInv(dispersionInv), 
      topicTerm(topicTerm), 
      topicTermDispersion(topicTermDispersion), 
      alpha(alpha), 
      ex(ex){}
  
  // accumulate
  void operator()(std::size_t begin, std::size_t end){
    // latent probabilities
    size_t nGene = eta.nrow();
    size_t nTopic = eta.ncol();
    size_t N = nGene * nTopic;
    
    std::random_device rd;
    
    for(std::size_t i=begin; i<end; i++){
      std::mt19937 gen(rd() * (i+1));
      std::vector<double> latentProb(N);
      calcLatentProb(cnt.column(i), libSize[i], 
                     eta, topicTermDispersion, 
                     alpha, dispersionInv, 
                     latentProb, nGene, nTopic, gen);
      
      /* estimate expression */
      auto z = latentProb.begin();
      auto t = topicTermDispersion.begin();
      for(std::size_t j=0; j<nTopic; ++j){
        auto e = ex.column(i).begin();
        auto c = cnt.column(i).begin();
        auto di = dispersionInv.begin();
        for(std::size_t k=0; k<nGene; ++k){
//          *e += *z * *t;
          *e += *z * (*c + *di) / (1.0 + 1.0 / libSize[i] / *t);
          ++z; ++e; ++t;
          ++c; ++di;
        }
      }
    }
  }
};

#endif