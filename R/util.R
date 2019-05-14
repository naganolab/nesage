#' Train the Negative-binomial Sparse Additive Generative model.
#'
#' @param iter Number of iterations for training the model. 
#' @param cnt A matrix of read counts. Each row and column correspond to a gene and sample, respectively. 
#' @param library.size A vector of library sizes (i.e. total read count) of samples. If Null is given, 
#'     colSums(cnt) is automatically assigned. Default NULL.  
#' @param mean.cnt A vector of log transformed values of expected read count for genes. 
#'     log(rowSums(cnt) / sum(cnt)) is recommended. 
#' @param topic.n Number of topics. 
#' @param alpha.init Initial value of the parameters of the prior Dirichlet distribution for topic distributions. 
#'     Default 0.1.
#' @param update.alpha Boolean value. If True, the parameters of the prior Dirichlet distribution is updated. 
#'     Default FALSE
#' @param sgm.init Initial standard deviation of the posterior distribution of eta. Default 0.01. 
#' @param dispersion.init Initial value of a dispersion parameter. Default 1. 
#' @param rho Parameter for AdaDelta. Default 0.95
#' @param eps Parameter for AdaDelta. Default 1e-6
#' @param batch.size Minibatch size for stochastic gradient descent. If Null, number of columns of cnt is 
#'     automatically assigned. Default NULL. 
#' @param model If a list returned by this function rather than NULL is given, training restarts. Default NULL.  
#' @return A list containing the parameters of the trained model. 
#' @examples
#' \dontrun{
#' # Training without updating alpha. 
#' model <- nesage::train(
#'     iter = 1000, 
#'     cnt = cnt, 
#'     mean.cnt = log(rowSums(cnt)/sum(cnt)), 
#'     topic.n = 10, 
#'     update.alpha = FALSE
#' )
#' # Restart training and update alpha. 
#' model <- nesage::train(
#'     iter = 1000, 
#'     cnt = cnt, 
#'     model = model, 
#'     update.alpha = TRUE
#' )
#' }
#'
#' @export
train <- function(
  iter,
  cnt,
  library.size=NULL,
  mean.cnt,
  topic.n,
  alpha.init=0.1,
  update.alpha = FALSE, 
  sgm.init = 0.01,
  dispersion.init = 1, 
  rho = 0.95, 
  eps = 1e-6, 
  batch.size = NULL, 
  model = NULL
){
  if(is.null(ncol(cnt))) cnt <- matrix(cnt, length(cnt), 1)
  if(is.null(library.size)) library.size <- colSums(cnt)
  if(is.null(batch.size)) batch.size <- ncol(cnt)
  
  N <- ncol(cnt)
  G <- nrow(cnt)
  if(is.null(model)){
    model <- list(
      iter = 0,
      alpha = rep(alpha.init, topic.n),
      topic.counts = matrix(G / topic.n, N, topic.n),
      mu.eta = list(
        param = matrix(0, G, topic.n),
        mean.sq.grad = matrix(0, G, topic.n), 
        mean.sq.diff = matrix(0, G, topic.n)
      ), 
      sgm.eta = list(
        param = matrix(log(exp(sgm.init)-1), G, topic.n), 
        mean.sq.grad = matrix(0, G, topic.n), 
        mean.sq.diff = matrix(0, G, topic.n)
      ), 
      dispersion = list(
        param = rep(log(exp(dispersion.init)-1.0), G), 
        mean.sq.grad = rep(0, G),
        mean.sq.diff = rep(0, G)
      ),
      mean.cnt = mean.cnt,
      rho = rho, 
      eps = eps, 
      elbo = NULL
    )
  }
  elbo.old <- model$elbo
  model <- nesage(cnt, library.size, update.alpha, model, iter)
  model$elbo <- c(elbo.old, model$elbo)

  model
}

#' Train the model.
#' @param model A trained model, which is a list returned by train(). 
#' @param cnt A matrix of read counts. Each row and column correspond to a gene and sample, respectively. 
#' @param library.size A vector of library sizes (i.e. total read count) of samples. If Null is given, 
#'     colSums(cnt) is automatically assigned. Default NULL.  
#' @return A list, where the estimated count matrix is named as "cnt" and the log-transformed read-per-million value is 
#'    named as "rpm." 
#' @examples
#' \dontrun{
#' ex <- nesage::estimate.expression(model, cnt)
#' }
#' @export
estimate.expression <- function(
  model, cnt, library.size=NULL
){
  if(is.null(library.size)) library.size <- colSums(cnt)
  ex <- estimateExpression(cnt, library.size, model)
  rpm <- ex
  for(i in 1:ncol(ex)){
    rpm[,i] <- ex[,i] / sum(cnt[,i]) * 10^6
  }
  
  list(cnt = ex, rpm = rpm)
}

