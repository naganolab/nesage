# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#' @useDynLib nesage
NULL

nesage <- function(dataMat, library, updateAlpha, modelList, iter) {
    .Call(`_nesage_nesage`, dataMat, library, updateAlpha, modelList, iter)
}

estimateExpression <- function(cnt, library, modelList) {
    .Call(`_nesage_estimateExpression`, cnt, library, modelList)
}

