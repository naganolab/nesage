# nesage: Negative-binomial sparse additive generative model for sequencing data
Provides functionality for inference of the topic model for overdispersed
count data such like RNA-Seq data and prediction based on the inferred model.

## Installation
```
devtools::install_github("naganolab/nesage")
```

## Example
```cnt``` is a matrix of read counts, where each row and column correspond to a gene and sample, respectively. 
```
# Training without updating alpha. 
model <- nesage::train(
    iter = 1000, 
    cnt = cnt, 
    mean.cnt = log(rowSums(cnt)/sum(cnt)), 
    topic.n = 10, 
    update.alpha = FALSE
)
# Restart training and update alpha. 
model <- nesage::train(
    iter = 1000, 
    cnt = cnt, 
    model = model, 
    update.alpha = TRUE
)
# Estimate gene expression
ex <- nesage::estimate.expression(model, cnt)
```
