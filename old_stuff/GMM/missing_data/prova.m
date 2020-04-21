require(norm)
dat <- matrix(rnorm(1000),ncol=5) # original data
nvar <- ncol(dat)
n <- nrow(dat)
nmissing <- 50

dat_missing <- dat
dat_missing[sample(length(dat_missing),nmissing)] <- NA
is_na <- apply(dat_missing,2,is.na) # index if NAs

dat_impute <- dat_missing # data matrix for imputation

# set initial estimates to means from available data
for(i in 1:ncol(dat_impute)) dat_impute[is_na[,i],i] <- colMeans(dat_missing,na.rm = TRUE)[i]

# starting values for EM
means <- colMeans(dat_impute)
# NOTE: multiplying by (nrow-1)/(nrow) to get ML estimate
# For comparability with norm package output
sigma <- cov(dat_impute)*(nrow(dat_impute)-1)/nrow(dat_impute)

# get estimates from norm package for comparison
s <- prelim.norm(dat_missing)
e <- em.norm(s,criterion=1e-32,showits = FALSE)

# carry out EM over 100 iterations
for(j in 1:100)
{
  bias <- matrix(0,nvar,nvar)
  for(i in 1:n)
  {
    row_dat <- dat_missing[i,]
    avail <- which(!is.na(row_dat))
    if(length(avail)<nvar)
    {
      bias[-avail,-avail] <- bias[-avail,-avail] + sigma[-avail,-avail] - sigma[-avail,avail] %*% solve(sigma[avail,avail]) %*% sigma[avail,-avail]
      dat_impute[i,-avail] <- means[-avail] + (sigma[-avail,avail] %*% solve(sigma[avail,avail])) %*% (row_dat[avail]-means[avail])
    }
  }

  # get updated means and covariance matrix
  means <- colMeans(dat_impute)
  biased_sigma <- cov(dat_impute)*(n-1)/n

  # correct for bias in covariance matrix
  sigma <- biased_sigma + bias/n
}


# compare results to norm package output
# compare means
max(abs(getparam.norm(s,e)[[1]] - means))
# compare covariance matrix
max(abs(getparam.norm(s,e)[[2]] - sigma))
