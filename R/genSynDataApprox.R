library(bnlearn)
library(pcalg)
library(igraph)
library(randomForest)
library(ggplot2)
library(mixtools)
library(MASS)

genWeights <- function(k, s, d){
  set.seed(0)
    w <- c()
    for(i in 1:d){
        w <- cbind(w, as.matrix(s[1]*rnorm(k)))
    }
    for(si in s[2:length(s)]){
        w1 <- c()
        for(i in 1:d){
            w1 <- cbind(w1, as.matrix(si*rnorm(k)))
        }
        w <- rbind(w1, w)
    }
    w <- t(cbind(w, as.matrix(2*pi*runif(k*length(s)))))
    return(w)
}

genKernelFeatures <- function(v, w){
  v <- cbind(v, array(1, nrow(v)))
  f <- cos(v %*% w)
  f_avg <- apply(f, 2, mean)
  return(f_avg)
}

startup <- function(){
  org_path <- getwd()
  setwd('G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\ANM-master\\codeANM\\code\\startups')
  source('startupICML.R')
  source('startupBF.R')
  source('startupGDS.R')
  source('startupLINGAM.R')
  source('startupPC.R')
  source('startupScoreSEMIND.R')
  # source('startupGES.R')
  source("startupSID.R", chdir = TRUE)
  setwd('G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\ANM-master\\codeANM\\code\\experiments\\ANM')
  source("../../util/computeGaussKernel.R", chdir = TRUE)
  source("../../util_DAGs/computeCausOrder.R", chdir = TRUE)
  source("./experiment2parralel.R", chdir = TRUE)
  source("../../util_DAGs/randomB.R")
  source("../../util_DAGs/randomDAG.R")
  source("../../util_DAGs/sampleDataFromG.R")
  setwd(org_path)
}

startup()

genANMData <- function(linear=TRUE){
  dir <- getwd()
  setwd('G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\ANM-master\\codeANM\\code\\experiments\\ANM')
  source("../../util/computeGaussKernel.R", chdir = TRUE)
  source("../../util_DAGs/computeCausOrder.R", chdir = TRUE)
  source("./experiment2parralel.R", chdir = TRUE)
  source("../../util_DAGs/randomB.R")
  source("../../util_DAGs/randomDAG.R")
  source("../../util_DAGs/sampleDataFromG.R")
  source("../../startups/startupScoreSEMIND.R", chdir = TRUE)
  setwd(dir)
  test_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Test_data\\'
  
  nVec <- c(100,500)
  pVec <- c(10)
  numExp <- 10
  for(n in 1:numExp){
    for(s in nVec){
      for(p in pVec){
        p_con <-2/(p-1)
        trueG <- as.matrix(randomDAG(p,p_con))
        if(linear){
          trueB <- randomB(trueG,0.1,2,TRUE)
          X <- sampleDataFromG(s,trueG,funcType="linear", parsFuncType=list(B=trueB,kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariancesRandomExp", parsNoise=list(varMin=0.1,varMax=0.5,noiseExpVarMin=2,noiseExpVarMax=4))
          X <- as.matrix(X)
        }else{
          X <- sampleDataFromG(s,trueG,funcType="GAM", parsFuncType=list(kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariances", parsNoise=list(noiseExp=1,varMin=1,varMax=2))
        }
        write.table(trueG, paste(test_data_path, 'Target_P', p, '_S', s, '_N', n, '_NL.txt', sep=''), sep='\t', row.names=F)
        write.table(X, paste(test_data_path, 'Data_P', p, '_S', s, '_N', n, '_NL.txt', sep=''), sep='\t', row.names=F)
      }
    }
  }  
}

genTrainDataApproxKernel <- function(linear=TRUE, p=10, s=100, N=100){
    num_cg <- 50
    ss <- 100
    k <- 100
    N <- 5
    all_pa_residuals <- data.frame()
    all_no_pa_residuals <- data.frame()
    all_mix_pa_residuals <- data.frame()
    gamma <- c(0.15, 1.5, 15)
    w_pa <- genWeights(k, s=gamma, 20)
    w_res <- genWeights(k, s=gamma, 1)
    w_pa_res <- genWeights(k, s=gamma, 20)

    for(g in 1:num_cg){
        cat('Causal Graph:', g, '\n')
        p_con <- 2/(p-1)
        trainG <- as.matrix(randomDAG(p, p_con))
        true_dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(trainG, mode='directed'))
        child_nodes <- unique(which(trainG==1, arr.ind=T)[, 2])
        X <- list()
        cat('Generating data for G:', g, ' S:', ss, '\n')
        for(n in 1:N){
            if(linear){
                trueB <- randomB(trainG,0.1,2,TRUE)
                X[[n]] <- sampleDataFromG(ss,trainG,funcType="linear", parsFuncType=list(B=trueB,kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariancesRandomExp", parsNoise=list(varMin=0.1,varMax=0.5,noiseExpVarMin=2,noiseExpVarMax=4))
            }else{
                X[[n]] <- sampleDataFromG(ss,trainG,funcType="GAM", parsFuncType=list(kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariances", parsNoise=list(noiseExp=1,varMin=1,varMax=2))
            }
            X[[n]] <- scale(X[[n]])
        }
        cat('Getting residuals from pa...\n')
        for(c in child_nodes){
            cat('Child node:', c, '\n')
            pa_c <- as.matrix(which(trainG[, c]==1))
            cat('Pa c:', pa_c, '\n')
            pa_residuals <- matrix(0, nrow=N, ncol=3*k*length(gamma))
            cat('pa residuals dim:', dim(pa_residuals), '\n')
            if(linear){
                for(n in 1:N){
                    gp_fit <- train_linear(X[[n]][, pa_c], X[[n]][, c])
                    pa_residuals[n, ] <- c(genKernelFeatures(X[[n]][, pa_c, drop=FALSE], w_pa[c(1:length(pa_c), nrow(w_pa)), ]), genKernelFeatures(gp_fit$residuals, w_res), genKernelFeatures(cbind(X[[n]][, pa_c, drop=FALSE], gp_fit$residuals), w_pa_res[c(1:(length(pa_c)+1), nrow(w_pa_res)), ]))
                }
            }else{
                for(n in 1:N){
                    gam_fit <- train_gam(X[[n]][, pa_c], X[[n]][, c])
                    pa_residuals[n, ] <- c(genKernelFeatures(X[[n]][, pa_c, drop=FALSE], w_pa[c(1:length(pa_c), nrow(w_pa)), ]), genKernelFeatures(gam_fit$residuals, w_res), genKernelFeatures(cbind(X[[n]][, pa_c, drop=FALSE], gam_fit$residuals), w_pa_res[c(1:(length(pa_c)+1), nrow(w_pa_res)), ]))
                }
            }
            all_pa_residuals <- rbind(all_pa_residuals, pa_residuals)
      
            # Generate non-parent residuals
            cat('Getting residuals from non-pa\n')
            min_k <- 3
            min_max_d <- 5
            no_pa_c <- which(trainG[, c]==0)
            des_c <- as.numeric(descendants(as.bn(true_dag), as.character(c)))
            anc_c <- as.numeric(ancestors(as.bn(true_dag), as.character(c)))
            non_des_c <- anc_c[!anc_c %in% pa_c]
            no_pa_c <- no_pa_c[!no_pa_c %in% c]
            # no_pa_c <- no_pa_c[!no_pa_c %in% c(des_c)]
            cat('pa_c:', pa_c, ' no_pa_c:', no_pa_c, ' des_c:', des_c, ' anc_c:', anc_c, ' non_des_c:', non_des_c, '\n')
            if(length(no_pa_c) > 0){
                max_d <- 1:min(min_max_d, length(no_pa_c))
                no_pa_residuals <- matrix(0, nrow=N, ncol=3*k*length(gamma))
                # no_pa_residuals <- matrix(0, nrow=N, ncol=s)
                cat('non-residuals dim:', dim(no_pa_residuals), '\n')
                for(m in max_d){
                    cat('No Parent size:', m, '\n')
                    rand_pa <- matrix(0, nrow=min_k, ncol=m)
                    for(j in 1:min_k){
                        if(length(no_pa_c)==1){
                          rand_pa[j, ] <- no_pa_c 
                        }else{
                          rand_pa[j, ] <- sample(t(no_pa_c), m) 
                        }
                        cat('no_pa_j: ', rand_pa[j, ], '\n')
                        if(linear){
                            for(n in 1:N){
                                gp_fit <- train_linear(X[[n]][, rand_pa[j, ]], X[[n]][, c])
                                no_pa_residuals[n, ] <- c(genKernelFeatures(X[[n]][, rand_pa[j, ], drop=FALSE], w_pa[c(1:m, nrow(w_pa)), ]), genKernelFeatures(gp_fit$residuals, w_res), genKernelFeatures(cbind(X[[n]][, rand_pa[j, ], drop=FALSE], gp_fit$residuals), w_pa_res[c(1:(m+1), nrow(w_pa_res)), ]))
                            }
                        }else{
                            for(n in 1:N){
                                gam_fit <- train_gam(X[[n]][, rand_pa[j, ]], X[[n]][, c])
                                no_pa_residuals[n, ] <- c(genKernelFeatures(X[[n]][, rand_pa[j, ], drop=FALSE], w_pa[c(1:m, nrow(w_pa)), ]), genKernelFeatures(gam_fit$residuals, w_res), genKernelFeatures(cbind(X[[n]][, rand_pa[j, ], drop=FALSE], gam_fit$residuals), w_pa_res[c(1:(m+1), nrow(w_pa_res)), ]))
                            }
                        }
                        all_no_pa_residuals <- rbind(all_no_pa_residuals, no_pa_residuals)
                    }  
                }
                # Generate mix-parent residuals

                min_max_pa_d <- 4
                if(length(non_des_c) > 0){
                    cat('Getting residuals from mix-pa\n')
                    max_pa <- min(min_max_pa_d, length(non_des_c))
                    max_pa_d <- 1:max_pa
                    mix_pa_residuals <- matrix(0, nrow=N, ncol=3*k*length(gamma))
                    for(m in max_pa_d){
                        rand_mix_pa <- matrix(0, nrow=(min_k-1), ncol=(m+nrow(pa_c)))
                        for(j in 1:(min_k-1)){
                            if(length(non_des_c)==1){
                                rand_mix_pa[j, ] <- union(t(pa_c), non_des_c)
                            }else{
                                rand_mix_pa[j, ] <- union(t(pa_c), sample(t(non_des_c), m)) 
                            }
                            cat('mix_pa_k:', rand_mix_pa[j, ], '\n')
                            if(linear){
                                for(n in 1:N){
                                    gp_fit <- train_linear(X[[n]][, rand_mix_pa[j, ]], X[[n]][, c])
                                    mix_pa_residuals[n, ] <- c(genKernelFeatures(X[[n]][, rand_mix_pa[j, ], drop=FALSE], w_pa[c(1:(m+nrow(pa_c)), nrow(w_pa)), ]), genKernelFeatures(gp_fit$residuals, w_res), genKernelFeatures(cbind(X[[n]][, rand_mix_pa[j, ], drop=FALSE], gp_fit$residuals), w_pa_res[c(1:(m+nrow(pa_c)+1), nrow(w_pa_res)), ]))
                                }
                            }else{
                                for(n in 1:N){
                                    gam_fit <- train_gam(X[[n]][, rand_mix_pa[j, ]], X[[n]][, c])
                                    mix_pa_residuals[n, ] <- c(genKernelFeatures(X[[n]][, rand_mix_pa[j, ], drop=FALSE], w_pa[c(1:(m+nrow(pa_c)), nrow(w_pa)), ]), genKernelFeatures(gam_fit$residuals, w_res), genKernelFeatures(cbind(X[[n]][, rand_mix_pa[j, ], drop=FALSE], gam_fit$residuals), w_pa_res[c(1:(m+nrow(pa_c)+1), nrow(w_pa_res)), ]))                                }
                            }
                            all_mix_pa_residuals <- rbind(all_mix_pa_residuals, mix_pa_residuals)
                        }
                    }
                }
            }
        }            
    }    
}

genTestDataApproxKernel <- function(linear=TRUE){
  nVec <- c(100, 500)
  pVec <- c(5, 10)
  num_test_cg <- 10
  gamma <- c(0.15, 1.5, 15)
  k <- 100
  X_test <- list()
  testG <- list()
  all_test_features <- list()
  if(validation_flag){
    test_data_path <- '//home//mandar//Independent Study//Third Chapter//Test_data//'
    numExp <- 1 
  }
  else{
    test_data_path <- '//home//mandar//Independent Study//Third Chapter//Test_data//'
    numExp <- 10 
  }
  for(p in pVec){
    for(s in nVec){
      if(linear){
        new_dir <- paste(test_data_path, 'P', p, '_S', s, '_sparse', sep='') 
      }else{
        new_dir <- paste(test_data_path, 'P', p, '_S', s, '_sparse', sep='')
      }
      dir.create(new_dir)
      setwd(new_dir)
      for(n in 1:num_test_cg){
        all_residuals <- data.frame()
        p_con <- 2/(p-1)
        testG[[n]] <- as.matrix(randomDAG(p,p_con))
        if(linear){
          trueB <- randomB(testG[[n]],0.1,2,TRUE)
          X_test[[n]]<- sampleDataFromG(ss,testG[[n]],funcType="linear", parsFuncType=list(B=trueB,kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariancesRandomExp", parsNoise=list(varMin=0.1,varMax=0.5,noiseExpVarMin=2,noiseExpVarMax=4))
          X_test[[n]] <- as.matrix(X_test[[n]])
        }else{
          X_test[[n]] <- sampleDataFromG(ss,testG[[n]],funcType="GAM", parsFuncType=list(kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariances", parsNoise=list(noiseExp=1,varMin=1,varMax=2))
        }
        X_test[[n]] <- scale(X_test[[n]])
        # write.table(testG[[n]], paste('Target_P', p, '_S', s, '_N', n, '.txt', sep=''), sep='\t', row.names=F)
        # write.table(X_test[[n]], paste('Data_P', p, '_S', s, '_N', n, '.txt', sep=''), sep='\t', row.names=F)
        if(p<=10){
          min_pa <- 5
        }else{
          min_pa <- 3
        }
        for(i in 1:p){
          pa_i <- seq(p)[-i]
          all_pa_subsets <- list()
          pred_data <- list()
          for(j in 1:min(min_pa, length(pa_i))){
            subsets <- combn(pa_i, j)
            cat('subsets col:', ncol(subsets), '\n')
            residuals <- matrix(0, nrow=ncol(subsets), ncol=3*k*length(gamma))
            for(c in 1:ncol(subsets)){
              if(linear){
                fit <- train_linear(X=X_test[[n]][, subsets[, c]], y=X_test[[n]][, i]) 
              }else{
                fit <- train_gam(X=X_test[[n]][, subsets[, c]], y=X_test[[n]][, i]) 
              }
              f1 <- genKernelFeatures(X_test[[n]][, subsets[, c], drop=FALSE], w_pa[c(1:nrow(subsets), nrow(w_pa)), ])
              f2 <- genKernelFeatures(fit$residuals, w_res)
              f3 <- genKernelFeatures(cbind(X_test[[n]][, subsets[, c], drop=FALSE], fit$residuals), w_pa_res[c(1:(nrow(subsets)+1), nrow(w_pa_res)), ])
              residuals[c, ] <- c(f1, f2, f3)
            }
            all_residuals <- rbind(all_residuals, residuals)
          }
        }
        all_test_features[[n]] <- all_residuals
        # write.table(all_residuals, paste('X_residuals_N', n, '.txt', sep=''), sep='\t', row.names=F)
      }
    }
  }  
}

genTrainData <- function(linear=TRUE){
  num_cg <- 5
  nVec <- c(100)
  pVec <- c(10)
  N <- 100
  linear <- TRUE
  train_data_path <- '//home//mandar//Independent Study//Third Chapter//Train_data//'
    for(p in pVec){
        for(s in nVec){
            all_pa_residuals <- data.frame()
            all_no_pa_residuals <- data.frame()
            all_mix_pa_residuals <- data.frame()
            for(g in 1:num_cg){
                cat('Causal Graph:', g, '\n')
                p_con <- 2/(p-1)
                trueG <- as.matrix(randomDAG(p, p_con))
                true_dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(trueG, mode='directed'))
                child_nodes <- unique(which(trueG==1, arr.ind=T)[, 2])
                X <- list()
                cat('Generating data for G:', g, ' S:', ss, '\n')
                for(n in 1:N){
                    if(linear){
                        trueB <- randomB(trueG,0.1,2,TRUE)
                        X[[n]] <- sampleDataFromG(ss,trueG,funcType="linear", parsFuncType=list(B=trueB,kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariancesRandomExp", parsNoise=list(varMin=0.1,varMax=0.5,noiseExpVarMin=2,noiseExpVarMax=4))
                    }else{
                        X[[n]] <- sampleDataFromG(ss,trueG,funcType="GAM", parsFuncType=list(kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariances", parsNoise=list(noiseExp=1,varMin=1,varMax=2))
                    }
                }
                X[[n]] <- scale(X[[n]])
                cat('Getting residuals from pa...\n')
                for(c in child_nodes){
                    cat('Child node:', c, '\n')
                    pa_c <- as.matrix(which(trueG[, c]==1))
                    cat('Pa c:', pa_c, '\n')
                    pa_residuals <- matrix(0, nrow=N, ncol=3*ss)
                    cat('pa residuals dim:', dim(pa_residuals), '\n')
                    xnorm <- as.matrix(dist(X[[n]][, pa_c],method="euclidean",diag=TRUE,upper=TRUE))
                    xnorm <- xnorm^2
                    ynorm <- as.matrix(dist(gp_fit$residuals,method="euclidean",diag=TRUE,upper=TRUE))
                    ynorm <- ynorm^2
                    xynorm <- as.matrix(dist(cbind(X[[n]][, pa_c], gp_fit$residuals),method="euclidean",diag=TRUE,upper=TRUE))
                    xynorm <- xynorm^2
                    xhilf <- xnorm
                    yhilf <- ynorm
                    xyhilf <- xynorm
                    sigmax <- sqrt(0.5*median(xhilf[lower.tri(xhilf,diag=FALSE)]))
                    sigmay <- sqrt(0.5*median(yhilf[lower.tri(yhilf,diag=FALSE)]))
                    sigmaxy <- sqrt(0.5*median(xyhilf[lower.tri(xyhilf,diag=FALSE)]))
                    if(linear){
                        for(n in 1:N){                          
                          gp_fit <- train_linear(X[[n]][, pa_c], X[[n]][, c])
                          f1 <- kernelMatrix(rbfdot(sigma=1/(2*sigmax^2)), X[[n]][, pa_c])
                          f2 <- kernelMatrix(rbfdot(sigma=1/(2*sigmay^2)), gp_fit$residuals)
                          f3 <- kernelMatrix(rbfdot(sigma=1/(2*sigmaxy^2)), cbind(X[[n]][, pa_c], gp_fit$residuals))
                          pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                        }
                    }else{
                        for(n in 1:N){
                            gam_fit <- train_gam(X[[n]][, pa_c], X[[n]][, c])
                            f1 <- kernelMatrix(rbfdot(sigma=1/(2*sigmax^2)), X[[n]][, pa_c])
                            f2 <- kernelMatrix(rbfdot(sigma=1/(2*sigmay^2)), gam_fit$residuals)
                            f3 <- kernelMatrix(rbfdot(sigma=1/(2*sigmaxy^2)), cbind(X[[n]][, pa_c], gam_fit$residuals))
                            pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                        }
                    }
                    all_pa_residuals <- rbind(all_pa_residuals, pa_residuals)
              
                    # Generate non-parent residuals
                    cat('Getting residuals from non-pa\n')
                    k <- 3
                    min_max_d <- 6
                    no_pa_c <- which(trueG[, c]==0)
                    des_c <- as.numeric(descendants(as.bn(true_dag), as.character(c)))
                    anc_c <- as.numeric(ancestors(as.bn(true_dag), as.character(c)))
                    non_des_c <- anc_c[!anc_c %in% pa_c]
                    no_pa_c <- no_pa_c[!no_pa_c %in% c]
                    # no_pa_c <- no_pa_c[!no_pa_c %in% c(des_c)]
                    cat('pa_c:', pa_c, ' no_pa_c:', no_pa_c, ' non_des_c:', non_des_c, '\n')
                    if(length(no_pa_c) > 0){
                        max_d <- 1:min(min_max_d, length(no_pa_c))
                        no_pa_residuals <- matrix(0, nrow=N, ncol=3*ss)
                        # no_pa_residuals <- matrix(0, nrow=N, ncol=s)
                        cat('non-residuals dim:', dim(no_pa_residuals), '\n')
                        for(m in max_d){
                            cat('No Parent size:', m, '\n')
                            rand_pa <- matrix(0, nrow=k, ncol=m)
                            for(j in 1:k){
                                if(length(no_pa_c)==1){
                                  rand_pa[j, ] <- no_pa_c 
                                }else{
                                  rand_pa[j, ] <- sample(t(no_pa_c), m) 
                                }
                                cat('no_pa_k: ', rand_pa[j, ], '\n')
                                xnorm <- as.matrix(dist(X[[n]][, rand_pa[j, ]],method="euclidean",diag=TRUE,upper=TRUE))
                                xnorm <- xnorm^2
                                ynorm <- as.matrix(dist(gp_fit$residuals,method="euclidean",diag=TRUE,upper=TRUE))
                                ynorm <- ynorm^2
                                xynorm <- as.matrix(dist(cbind(X[[n]][, rand_pa[j, ]], gp_fit$residuals),method="euclidean",diag=TRUE,upper=TRUE))
                                xynorm <- xynorm^2
                                xhilf <- xnorm
                                yhilf <- ynorm
                                xyhilf <- xynorm
                                sigmax <- sqrt(0.5*median(xhilf[lower.tri(xhilf,diag=FALSE)]))
                                sigmay <- sqrt(0.5*median(yhilf[lower.tri(yhilf,diag=FALSE)]))
                                sigmaxy <- sqrt(0.5*median(xyhilf[lower.tri(xyhilf,diag=FALSE)]))

                                if(linear){
                                    for(n in 1:N){
                                        gp_fit <- train_linear(X[[n]][, rand_pa[j, ]], X[[n]][, c])                                      
                                        xnorm <- as.matrix(dist(X[[n]][, rand_pa[j, ]],method="euclidean",diag=TRUE,upper=TRUE))
                                        xnorm <- xnorm^2
                                        ynorm <- as.matrix(dist(gp_fit$residuals,method="euclidean",diag=TRUE,upper=TRUE))
                                        ynorm <- ynorm^2
                                        xynorm <- as.matrix(dist(cbind(X[[n]][, rand_pa[j, ]], gp_fit$residuals),method="euclidean",diag=TRUE,upper=TRUE))
                                        xynorm <- xynorm^2
                                        xhilf <- xnorm
                                        yhilf <- ynorm
                                        xyhilf <- xynorm
                                        sigmax <- sqrt(0.5*median(xhilf[lower.tri(xhilf,diag=FALSE)]))
                                        sigmay <- sqrt(0.5*median(yhilf[lower.tri(yhilf,diag=FALSE)]))
                                        sigmaxy <- sqrt(0.5*median(xyhilf[lower.tri(xyhilf,diag=FALSE)]))

                                        f1 <- kernelMatrix(rbfdot(sigma=1/(2*sigmax^2)), X[[n]][, rand_pa[j, ]])
                                        f2 <- kernelMatrix(rbfdot(sigma=1/(2*sigmay^2)), gp_fit$residuals)
                                        f3 <- kernelMatrix(rbfdot(sigma=1/(2*sigmaxy^2)), cbind(X[[n]][, rand_pa[j, ]], gp_fit$residuals))
                                        no_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                                    }
                                }else{
                                    for(n in 1:N){
                                        gam_fit <- train_gam(X[[n]][, rand_pa[j, ]], X[[n]][, c])
                                        xnorm <- as.matrix(dist(X[[n]][, rand_pa[j, ]],method="euclidean",diag=TRUE,upper=TRUE))
                                        xnorm <- xnorm^2
                                        ynorm <- as.matrix(dist(gam_fit$residuals,method="euclidean",diag=TRUE,upper=TRUE))
                                        ynorm <- ynorm^2
                                        xynorm <- as.matrix(dist(cbind(X[[n]][, rand_pa[j, ]], gam_fit$residuals),method="euclidean",diag=TRUE,upper=TRUE))
                                        xynorm <- xynorm^2
                                        xhilf <- xnorm
                                        yhilf <- ynorm
                                        xyhilf <- xynorm
                                        sigmax <- sqrt(0.5*median(xhilf[lower.tri(xhilf,diag=FALSE)]))
                                        sigmay <- sqrt(0.5*median(yhilf[lower.tri(yhilf,diag=FALSE)]))
                                        sigmaxy <- sqrt(0.5*median(xyhilf[lower.tri(xyhilf,diag=FALSE)]))

                                        f1 <- kernelMatrix(rbfdot(sigma=1/(2*sigmax^2)), X[[n]][, rand_pa[j, ]])
                                        f2 <- kernelMatrix(rbfdot(sigma=1/(2*sigmay^2)), gam_fit$residuals)
                                        f3 <- kernelMatrix(rbfdot(sigma=1/(2*sigmaxy^2)), cbind(X[[n]][, rand_pa[j, ]], gam_fit$residuals))
                                        no_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                                        # no_pa_residuals[n, ] <- apply(f2, 2, mean)
                                    }
                                }
                                all_no_pa_residuals <- rbind(all_no_pa_residuals, no_pa_residuals)
                            }  
                        }
                        # Generate mix-parent residuals
                        if(length(non_des_c) > 0){
                            min_max_pa_d <- 6
                            cat('Getting residuals from mix-pa\n')
                            max_pa <- min(min_max_pa_d, length(non_des_c))
                            max_pa_d <- 1:max_pa
                            mix_pa_residuals <- matrix(0, nrow=N, ncol=3*ss)
                            for(m in max_pa_d){
                                rand_mix_pa <- matrix(0, nrow=(k-1), ncol=(m+nrow(pa_c)))
                                for(j in 1:(k-1)){
                                    if(length(non_des_c)==1){
                                        rand_mix_pa[j, ] <- union(t(pa_c), non_des_c)
                                    }else{
                                        rand_mix_pa[j, ] <- union(t(pa_c), sample(t(non_des_c), m)) 
                                    }
                                    cat('mix_pa_k:', rand_mix_pa[j, ], '\n')
                                    if(linear){
                                        for(n in 1:N){
                                            gp_fit <- train_linear(X[[n]][, rand_mix_pa[j, ]], X[[n]][, c])
                                            xnorm <- as.matrix(dist(X[[n]][, rand_mix_pa[j, ]],method="euclidean",diag=TRUE,upper=TRUE))
                                            xnorm <- xnorm^2
                                            ynorm <- as.matrix(dist(gp_fit$residuals,method="euclidean",diag=TRUE,upper=TRUE))
                                            ynorm <- ynorm^2
                                            xynorm <- as.matrix(dist(cbind(X[[n]][, rand_mix_pa[j, ]], gp_fit$residuals),method="euclidean",diag=TRUE,upper=TRUE))
                                            xynorm <- xynorm^2
                                            xhilf <- xnorm
                                            yhilf <- ynorm
                                            xyhilf <- xynorm
                                            sigmax <- sqrt(0.5*median(xhilf[lower.tri(xhilf,diag=FALSE)]))
                                            sigmay <- sqrt(0.5*median(yhilf[lower.tri(yhilf,diag=FALSE)]))
                                            sigmaxy <- sqrt(0.5*median(xyhilf[lower.tri(xyhilf,diag=FALSE)]))

                                            f1 <- kernelMatrix(rbfdot(sigma=1/(2*sigmax^2)), X[[n]][, rand_mix_pa[j, ]])
                                            f2 <- kernelMatrix(rbfdot(sigma=1/(2*sigmay^2)), gp_fit$residuals)
                                            f3 <- kernelMatrix(rbfdot(sigma=1/(2*sigmaxy^2)), cbind(X[[n]][, rand_mix_pa[j, ]], gp_fit$residuals))
                                            mix_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                                        }
                                    }else{
                                        for(n in 1:N){
                                            gam_fit <- train_gam(X[[n]][, rand_mix_pa[j, ]], X[[n]][, c])
                                            xnorm <- as.matrix(dist(X[[n]][, rand_mix_pa[j, ]],method="euclidean",diag=TRUE,upper=TRUE))
                                            xnorm <- xnorm^2
                                            ynorm <- as.matrix(dist(gam_fit$residuals,method="euclidean",diag=TRUE,upper=TRUE))
                                            ynorm <- ynorm^2
                                            xynorm <- as.matrix(dist(cbind(X[[n]][, rand_mix_pa[j, ]], gam_fit$residuals),method="euclidean",diag=TRUE,upper=TRUE))
                                            xynorm <- xynorm^2
                                            xhilf <- xnorm
                                            yhilf <- ynorm
                                            xyhilf <- xynorm
                                            sigmax <- sqrt(0.5*median(xhilf[lower.tri(xhilf,diag=FALSE)]))
                                            sigmay <- sqrt(0.5*median(yhilf[lower.tri(yhilf,diag=FALSE)]))
                                            sigmaxy <- sqrt(0.5*median(xyhilf[lower.tri(xyhilf,diag=FALSE)]))

                                            f1 <- kernelMatrix(rbfdot(sigma=1/(2*sigmax^2)), X[[n]][, rand_mix_pa[j, ]])
                                            f2 <- kernelMatrix(rbfdot(sigma=1/(2*sigmay^2)), gam_fit$residuals)
                                            f3 <- kernelMatrix(rbfdot(sigma=1/(2*sigmaxy^2)), cbind(X[[n]][, rand_mix_pa[j, ]], gam_fit$residuals))
                                            mix_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                                            # mix_pa_residuals[n, ] <- apply(f2, 2, mean)
                                        }
                                    }
                                    all_mix_pa_residuals <- rbind(all_mix_pa_residuals, mix_pa_residuals)
                                }
                            }
                        }
                    }
                }            
            }
        }
    }
}

genTestData <- function(linear=TRUE){
  nVec <- c(100, 500)
  pVec <- c(5, 10)
  num_test_cg <- 50
  
  if(validation_flag){
    test_data_path <- '//home//mandar//Independent Study//Third Chapter//Test_data//'
    numExp <- 1 
  }
  else{
    test_data_path <- '//home//mandar//Independent Study//Third Chapter//Test_data//'
    numExp <- 10 
  }
  for(p in pVec){
    for(s in nVec){
      if(linear){
        new_dir <- paste(test_data_path, 'P', p, '_S', s, '_sparse', sep='') 
      }else{
        new_dir <- paste(test_data_path, 'P', p, '_S', s, '_sparse', sep='')
      }
      dir.create(new_dir)
      setwd(new_dir)
      for(n in 1:num_test_cg){
        all_residuals <- data.frame()
        p_con <- 2/(p-1)
        trueG <- as.matrix(randomDAG(p,p_con))
        if(linear){
          trueB <- randomB(trueG,0.1,2,TRUE)
          X <- sampleDataFromG(s,trueG,funcType="linear", parsFuncType=list(B=trueB,kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariancesRandomExp", parsNoise=list(varMin=0.1,varMax=0.5,noiseExpVarMin=2,noiseExpVarMax=4))
          X <- as.matrix(X)
          # X <- RCIT::normalize(X)
        }else{
          X <- sampleDataFromG(s,trueG,funcType="GAM", parsFuncType=list(kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariances", parsNoise=list(noiseExp=1,varMin=1,varMax=2))
          # X <- RCIT::normalize(X)
        }
        write.table(trueG, paste('Target_P', p, '_S', s, '_N', n, '.txt', sep=''), sep='\t', row.names=F)
        write.table(X, paste('Data_P', p, '_S', s, '_N', n, '.txt', sep=''), sep='\t', row.names=F)
        if(p < 10){
          min_pa <- p-1
        }else if(p==10){
          min_pa <- 5
        }else{
          min_pa <- 3
        }
        for(i in 1:p){
          sub_idx <- 1
          pa_i <- seq(p)[-i]
          all_pa_subsets <- list()
          pred_data <- list()
          for(j in 1:min(min_pa, length(pa_i))){
            subsets <- combn(pa_i, j)
            cat('subsets col:', ncol(subsets), '\n')
            residuals <- matrix(0, nrow=ncol(subsets), ncol=3*s)
            for(c in 1:ncol(subsets)){
              all_pa_subsets[[sub_idx]] <- subsets[, c]
              if(linear){
                fit <- train_linear(X=X[, subsets[, c]], y=X[, i]) 
              }else{
                fit <- train_gam(X=X[, subsets[, c]], y=X[, i]) 
              }
              xnorm <- as.matrix(dist(X[, subsets[, c]],method="euclidean",diag=TRUE,upper=TRUE))
              xnorm <- xnorm^2
              ynorm <- as.matrix(dist(fit$residuals,method="euclidean",diag=TRUE,upper=TRUE))
              ynorm <- ynorm^2
              xynorm <- as.matrix(dist(cbind(X[, subsets[, c]], fit$residuals),method="euclidean",diag=TRUE,upper=TRUE))
              xynorm <- xynorm^2
              xhilf <- xnorm
              yhilf <- ynorm
              xyhilf <- xynorm
              sigmax <- sqrt(0.5*median(xhilf[lower.tri(xhilf,diag=FALSE)]))
              sigmay <- sqrt(0.5*median(yhilf[lower.tri(yhilf,diag=FALSE)]))
              sigmaxy <- sqrt(0.5*median(xyhilf[lower.tri(xyhilf,diag=FALSE)]))
              f1 <- kernelMatrix(rbfdot(sigma=1/(2*sigmax^2)), X[, subsets[, c]])
              f2 <- kernelMatrix(rbfdot(sigma=1/(2*sigmay^2)), fit$residuals)
              f3 <- kernelMatrix(rbfdot(sigma=1/(2*sigmaxy^2)), cbind(X[, subsets[, c]], fit$residuals))
              residuals[c, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
              # residuals[c, ] <- apply(f2, 2, mean)
              # residuals[c, ] <- abs(fit$residuals)
            }
            all_residuals <- rbind(all_residuals, residuals)
            # write.table(t(subsets), paste('X_', i, '_subset_', j, '_P', p, '_S', s, '_N', n, '_pa.txt', sep=''), sep='\t', row.names=F)
          }
        }
        write.table(all_residuals, paste('X_residuals_N', n, '.txt', sep=''), sep='\t', row.names=F)
      }
    }
  }  
}

trainClassifier <- function(pa_residuals, mix_pa_residuals, no_pa_residuals){
  pa_train_idx <- sample(1:nrow(pa_residuals), nrow(pa_residuals))
  mix_pa_train_idx <- sample(1:nrow(mix_pa_residuals), nrow(pa_residuals))
  no_pa_train_idx <- sample(1:nrow(no_pa_residuals), 2*nrow(pa_residuals))
  
  train_data <- rbind(pa_residuals[pa_train_idx, ], mix_pa_residuals[mix_pa_train_idx, ], no_pa_residuals[no_pa_train_idx, ])
  labels <- c(rep(1, 2*length(pa_train_idx)), rep(0, length(no_pa_train_idx)))
  
  rf_fit <- randomForest(train_data, as.factor(labels), ntree = 501)
  return(rf_fit)
}

plotKernelstats <- function(pa_residuals, mix_pa_residuals, no_pa_residuals){
  t1 <- apply(pa_residuals, 2, mean)
  t2 <- apply(mix_pa_residuals, 2, mean)
  t3 <- apply(no_pa_residuals, 2, mean)
  t1_sd <- apply(pa_residuals, 2, sd)
  t2_sd <- apply(mix_pa_residuals, 2, sd)
  t3_sd <- apply(no_pa_residuals, 2, sd)
  p <- ncol(pa_residuals)
  stat_1 <- matrix(0, nrow=3*p, ncol=3)
  stat_1 <- data.frame(stat_1)
  stat_1[, 1] <- c(rep('True parents', p), rep('Mix parents', p), rep('Non-parents', p))
  stat_1[, 2] <- c(t1, t2, t3)
  stat_1[, 3] <- c(t1_sd, t2_sd, t3_sd)
  stat_1[, 1] <- as.factor(stat_1[, 1])
  stat_1[, 1] <- factor(stat_1[, 1], levels=c("True parents", "Mix parents", "Non-parents"))
  colnames(stat_1) <- c('Variable_set', 'mean', 'sd')
  # Save image with size 8x14 inches
  ggplot(stat_1, aes(x=1:(3*p), y=mean, colour=Variable_set)) + 
    geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), size=.3, width=.1) + theme_bw() + 
    theme(text=element_text(size=23), panel.grid.minor = element_blank(), legend.title = element_text(size=20),
    panel.grid.major = element_blank(), panel.background = element_blank(), legend.key=element_rect(color="grey", size=1, linetype="solid"), 
    legend.key.size = unit(1, "cm")) +
    geom_line(size=1) +
    geom_point(size=3) + 
    # guides(colour = guide_legend(override.aes = list(size = 3))) +
    scale_x_continuous(name="Features", limits=c(0, 3*p), breaks = seq(0, 3*p, 100)) +
    labs(color="Variable set", x="Features", y="Mean and sd values of m_k")
}