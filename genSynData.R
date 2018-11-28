library(mgcv)
library(MASS)
library(bnlearn)
library(pcalg)
library(RCIT)
library(igraph)
library(randomForest)
library(cvTools)
library(ggplot2)

startup <- function(){
  org_path <- getwd()
  setwd('G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\ANM-master\\codeANM\\code\\startups')
  source('startupICML.R')
  source('startupBF.R')
  source('startupGDS.R')
  source('startupLINGAM.R')
  source('startupPC.R')
  source('startupScoreSEMIND.R')
  source('startupGES.R')
  source("startupSID.R", chdir = TRUE)
  setwd('G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\ANM-master\\codeANM\\code\\experiments\\ANM')
  source("../../util/computeGaussKernel.R", chdir = TRUE)
  source("../../util_DAGs/computeCausOrder.R", chdir = TRUE)
  source("./experiment2parralel.R", chdir = TRUE)
  source("../../util_DAGs/randomB.R")
  source("../../util_DAGs/randomDAG.R")
  source("../../util_DAGs/sampleDataFromG.R")
  source("../../startups/startupScoreSEMIND.R", chdir = TRUE)
  setwd(org_path)
}

startup()

genTrainData <- function(linear=TRUE){
  num_cg <- 5
  nVec <- c(100)
  pVec <- c(10)
  N <- 100
  linear <- TRUE
  p <- 10
  s <- 100
  sig <- 1
  train_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Train_data_mix_pa\\'
  for(p in pVec){
    for(s in nVec){
      all_pa_residuals <- data.frame()
      all_no_pa_residuals <- data.frame()
      all_mix_pa_residuals <- data.frame()
      for(g in 1:num_cg){
        cat('Causal Graph:', g, '\n')
 
        p_con <- 2*2/(p-1)
        trueG <- as.matrix(randomDAG(p, p_con))
        true_dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(trueG, mode="directed"))
        child_nodes <- unique(which(trueG==1, arr.ind=T)[, 2])
        X <- list()
        cat('Generating data for G:', g, ' S:', s, '\n')
        for(n in 1:N){
          if(linear){
            trueB <- randomB(trueG,0.1,2,TRUE)
            X[[n]] <- sampleDataFromG(s,trueG,funcType="linear", parsFuncType=list(B=trueB,kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariancesRandomExp", parsNoise=list(varMin=0.1,varMax=0.5,noiseExpVarMin=2,noiseExpVarMax=4))
          }else{
            X[[n]] <- sampleDataFromG(s,trueG,funcType="GAM", parsFuncType=list(kap=1,sigmax=1,sigmay=1,output=FALSE), noiseType="normalRandomVariances", parsNoise=list(noiseExp=1,varMin=1,varMax=2))
          }
        }
        cat('Getting residuals from pa...\n')
        for(c in child_nodes){
          cat('Child node:', c, '\n')
          pa_c <- as.matrix(which(trueG[, c]==1))
          cat('Pa c:', pa_c, '\n')
  
          pa_residuals <- matrix(0, nrow=N, ncol=3*s)
          # pa_residuals <- matrix(0, nrow=N, ncol=s)
          cat('pa residuals dim:', dim(pa_residuals), '\n')
          if(linear){
            for(n in 1:N){
              gp_fit <- train_linear(X[[n]][, pa_c], X[[n]][, c])
              f1 <- kernelMatrix(rbfdot(sigma=sig), X[[n]][, pa_c])
              f2 <- kernelMatrix(rbfdot(sigma=sig), gp_fit$residuals)
              f3 <- kernelMatrix(rbfdot(sigma=sig), cbind(X[[n]][, pa_c], gp_fit$residuals))
              pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
            }
          }else{
            for(n in 1:N){
              gam_fit <- train_gam(X[[n]][, pa_c], X[[n]][, c])
              f1 <- kernelMatrix(rbfdot(sigma=sig), X[[n]][, pa_c])
              f2 <- kernelMatrix(rbfdot(sigma=sig), gam_fit$residuals)
              f3 <- kernelMatrix(rbfdot(sigma=sig), cbind(X[[n]][, pa_c], gam_fit$residuals))
              pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
            }
          }
          all_pa_residuals <- rbind(all_pa_residuals, pa_residuals)
          
          # Generate non-parent residuals
          cat('Getting residuals from non-pa\n')
          k <- 3
          no_pa_c <- which(trueG[, c]==0)
          des_c <- as.numeric(descendants(as.bn(true_dag), as.character(c)))
          anc_c <- as.numeric(ancestors(as.bn(true_dag), as.character(c)))
          non_des_c <- anc_c[!anc_c %in% pa_c]
          no_pa_c <- no_pa_c[!no_pa_c %in% c]
          no_pa_c <- no_pa_c[!no_pa_c %in% c(anc_c)]
          cat('pa_c:', pa_c, ' no_pa_c:', no_pa_c, ' non_des_c:', non_des_c, '\n')
          if(length(no_pa_c) > 0){
            max_d <- 1:min(4, length(no_pa_c))
            no_pa_residuals <- matrix(0, nrow=N, ncol=3*s)
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
                if(linear){
                  for(n in 1:N){
                    gp_fit <- train_linear(X[[n]][, rand_pa[j, ]], X[[n]][, c])
                    f1 <- kernelMatrix(rbfdot(sigma=sig), X[[n]][, rand_pa[j, ]])
                    f2 <- kernelMatrix(rbfdot(sigma=sig), gp_fit$residuals)
                    f3 <- kernelMatrix(rbfdot(sigma=sig), cbind(X[[n]][, rand_pa[j, ]], gp_fit$residuals))
                    no_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                    # no_pa_residuals[n, ] <- apply(f2, 2, mean)
                    # no_pa_residuals[n, ] <- gp_fit$residuals
                  }
                  # kno_pa_residuals <- kernelMatrix(rbfdot(sigma=0.05), no_pa_residuals)
                }else{
                  for(n in 1:N){
                    gam_fit <- train_gam(X[[n]][, rand_pa[j, ]], X[[n]][, c])
                    f1 <- kernelMatrix(rbfdot(sigma=sig), X[[n]][, rand_pa[j, ]])
                    f2 <- kernelMatrix(rbfdot(sigma=sig), gam_fit$residuals)
                    f3 <- kernelMatrix(rbfdot(sigma=sig), cbind(X[[n]][, rand_pa[j, ]], gam_fit$residuals))
                    no_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                    # no_pa_residuals[n, ] <- apply(f2, 2, mean)
                  }
                }
                all_no_pa_residuals <- rbind(all_no_pa_residuals, no_pa_residuals)
                # write.table(residuals, paste('X_', c, '_S', S, '_no_pa_', m, '_', j, '_residuals.txt', sep=''), sep='\t', row.names=F)
              }
              # write.table(rand_pa, paste('X_', c, '_S', S, '_no_pa_', m, '.txt', sep=''), sep='\t', row.names=F)
            }
          # Generate mix-parent residuals
            if(length(non_des_c) > 0){
              cat('Getting residuals from mix-pa\n')
              max_pa <- min(3, length(non_des_c))
              max_pa_d <- 1:max_pa
              mix_pa_residuals <- matrix(0, nrow=N, ncol=3*s)
              # mix_pa_residuals <- matrix(0, nrow=N, ncol=s)
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
                      f1 <- kernelMatrix(rbfdot(sigma=sig), X[[n]][, rand_mix_pa[j, ]])
                      f2 <- kernelMatrix(rbfdot(sigma=sig), gp_fit$residuals)
                      f3 <- kernelMatrix(rbfdot(sigma=sig), cbind(X[[n]][, rand_mix_pa[j, ]], gp_fit$residuals))
                      mix_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                      # mix_pa_residuals[n, ] <- apply(f2, 2, mean)
                      # mix_pa_residuals[n, ] <- gp_fit$residuals
                    }
                    # kmix_pa_residuals <- kernelMatrix(rbfdot(sigma=0.05), mix_pa_residuals)
                  }else{
                    for(n in 1:N){
                      gam_fit <- train_gam(X[[n]][, rand_mix_pa[j, ]], X[[n]][, c])
                      f1 <- kernelMatrix(rbfdot(sigma=sig), X[[n]][, rand_mix_pa[j, ]])
                      f2 <- kernelMatrix(rbfdot(sigma=sig), gam_fit$residuals)
                      f3 <- kernelMatrix(rbfdot(sigma=sig), cbind(X[[n]][, rand_mix_pa[j, ]], gam_fit$residuals))
                      mix_pa_residuals[n, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
                      # mix_pa_residuals[n, ] <- apply(f2, 2, mean)
                    }
                  }
                  all_mix_pa_residuals <- rbind(all_mix_pa_residuals, mix_pa_residuals)
                  # write.table(residuals, paste('X_', c, '_S', S, '_mix_pa_', m, '_', j, '_residuals.txt', sep=''), sep='\t', row.names=F)
                }
                # write.table(rand_mix_pa, paste('X_', c, '_S', S, '_mix_pa_', m, '.txt', sep=''), sep='\t', row.names=F)
              }
            }
          }
        }            
      }
      plotKernelstats(all_pa_residuals, all_mix_pa_residuals, all_no_pa_residuals)
      rf_fit <- trainClassifier(all_pa_residuals, all_mix_pa_residuals, all_no_pa_residuals)
      save(all_pa_residuals, all_mix_pa_residuals, all_no_pa_residuals, rf_fit, list=c('all_pa_residuals', 'all_mix_pa_residuals', 'all_no_pa_residuals', 'rf_fit'), file=paste(train_data_path, 'X_train_residuals_p15_s200_sig1_dense.RData', sep=''))
      write.table(all_pa_residuals, paste(train_data_path, 'X_pa_res_P', p, '_G', num_cg, '_S', s, '.txt', sep=''), sep='\t', row.names = F)
      write.table(all_mix_pa_residuals, paste(train_data_path, 'X_mix_pa_res_P', p, '_G', num_cg, '_S', s, '.txt', sep=''), sep='\t', row.names = F)
      write.table(all_no_pa_residuals, paste(train_data_path, 'X_no_pa_res_P', p, '_G', num_cg, '_S', s, '.txt', sep=''), sep='\t', row.names = F)
    }
  }
}

genTestData <- function(linear=TRUE){
  nVec <- c(100, 500)
  pVec <- c(5, 10)
  num_test_cg <- 50
  
  if(validation_flag){
    test_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Validation_data_mix_pa\\'
    numExp <- 1 
  }
  else{
    test_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Test_data_mix_pa\\'
    numExp <- 10 
  }
  for(p in pVec){
    for(s in nVec){
      if(linear){
        new_dir <- paste(test_data_path, 'P', p, '_S', s, '_sparse', sep='') 
      }else{
        new_dir <- paste(test_data_path, 'P', p, '_S', s, '_dense', sep='')
      }
      dir.create(new_dir)
      setwd(new_dir)
      for(n in 1:num_test_cg){
        all_residuals <- data.frame()
        p_con <- 2*2/(p-1)
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
        write.table(trueG, paste('Target_P', p, '_S', s, '_N_sig_1', n, '.txt', sep=''), sep='\t', row.names=F)
        write.table(X, paste('Data_P', p, '_S', s, '_N_sig_1', n, '.txt', sep=''), sep='\t', row.names=F)
        if(p < 10){
          min_pa <- p-1
        }else if(p==10){
          min_pa <- 5
        }else{
          min_pa <- 3
        }
        sig <- 1
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
              f1 <- kernelMatrix(rbfdot(sigma=sig), X[, subsets[, c]])
              f2 <- kernelMatrix(rbfdot(sigma=sig), fit$residuals)
              f3 <- kernelMatrix(rbfdot(sigma=sig), cbind(X[, subsets[, c]], fit$residuals))
              residuals[c, ] <- c(apply(f1, 2, mean), apply(f2, 2, mean), apply(f3, 2, mean))
              # residuals[c, ] <- apply(f2, 2, mean)
              # residuals[c, ] <- abs(fit$residuals)
            }
            all_residuals <- rbind(all_residuals, residuals)
            # write.table(t(subsets), paste('X_', i, '_subset_', j, '_P', p, '_S', s, '_N', n, '_pa.txt', sep=''), sep='\t', row.names=F)
          }
        }
        write.table(all_residuals, paste('X_residuals_N_sig_1', n, '.txt', sep=''), sep='\t', row.names=F)
      }
    }
  }  
}

plotKernelstats <- function(pa_residuals, mix_pa_residuals, no_pa_residuals){
  t1 <- apply(pa_residuals, 2, median)
  t2 <- apply(mix_pa_residuals, 2, median)
  t3 <- apply(no_pa_residuals, 2, median)
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

trainClassifier <- function(pa_residuals, mix_pa_residuals, no_pa_residuals){
  pa_train_idx <- sample(1:nrow(pa_residuals), nrow(pa_residuals))
  mix_pa_train_idx <- sample(1:nrow(mix_pa_residuals), nrow(pa_residuals))
  no_pa_train_idx <- sample(1:nrow(no_pa_residuals), 2*nrow(pa_residuals))
  
  train_data <- rbind(pa_residuals[pa_train_idx, ], mix_pa_residuals[mix_pa_train_idx, ], no_pa_residuals[no_pa_train_idx, ])
  labels <- c(rep(1, 2*length(pa_train_idx)), rep(0, length(no_pa_train_idx)))
  
  rf_fit <- randomForest(train_data, as.factor(labels), ntree=501)
  return(rf_fit)
}

predictCR <- function(rf_fit){
  test_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Test_data_mix_pa\\'
  test_g <- 20
  state_of_art <- c('RESIT', 'GDS', 'LINGAM', 'PC', 'GES', 'MMHC', 'Random')
  cr_pam <- c('CRP-CAM-1', 'CRP-CAM-2', 'CRP-CAM-3')
  metrics <- c('SHD', 'SID', 'd', 'FPR')
  cd_results <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=length(metrics), dimnames=list(c(state_of_art, cr_pam), metrics))
  shd_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
  sid_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
  d_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
  fpr_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
  for(g in 1:test_g){
    data <- read.table(paste(test_data_path, 'P', p, '_S', s, '_sparse\\Data_P', p, '_S', s, '_N_sig_1', g, '.txt', sep=''), sep='\t', header=T)
    target <- read.table(paste(test_data_path, 'P', p, '_S', s, '_sparse\\Target_P', p, '_S', s, '_N_sig_1', g, '.txt', sep=''), sep='\t', header=T)
    dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(as.matrix(target), mode="directed"))
    test_data <- read.table(paste(test_data_path, 'P', p, '_S', s, '_sparse\\X_residuals_N_sig_1', g, '.txt', sep=''), sep='\t', header=T)
    # test_data <- test_data[, 101:200]
    # colnames(test_data) <- paste("V", 1:100, sep='')
    # 
    # X_test_norm <- test_data
    # for(i in 1:ncol(X_test_norm)){
    #   X_test_norm[, i] <- (X_test_norm[, i] - X_min[i])/(X_max[i] - X_min[i])
    # }
    
    # rf_pred <- matrix(0, nrow=nrow(test_data), ncol=1)
    # rf_pred[, 1] <- predict(rf_fit, test_data)
    # rf_pred <- rf_pred - 1
    # 
    if(p < 10){
      max_pa_test <- p-1
    }else if(p==10){
      max_pa_test <- 5
    }else{
      max_pa_test <- 3
    }
    # amat_res <- buildEstAmat(rf_pred, nrow(test_data), p, max_pa_test, 0.5)
    
    soa_result <- runCDMethods(data, as.matrix(target), linear)
    # crpam_result <- t(unlist(sapply(amat_res, function(x) getPerformanceMetrics(x, as.matrix(target)))))
    cd_results[state_of_art, ] <- cd_results[state_of_art, ] + soa_result
    # cd_results[cr_pam, ] <- cd_results[cr_pam, ] + crpam_result
    shd_result[state_of_art, g] <- soa_result[state_of_art, 1]
    # shd_result[cr_pam, g] <- crpam_result[, 1]
    sid_result[state_of_art, g] <- soa_result[state_of_art, 2]
    # sid_result[cr_pam, g] <- crpam_result[, 2]
    d_result[state_of_art, g] <- soa_result[state_of_art, 3]
    # d_result[cr_pam, g] <- crpam_result[, 3]
    fpr_result[state_of_art, g] <- soa_result[state_of_art, 4]
    # fpr_result[cr_pam, g] <- crpam_result[, 4]
  }
  cd_results <- cd_results/test_g
  sd_result <- cbind(apply(shd_result, 1, sd), cbind(apply(sid_result, 1, sd)), cbind(apply(d_result, 1, sd), apply(fpr_result, 1, sd)))
  colnames(sd_result) <- c('SHD', 'SID', 'd', 'FPR')
  print(cd_results)
  print(apply(shd_result, 1, sd))
  print(apply(d_result, 1, sd))
  print(apply(fpr_result, 1, sd))
  write.table(cd_results, paste(test_data_path, 'Results_GDS_P', p, '_S', s, '_TrainP', p, 'G10_TestG', test_g, '_sig_1_dense.txt', sep=''), sep='\t')
  write.table(sd_result, paste(test_data_path, 'Results_GDS_SD_P', p, '_S', s, '_TrainP', p, 'G10_TestG', test_g, '_sig_1_dense.txt', sep=''), sep='\t')
  
}

performSampling <- function(all_pa_residuals, all_mix_pa_residuals, all_no_pa_residuals){
  n_pa <- nrow(all_pa_residuals)
  n_mix_pa <- nrow(all_mix_pa_residuals)
  n_no_pa <- nrow(all_no_pa_residuals)
  pa_sample_size <- n_pa
  k <- floor(n_no_pa/(2*pa_sample_size))
  cv_folds <- cvFolds(n_no_pa, K=k)
  cv_fold_mat <- matrix(0, nrow=(n_no_pa/k), ncol=k)
  pa_idx <- sample(1:n_pa, pa_sample_size)
  labels <- c(rep(1, 2*pa_sample_size), rep(0, (n_no_pa/k)))
  mix_pa_idx <- sample(1:n_mix_pa, pa_sample_size)
  cv_fold_mat[, i] <- cv_folds$subset[cv_folds$which==i]
  train_data <- rbind(all_pa_residuals[pa_idx, ], rbind(all_mix_pa_residuals[mix_pa_idx, ], all_no_pa_residuals[cv_fold_mat[, i], ]))
  rf_fit <- randomForest(train_data, as.factor(labels))
  return(rf_fit)
}

performValidation <- function(all_pa_residuals, all_mix_pa_residuals, all_no_pa_residuals, n, p, ss){
  validation_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Validation_data_mix_pa\\'
  data <- read.table(paste(validation_data_path, 'P', p, '_G', 1, '\\Data_P', p, '_S', S, '_N', 1, '.txt', sep=''), sep='\t', header=T)
  target <- read.table(paste(validation_data_path, 'P', p, '_G', 1, '\\Target_P', p, '_S', S, '_N', 1, '.txt', sep=''), sep='\t', header=T)
  dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(as.matrix(target), mode="directed"))
  test_data <- read.table(paste(validation_data_path, 'P', p, '_G', 1, '\\X_all_pa_residuals.txt', sep=''), sep='\t', header=T)
  best_shd <- Inf
  
  for(s in ss){
    pa_train_idx <- sample(1:nrow(all_pa_residuals), nrow(all_pa_residuals))
    mix_pa_train_idx <- sample(1:nrow(all_mix_pa_residuals), s-nrow(all_pa_residuals))
    no_pa_train_idx <- sample(1:nrow(all_no_pa_residuals), s)
    pa_data <- all_pa_residuals[pa_train_idx, ]
    mix_pa_data <- all_mix_pa_residuals[mix_pa_train_idx, ]
    no_pa_data <- all_no_pa_residuals[no_pa_train_idx, ]
    train_data <- rbind(pa_data, rbind(mix_pa_data, no_pa_data))
    labels <- c(rep(1, s), rep(0, s))
    rf_fit <- randomForest(train_data, as.factor(labels))
    
    test_pred_prob <- predict(rf_fit, test_data, type='prob')
    amat <- buildEstAmat(test_pred_prob, nrow(test_data), p)
    est_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(amat, mode="directed"))
    shd_score <- pcalg::shd(dag, est_graph)
    if(shd_score < best_shd){
      best_shd <- shd_score
      best_ss <- s
    }
  }
  cat('best_shd:', best_shd, ' best_ss:', best_ss, '\n')
}