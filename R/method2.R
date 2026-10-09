# =============================
# Load features to build classifier
train_g <- 10
test_g <- 50
P <- c(5, 10, 20)
nVec <- c(100, 500)
K <- 50
all_K <- c(50, 100, 200, 300)
all_ss <- c(1)
S <- 100
k <- 5
max_d <- 1:3
max_pa_d <- 1:3
train_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Train_data_mix_pa\\'
test_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Test_data_mix_pa\\'

for(s in nVec){
  for(p in P){
  
    res <- performValidation(p, all_K, all_ss)
    rf_trained <- res[[1]]
    rf_trained <- rf_trained[[1]]
    new_K <- res[[2]]
    pa_features <- data.frame()
    mix_pa_features <- data.frame()
    no_pa_features <- data.frame()
    for(g in 1:train_g){
      child_nodes <- read.table(paste(train_data_path, 'P', p, '_G', g, '\\child_nodes_S', s, '.txt', sep=''), sep='\t', header=T)
      for(c in 1:nrow(child_nodes)){
        pa_features <- rbind(pa_features, read.table(paste(train_data_path, 'P', p, '_G', g, '\\X_', child_nodes[c, 1], '_', K, '_S', s, '_pa_features.txt', sep=''), sep='\t', header=F))
        for(m in max_d){
          for(n in 1:k){
            no_pa_features <- rbind(no_pa_features, read.table(paste(train_data_path, 'P', p, '_G', g, '\\X_', child_nodes[c, 1], '_', K, '_S', s, '_no_pa_subset_', m, '_', n, '_features.txt', sep=''), sep='\t', header=F))
          }
        }
        for(m in max_pa_d){
          for(n in 1:(k-1)){
            mix_pa_features <- rbind(mix_pa_features, read.table(paste(train_data_path, 'P', p, '_G', g, '\\X_', child_nodes[c, 1], '_', K, '_S', s, '_mix_pa_subset_', m, '_', n, '_features.txt', sep=''), sep='\t', header=F)) 
          }
        }
      }
    }
    # set.seed(1)
    pa_train_idx <- sample(1:nrow(pa_features), nrow(pa_features))
    mix_pa_train_idx <- sample(1:nrow(mix_pa_features), nrow(pa_features))
    no_pa_train_idx <- sample(1:nrow(no_pa_features), 2*nrow(pa_features))
    
    train_data <- rbind(pa_features[pa_train_idx, ], mix_pa_features[mix_pa_train_idx, ], no_pa_features[no_pa_train_idx, ])
    labels <- c(rep(1, 2*length(pa_train_idx)), rep(0, length(no_pa_train_idx)))
    
    rf_fit <- randomForest(train_data, as.factor(labels))
    state_of_art <- c('RESIT', 'GDS', 'LINGAM', 'PC', 'GES', 'MMHC', 'Random')
    cr_pam <- c('CRP-CAM-1', 'CRP-CAM-2', 'CRP-CAM-3')
    metrics <- c('SHD', 'SID', 'd', 'FPR')
    cd_results <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=length(metrics), dimnames=list(c(state_of_art, cr_pam), metrics))
    shd_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
    sid_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
    d_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
    fpr_result <- matrix(0, nrow=length(state_of_art)+length(cr_pam), ncol=test_g, dimnames=list(c(state_of_art, cr_pam), c()))
    for(g in 1:test_g){
      # data <- read.table(paste(test_data_path, 'P', p, '_S', s, '_sparse\\Data_P', p, '_S', s, '_N', g, '.txt', sep=''), sep='\t', header=T)
      data <- X_test[[g]]
      # target <- read.table(paste(test_data_path, 'P', p, '_S', s, '_sparse\\Target_P', p, '_S', s, '_N', g, '.txt', sep=''), sep='\t', header=T)
      target <- testG[[g]]
      dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(as.matrix(target), mode="directed"))
      # test_data <- read.table(paste(test_data_path, 'P', p, '_S', s, '_sparse\\X_residuals_N', g, '.txt', sep=''), sep='\t', header=T)
      test_data <- all_test_features[[g]]
    
      rf_pred <- matrix(0, nrow=nrow(test_data), ncol=1)
      rf_pred[, 1] <- predict(rf_fit, test_data)
      rf_pred[, 1] <- rf_pred[, 1] - 1
      if(p<=10){
        max_pa_test <- 5
      }else{
        max_pa_test <- 3
      }
      amat_res <- buildEstAmat(rf_pred, nrow(test_data), p, max_pa_test, 0.5)

      soa_result <- runCDMethods(data, as.matrix(target), linear)
      crpam_result <- t(unlist(sapply(amat_res, function(x) getPerformanceMetrics(x, as.matrix(target)))))
      cd_results[state_of_art, ] <- cd_results[state_of_art, ] + soa_result
      cd_results[cr_pam, ] <- cd_results[cr_pam, ] + crpam_result
      shd_result[state_of_art, g] <- soa_result[state_of_art, 1]
      shd_result[cr_pam, g] <- crpam_result[, 1]
      sid_result[state_of_art, g] <- soa_result[state_of_art, 2]
      sid_result[cr_pam, g] <- crpam_result[, 2]
      d_result[state_of_art, g] <- soa_result[state_of_art, 3]
      d_result[cr_pam, g] <- crpam_result[, 3]
      fpr_result[state_of_art, g] <- soa_result[state_of_art, 4]
      fpr_result[cr_pam, g] <- crpam_result[, 4]
    }
    cd_results <- cd_results/test_g
    sd_result <- cbind(apply(shd_result, 1, sd), cbind(apply(sid_result, 1, sd)), cbind(apply(d_result, 1, sd), apply(fpr_result, 1, sd)))
    colnames(sd_result) <- c('SHD', 'SID', 'd', 'FPR')
    print(cd_results)
    print(apply(shd_result, 1, sd))
    print(apply(d_result, 1, sd))
    print(apply(fpr_result, 1, sd))
    write.table(cd_results, paste(test_data_path, 'Results_P', p, '_S', s, '_TrainP', p, 'G10_TestG', test_g, '.txt', sep=''), sep='\t')
    write.table(sd_result, paste(test_data_path, 'Results_SD_P', p, '_S', s, '_TrainP', p, 'G10_TestG', test_g, '.txt', sep=''), sep='\t')
  }
}

buildEstAmat <- function(rf_pred, num_test_row, P, max_pa_test, thresh){
  final_pred <- apply(rf_pred, 1, function(x) ifelse(sum(x)>0.5*ncol(rf_pred), 1, 0))
  final_score <- apply(rf_pred, 1, function(x) sum(x)/ncol(rf_pred))
  
  all_pa_subsets <- list()
  idx <- 1
  n_sub <- num_test_row/P
  var <- array(0, num_test_row)
  for(p in 1:P){
    pa_p <- seq(P)[-p]
    var[((p-1)*n_sub+1):(p*n_sub)] <- p
    for(l in 1:max_pa_test){
      subsets <- combn(pa_p, l)
      for(c in 1:ncol(subsets)){
        all_pa_subsets[[idx]] <- subsets[, c] 
        idx <- idx + 1
      }
    }
  }
  amat_1 <- amat_2 <- amat_3 <- matrix(0, nrow=P, ncol=P)
  for(p in 1:P){
    # cat('p:', p, '\n')
    var_pred <- final_pred[((p-1)*n_sub+1):(p*n_sub)]
    var_pa_sub <- all_pa_subsets[((p-1)*n_sub+1):(p*n_sub)]
    var_pa_score <- final_score[((p-1)*n_sub+1):(p*n_sub)]
    all_pa <- var_pa_sub[which(var_pred==1)]
    # heuristic 1
    if(max(var_pa_score) > 0.5){
      score <- max(var_pa_score)
      all_max_score_pa <- var_pa_sub[which(var_pa_score==score)]
      pa_len <- unlist(lapply(all_max_score_pa, length))
      pa_len_idx <- which(pa_len==min(pa_len))
      pa <- unique(unlist(all_max_score_pa[pa_len_idx]))
      pa <- unlist(all_pa[min(which(var_pa_score==1))])
      amat_1[pa, p] <- 1
      # amat_1[p, pa] <- 1
      # cat('Heuristic 1:', pa, '\n')
      # heuristic 2
      # max_score_pa <- unique(unlist(all_max_score_pa))
      # var_pa <- c()
      # for(m in max_score_pa){
      #   flag <- 0
      #   for(p_len in min(pa_len):max(pa_len)){
      #     flag <- flag + ifelse(sum(unlist(lapply(all_max_score_pa[which(pa_len==p_len)], function(x) m%in%x)))>0, 1, 0)
      #   }
      #   if(flag == (max(pa_len)-min(pa_len)+1)){
      #     var_pa <- c(var_pa, m)
      #   }
      # }
      # # cat('Heuristic 2:', var_pa, '\n')
      # amat_2[var_pa, p] <- 1
    }
    
    # heuristic 3
    t <- table(unlist(all_pa))
    print(t)
    var_pa <- as.numeric(names(t[which(t > thresh*length(all_pa))]))
    cat('Heuristic 3:', var_pa, ' all_pa: ', length(all_pa), '\n')
    amat_3[var_pa, p] <- 1
    # amat_3[p, var_pa] <- 1
  }
  return(list(amat_1, amat_2, amat_3))
}

runCDMethods <- function(data, trueG, linear){
  cd_methods <- c('RESIT', 'GDS', 'LINGAM', 'PC', 'GES', 'MMHC', 'Random')
  cd_scores <- matrix(0, nrow=length(cd_methods), ncol=4, dimnames=list(cd_methods, c()))
  p <- ncol(data)
  
  if(linear){
    pars <- list(regr.method = train_linear, regr.pars = list(), indtest.method = indtestHsic, indtest.pars = list())
    cat("running RESIT linear...\n")
    resICML <- ICML(data, alpha=0.05, model = train_linear, indtest = indtestHsic, output = FALSE)
    cat("running GDS linear...\n")
    # resGDS <- GDS(as.matrix(data), "SEMIND", pars, check = "checkUntilFirst", output = TRUE, kvec = c(10000), startAt = "emptyGraph")$Adj
  }else{
    pars <- list(regr.method = train_gam, regr.pars = list(), indtest.method = indtestHsic, indtest.pars = list())
    cat("running RESIT gam...\n")
    resICML <- ICML(data, alpha=0.05, model = train_gam, indtest = indtestHsic, output = FALSE)
    cat("running GDS gam...\n")
    # resGDS <- GDS(as.matrix(data), "SEMIND", pars, check = "checkUntilFirst", output = TRUE, kvec = c(10000), startAt = "emptyGraph")$Adj
  }
  
  cat("running LiNGAM...\n")
  resLINGAM <- lingamWrap(data)$Adj
  
  cat("running PC...\n")
  resPC <- pcWrap(data, alpha = 0.05, mmax=Inf)
  
  cat("running GES...\n")
  score <- new("GaussL0penObsScore", data)
  resGES_fit <- ges(score)
  resGES <- as.matrix(as(as(resGES_fit$essgraph,"graphNEL"),"Matrix"))
  
  cat("running MMHC...\n")
  resMMHC <- mmhc(data.frame(data))
  resMMHCAdj <- bnlearn::amat(resMMHC)
  
  cat("running Random...\n")
  resRand <- as.matrix(randomDAG(p,runif(1)))
  
  resICML_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(resICML, mode="directed"))
  # resGDS_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(resGDS, mode="directed"))
  resPC_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(resPC, mode="directed"))
  resLINGAM_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(resLINGAM, mode="directed"))
  resMMHC_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(resMMHCAdj, mode="directed"))
  resRand_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(resRand, mode="directed"))
  resGES_graph <- as(resGES_fit$essgraph, "graphNEL")
  
  cat("calculating performance metrics...\n")
  cd_scores[1, ] <- getPerformanceMetrics(resICML, trueG)
  # cd_scores[2, ] <- getPerformanceMetrics(resGDS, trueG)
  cd_scores[3, ] <- getPerformanceMetrics(resLINGAM, trueG)
  cd_scores[4, ] <- getPerformanceMetrics(resPC, trueG)
  cd_scores[5, ] <- getPerformanceMetrics(resGES, trueG)
  cd_scores[6, ] <- getPerformanceMetrics(resMMHCAdj, trueG)
  cd_scores[7, ] <- getPerformanceMetrics(resRand, trueG)
  return(cd_scores)
}

getPerformanceMetrics <- function(est_amat, trueG){
  sid <- structIntervDist(trueG, est_amat)$sid
  score <- array(0, 4)
  est_dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(est_amat, mode="directed"))
  dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(trueG, mode="directed"))
  s <- pcalg::shd(dag, est_dag)
  metrics <- compareGraphs(est_dag, dag)
  print(cat('SHD:', s, ' d:', sqrt((1-metrics[1])^2+(1-metrics[3])^2), ' fpr:', metrics[2]))
  score <- c(s, sid, sqrt((1-metrics[1])^2+(1-metrics[3])^2), metrics[2])
  return(score)
}

for(s in nVec){
  for(p in P){
    for(g in 1:test_g){
      data <- read.table(paste(test_data_path, 'P', p, '_G', g, '\\Data_P', p, '_S', s, '_N', g, '.txt', sep=''), sep='\t', header=T)
      target <- read.table(paste(test_data_path, 'P', p, '_G', g, '\\Target_P', p, '_S', s, '_N', g, '.txt', sep=''), sep='\t', header=T)
      dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(as.matrix(target), mode="directed"))
      test_data <- data.frame()
      for(v in 1:p){
        test_data <- rbind(test_data, read.table(paste(test_data_path, 'P', p, '_G', g, '\\X_', v, '_', K, '_P', p, '_S', s, '_N', g, '_pred_features.txt', sep=''), sep='\t', header=F))
      }
    }
  }
}

performValidation <- function(p, all_K, all_ss){
  train_data_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Train_data\\'
  validation_path <- 'G:\\Mandar\\NCSU\\Independent Study\\Third chapter\\Experiments\\Method\\Validation_data\\'
  train_g <- 10
  new_K <- 0
  new_s <- 0
  score <- Inf
  S <- 100
  rf_fit <- list()
  idx <- 1
  for(K in all_K){
    cat('K:', K, '\n')
    pa_features <- data.frame()
    no_pa_features <- data.frame()
    for(g in 1:train_g){
      child_nodes <- read.table(paste(train_data_path, 'P', p, '_G', g, '\\child_nodes_S', S, '.txt', sep=''), sep='\t', header=T)
      for(c in 1:nrow(child_nodes)){
        pa_features <- rbind(pa_features, read.table(paste(train_data_path, 'P', p, '_G', g, '\\X_', child_nodes[c, 1], '_', K, '_S', S, '_pa_features.txt', sep=''), sep='\t', header=F))
        for(m in max_d){
          for(n in 1:k){
            no_pa_features <- rbind(no_pa_features, read.table(paste(train_data_path, 'P', p, '_G', g, '\\X_', child_nodes[c, 1], '_', K, '_S', S, '_no_pa_subset_', m, '_', n, '_features.txt', sep=''), sep='\t', header=F)) 
          }
        }
      }
    }
    
    data <- read.table(paste(validation_path, 'P', p, '_G', 1, '\\Data_P', p, '_S', S, '_N', 1, '.txt', sep=''), sep='\t', header=T)
    target <- read.table(paste(validation_path, 'P', p, '_G', 1, '\\Target_P', p, '_S', S, '_N', 1, '.txt', sep=''), sep='\t', header=T)
    dag <- igraph.to.graphNEL(graph_from_adjacency_matrix(as.matrix(target), mode="directed"))
    test_data <- data.frame()
    for(v in 1:p){
      test_data <- rbind(test_data, read.table(paste(validation_path, 'P', p, '_G', 1, '\\X_', v, '_', K, '_P', p, '_S', S, '_N', 1, '_pred_features.txt', sep=''), sep='\t', header=F))
    }
    
    for(ss in all_ss){
      cat('K:', K, ' ss:', ss, '\n')
      set.seed(1)
      pa_train_idx <- sample(1:nrow(pa_features), nrow(pa_features))
      no_pa_train_idx <- sample(1:nrow(no_pa_features), ss*nrow(pa_features))
      
      train_data <- rbind(pa_features[pa_train_idx, ], no_pa_features[no_pa_train_idx, ])
      labels <- c(rep(1, length(pa_train_idx)), rep(0, length(no_pa_train_idx)))
      
      rf_fit[[idx]] <- randomForest(train_data, as.factor(labels))    
      
      test_pred <- predict(rf_fit[[idx]], test_data)
      test_pred_prob <- predict(rf_fit[[idx]], test_data, type='prob')
      test_pred_prod_cpy <- test_pred_prob
      
      amat <- buildEstAmat(test_pred_prob, nrow(test_data), p)
      
      est_graph <- igraph.to.graphNEL(graph_from_adjacency_matrix(amat, mode="directed"))
      new_score <- getPerformanceMetrics(est_graph, dag)
      if(new_score[1] < score){
        cat('local best score K:', K, ' ss:', ss, '\n')
        score <- new_score[1]
        new_K <- K
        new_s <- ss
        new_idx <- idx
      }
      idx <- idx + 1
    }    
  }
  return(list(rf_fit[[new_idx]], new_K))
}

cd_results <- runCDMethods(data)
cd_results <- rbind(cd_results, getPerformanceMetrics(est_graph, dag))

plot(apply(d1, 1, mean), col='red', xlim=c(0, nrow(d)), ylim=c(min(apply(d, 1, mean)), max(apply(d, 1, mean))))
points(apply(d2, 1, mean), col='green')
points(apply(k_pred_data, 1, mean), col='blue')

labels <- c(rep(1, nrow(d1)), rep(0, nrow(d2)))
rf_fit <- randomForest(d, as.factor(labels), ntree=500)
predict_class <- predict(rf_fit, k_pred_data)
predict_prob <- predict(rf_fit, k_pred_data, type="prob")
cat('Class distribution\n')
table(predict_class)
cat('Pa distribution\n')
predict_prob[order(predict_prob[,2], decreasing = T)[1:5], ]
sort(table(unlist(all_pa_subsets[predict_class==1])))
cat('top-k prob pa\n')


pa_stats <- getHSIC(gp_data, pa_i, i)
no_pa_stats <- getHSIC(gp_data, seq(ncol(gp_data))[-pa_i], i)

getKernelMatrix <- function(i){
  pa <- read.table(paste(edge_data_path, 'pa_', i, '.txt', sep=''), sep='\t', header=T)
  res <- read.table(paste(edge_data_path, 'X_', i, '_pa_gam_residuals.txt', sep=''), sep='\t', header=T)
  n_pa <- nrow(pa)
  pa_data <- list()
  for(j in 1:n_pa){
    cat('Reading.... X_', pa[j,], '\n')
    pa_data[[j]] <- read.table(paste(edge_data_path, 'X_', pa[j,], '.txt', sep=''), sep='\t')
  }
  ss <- nrow(pa_data[[1]])
  N <- ncol(pa_data[[1]])
  features <- matrix(0, nrow=N, ncol=ss)
  for(j in 1:N){
    pa_res <- matrix(0, nrow=ss, ncol=n_pa+1)
    for(p in 1:n_pa){
      pa_res[, p] <- (pa_data[[p]][, j] - mean(pa_data[[p]][, j]))/sd(pa_data[[p]][, j])
    }
    pa_res[, n_pa+1] <- (res[, j] - mean(res[, j]))/sd(res[, j])
    k_mat <- kernelMatrix(rbfdot(sigma=0.05), pa_res)
    features[j, ] <- apply(k_mat, 1, mean)
  }
  return(features)
}

getNoPaKernelMatrix <- function(i, k){
  no_pa_data <- data.frame()
  for(l in 1:k){
    pa <- read.table(paste(edge_data_path, 'no_pa_', i, '_subset_', l, '.txt', sep=''), sep='\t', header=T)
    n_sub <- nrow(pa)
    n_pa <- ncol(pa)
    pa_data <- list()
    idx <- 1
    for(j in 1:n_sub){
      for(m in 1:n_pa){
        cat('Reading... X_', pa[j, m], '\n')
        pa_data[[idx]] <- read.table(paste(edge_data_path, 'X_', pa[j, m], '.txt', sep=''), sep='\t')
        idx <- idx + 1
      }
    }
    pa_data_idx <- 0
    N <- ncol(pa_data[[1]])
    ss <- nrow(pa_data[[1]])
    features <- matrix(0, nrow=N, ncol=ss)
    for(n in 1:n_sub){
      cat('Reading... X_', i, '_no_pa_subset_', l, '_', n, '_gp_residuals.txt', '\n')
      res <- read.table(paste(edge_data_path, 'X_', i, '_no_pa_subset_', l, '_', n, '_gp_residuals.txt', sep=''), sep='\t', header=T)
      for(j in 1:N){
        pa_res <- matrix(0, nrow=ss, ncol=n_pa+1)
        for(p in 1:n_pa){
          x <- pa_data[[pa_data_idx + p]]
          pa_res[, p] <- (x[, j] - mean(x[, j]))/sd(x[, j])
        }
        pa_res[, n_pa+1] <- (res[, j] - mean(res[, j]))/sd(res[, j])
        k_mat <- kernelMatrix(rbfdot(sigma=0.05), pa_res)
        features[j, ] <- apply(k_mat, 1, mean)
      }
      pa_data_idx <- pa_data_idx + n_pa
      no_pa_data <- rbind(no_pa_data, features)
    }
  }
  return(no_pa_data)
}

getPredKernelMatrix <- function(j){
  pred_features <- data.frame()
  dat <- read.table(paste(home, 'dat.csv', sep=''), sep=',', header=T)
  pa <- read.table(paste(edge_data_path, 'X_', j, '_subset_1_pa.txt', sep=''), sep='\t', header=T)
  pa_features <- matrix(0, nrow=((2^length(pa)-1)), ncol=nrow(dat))
  idx <- 0
  for(p in 1:length(pa)){
    cat('Reading subset... X_', j, '_subset_', p, '_pa')
    cat('Reading residuals... X_', j, '_subset_', p, '_pa_gam_residuals')
    pa_sub <- read.table(paste(edge_data_path, 'X_', j, '_subset_', p, '_pa.txt', sep=''), sep='\t', header=T)
    pa_residuals <- read.table(paste(edge_data_path, 'X_', j, '_subset_', p, '_pa_gam_residuals.txt', sep=''), sep='\t', header=T)  
    pa_sub <- t(pa_sub)
    n_pa_sub <- ncol(pa_sub)
    for(n in 1:n_pa_sub){
      pa_j <- pa_sub[, n]
      features <- matrix(0, nrow=nrow(dat), ncol=(length(pa_j)+1))
      for(k in 1:length(pa_j)){
        features[, k] <- (dat[, pa_j[k]] - mean(dat[, pa_j[k]]))/sd(dat[, pa_j[k]])
      }
      features[, (length(pa_j)+1)] <- (pa_residuals[, n] - mean(pa_residuals[, n]))/sd(pa_residuals[, n])
      k_mat <- kernelMatrix(rbfdot(sigma=0.05), features)
      pred_features <- rbind(pred_features, apply(k_mat, 1, mean))
    }
  }
  return(pred_features)
}

getHSIC <- function(data, pa, i){
  ss <- nrow(data)
  k <- 1
  H<-diag(1,ss)-1/ss*matrix(1,ss,ss)
  hsic_stat_pa <- hsic_stat_no_pa <- all_subsets <- list()
  for(j in 1:length(pa)){
    subsets <- combn(pa, j)
    for(c in 1:ncol(subsets)){
      gp_fit <- train_gp(data[, subsets[, c]], data[, i])
      residuals <- gp_fit$residuals
      pa_i <- subsets[, c]
      no_pa_i <- pa[!pa %in% pa_i]
      X_pa <- as.matrix(dist(data[, pa_i], method="euclidean", diag=TRUE, upper=TRUE))
      X_pa <- X_pa^2
      X_no_pa <- as.matrix(dist(data[, no_pa_i], method="euclidean", diag=TRUE, upper=TRUE))
      X_no_pa <- X_no_pa^2
      X_res <- as.matrix(dist(residuals, method="euclidean", diag=TRUE, upper=TRUE))
      X_res <- X_res^2
      
      sigma_pa <- sqrt(0.5*median(X_pa[lower.tri(X_pa, diag=FALSE)]))
      sigma_no_pa <- sqrt(0.5*median(X_no_pa[lower.tri(X_no_pa, diag=FALSE)]))
      sigma_res <- sqrt(0.5*median(X_res[lower.tri(X_res, diag=FALSE)]))
      KX_pa <- exp(-X_pa/(2*sigma_pa^2))
      KX_no_pa <- exp(-X_no_pa/(2*sigma_no_pa^2))
      KY_res <- exp(-X_res/(2*sigma_res^2))
      
      HSIC_pa <- diag(KX_pa%*%H%*%KY_res%*%H)
      HSIC_no_pa <- diag(KX_no_pa%*%H%*%KY_res%*%H)
      hsic_stat_pa[[k]] <- HSIC_pa
      hsic_stat_no_pa[[k]] <- HSIC_no_pa
      all_subsets[[k]] <- subsets[, c]
      k <- k + 1
    }
  }
  return(list(hsic_stat, hsic_stat_no_pa, all_subsets))
}