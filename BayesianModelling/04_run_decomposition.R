## =============================================================================
## 03b_run_decomposition.R — Bias source decomposition
## =============================================================================
## Implements the four controlled configurations from Mathematical Supplement S3
## to isolate bias from:
##   (a) interval censoring alone
##   (b) measurement-class structure on perfect data
##   (c) measurement-class approximation error (misclass, no death confound)
##   (d) full model (misclass + death confound)
## =============================================================================

#--- Pre-amble --- #
library(foreach)
library(doParallel)
library(rstan)
library(survival)

# --- Stan configuration --- #
Sys.setenv(LOCAL_CPPFLAGS = '-march=native -mtune=native')
rstan_options(auto_write = TRUE)
options(mc.cores = 1)

# --- Load config and helper functions --- #
source("scripts/00_config.R")
source("scripts/01_datasets.R")

# --- Parallelisation setup --- #
num_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
if (is.na(num_cores)) num_cores <- parallel::detectCores()
cl <- makeCluster(num_cores)
registerDoParallel(cl)

mod <- stan_model("stan/model_revised.stan", verbose = FALSE, auto_write = TRUE)

# --- Build decomposition scenarios --- #
decomp <- build_decomposition_scenarios()
cat(nrow(decomp), "decomposition scenarios\n")

N_REP_DECOMP <- 300L
N_tot <- compute_sample_size(0.7)$total_N
vs <- VISIT_SCHEDULES[["3mo"]]

dir.create("results/decomposition", showWarnings = FALSE, recursive = TRUE)

## ------ Run decomposition scenarios ------ ##
for (sc in seq_len(nrow(decomp))) {
  s <- decomp[sc, ]
  csv_file <- file.path("results/decomposition", paste0(s$label, ".csv"))

  # --- Check if results already exist --- #
  if (file.exists(csv_file)) {
    cat("[SKIP]", sc, "/", nrow(decomp), ":", s$label, "\n"); next
  }
  cat("\n===== Decomp", sc, "/", nrow(decomp), ":", s$label, "=====\n")

  # --- Run simulations --- #
  res <- foreach(
    r = 1:N_REP_DECOMP, .combine = rbind,
    .packages = c("rstan", "survival"),
    .export = c("simulate_trial", "prepare_stan_data", "mod",
                "rweibull_conditional", "weibull_scale_from_rate",
                "s", "N_tot", "vs",
                "LAMBDA_OS_BASE", "LAMBDA_PFS_BASE", "MAX_FOLLOWUP",
                "INCLUDE_FP", "ESTIMATE_W_B", "ESTIMATE_SP",
                "MAX_DELAY_DEFAULT", "SP_FIXED_DEFAULT",
                "STAN_ITER", "STAN_WARMUP", "STAN_CHAINS",
                "STAN_ADAPT", "STAN_TREEDEPTH"),
    .inorder = FALSE
  ) %dopar% {
    ## Simulate with scenario-specific lambda_2l and lambda_fast
    d <- simulate_trial(
      iter_idx = r, total_patients = N_tot,
      sensitivity = s$Se, specificity = s$Sp,
      visit_times = vs,
      os_hr = s$hr_death, pfs_hr = s$hr_prog,
      lambda_2l = s$lambda_2l_use,
      lambda_fast = s$lambda_fast_use,
      max_followup = MAX_FOLLOWUP
    )

    ## Naive Cox OS
    cx <- coxph(Surv(os_time, os_event) ~ arm, data = d)
    nb <- as.numeric(coef(cx))
    nci <- as.numeric(exp(confint(cx)))

    ## Bayesian HID with scenario-specific w_B
    sdata <- prepare_stan_data(d, vs,
                               w_B = s$w_B_use,
                               max_delay = MAX_DELAY_DEFAULT,
                               Sp_fixed = SP_FIXED_DEFAULT)
    fit <- tryCatch(
      sampling(mod, data = sdata, iter = STAN_ITER, warmup = STAN_WARMUP,
               chains = STAN_CHAINS, cores = 1, refresh = 0,
               control = list(adapt_delta = STAN_ADAPT,
                              max_treedepth = STAN_TREEDEPTH)),
      error = function(e) NULL
    )

    if (is.null(fit)) {
      return(data.frame(
        rep = r, config = s$config_label,
        Se = s$Se, Sp = s$Sp, ref_Se = s$ref_Se, ref_Sp = s$ref_Sp,
        w_B_used = s$w_B_use,
        lambda_2l = s$lambda_2l_use, lambda_fast = s$lambda_fast_use,
        naive_os_beta = nb,
        naive_os_cover = as.numeric(nci[1] <= 0.7 & 0.7 <= nci[2]),
        bayes_beta_d = NA, bayes_hrd = NA,
        bayes_hrd_lo = NA, bayes_hrd_hi = NA,
        bayes_hrd_cover = NA,
        n_divergent = NA, stan_error = TRUE
      ))
    }

    post <- rstan::extract(fit)
    bd <- post$beta_d
    bhrd <- exp(median(bd))
    blo <- as.numeric(exp(quantile(bd, 0.025)))
    bhi <- as.numeric(exp(quantile(bd, 0.975)))
    bcv <- as.numeric(blo <= 0.7 & 0.7 <= bhi)

    sp_d <- get_sampler_params(fit, inc_warmup = FALSE)
    nd <- sum(sapply(sp_d, function(x) sum(x[, "divergent__"])))

    data.frame(
      rep = r, config = s$config_label,
      Se = s$Se, Sp = s$Sp, ref_Se = s$ref_Se, ref_Sp = s$ref_Sp,
      w_B_used = s$w_B_use,
      lambda_2l = s$lambda_2l_use, lambda_fast = s$lambda_fast_use,
      naive_os_beta = nb,
      naive_os_cover = as.numeric(nci[1] <= 0.7 & 0.7 <= nci[2]),
      bayes_beta_d = median(bd), bayes_hrd = bhrd,
      bayes_hrd_lo = blo, bayes_hrd_hi = bhi,
      bayes_hrd_cover = bcv,
      n_divergent = nd, stan_error = FALSE
    )
  }

  write.csv(res, csv_file, row.names = FALSE)
  valid <- !res$stan_error
  cat("  Config:", s$config_label,
      "  Bayes HR:", round(exp(mean(log(res$bayes_hrd[valid]), na.rm=TRUE)), 3),
      "  Bias:", round(mean(res$bayes_beta_d[valid], na.rm=TRUE) - log(0.7), 3),
      "  Cover:", round(mean(res$bayes_hrd_cover[valid], na.rm=TRUE), 3), "\n")
}

stopCluster(cl)
cat("\nDecomposition simulation complete.\n")
