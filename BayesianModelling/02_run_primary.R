## =============================================================================
## 02_run_primary.R — Primary simulation study
## =============================================================================
## 
## 
## =============================================================================

### --------- Pre-amble --------- ###
# --- Load libraries --- #
library(foreach)
library(doParallel)
library(rstan)
library(survival)

# --- Stan configuration --- #
Sys.setenv(LOCAL_CPPFLAGS = '-march=native -mtune=native') # Not sure why I needed this, but Stan wouldn't compile on the HPC without it
rstan_options(auto_write = TRUE) 
options(mc.cores = 1) # I only want 1 core per Stan model, since I'm parallelising at the trial level

# --- Load config and helper functions --- #
source("scripts/00_config.R")
source("scripts/01_datasets.R")

# --- Parallelisation setup --- #
num_cores <- as.numeric(Sys.getenv("SLURM_CPUS_PER_TASK"))
if (is.na(num_cores)) num_cores <- parallel::detectCores()
cat("Using", num_cores, "cores\n")
cl <- makeCluster(num_cores)
registerDoParallel(cl)

mod <- stan_model("stan/model_revised.stan", verbose = FALSE, auto_write = TRUE)

# --- Build scenarios --- #
scenarios <- build_primary_scenarios()
cat(nrow(scenarios), "primary scenarios\n")
scenarios$N_total <- sapply(scenarios$hr_death, function(h) compute_sample_size(h)$total_N)
# For scenarios with hr_death = 1, set N_total to be 402 which is the same as the sample size for hr_death = 0.7
scenarios$N_total[scenarios$hr_death == 1] <- 402
dir.create("results/primary", showWarnings = FALSE, recursive = TRUE)


## ------ Run scenarios ------ ##
for (sc in seq_len(nrow(scenarios))) {
  # --- Extract scenario parameters --- #
  Se_s   <- scenarios$Se[sc]
  Sp_s   <- scenarios$Sp[sc]
  hrd_s  <- scenarios$hr_death[sc]
  hrp_s  <- scenarios$hr_prog[sc]
  N_tot  <- scenarios$N_total[sc]
  vs     <- VISIT_SCHEDULES[[scenarios$visit_schedule[sc]]]
  lab    <- scenarios$label[sc]
  ap_true <- scenarios$alpha_p_true[sc]
  ad_true <- scenarios$alpha_d_true[sc]

  # --- Check if results already exist --- #
  csv_file <- file.path("results/primary", paste0(lab, ".csv"))
  if (file.exists(csv_file)) {
    cat("[SKIP]", sc, "/", nrow(scenarios), ":", lab, "\n"); next
  }
  cat("\n===== Scenario", sc, "/", nrow(scenarios), ":", lab,
      " (N=", N_tot, ") =====\n")




  # --- Run simulations --- #
  res <- foreach(
    r = 1:N_REP, .combine = rbind,
    .packages = c("rstan", "survival"),
    .export = c("simulate_trial", "prepare_stan_data", "mod",
                "rweibull_conditional", "weibull_scale_from_rate",
                "Se_s", "Sp_s", "hrd_s", "hrp_s", "N_tot", "vs",
                "ap_true", "ad_true",
                "LAMBDA_OS_BASE", "LAMBDA_PFS_BASE", "LAMBDA_2L", "LAMBDA_FAST",
                "MAX_FOLLOWUP", "CAL_PROBS",
                "W_B_DEFAULT", "MAX_DELAY_DEFAULT", "SP_FIXED_DEFAULT",
                "INCLUDE_FP", "ESTIMATE_W_B", "ESTIMATE_SP",
                "STAN_ITER", "STAN_WARMUP", "STAN_CHAINS",
                "STAN_ADAPT", "STAN_TREEDEPTH"),
    .inorder = FALSE
  ) %dopar% {



    ## 1. Simulate trial
    d <- simulate_trial(
      iter_idx = r, total_patients = N_tot,
      sensitivity = Se_s, specificity = Sp_s,
      visit_times = vs, os_hr = hrd_s, pfs_hr = hrp_s,
      alpha_p_true = ap_true, alpha_d_true = ad_true,
      max_followup = MAX_FOLLOWUP
    )




    ## 2. Naive Cox: OS
    cx_os   <- coxph(Surv(os_time, os_event) ~ arm, data = d)
    nb      <- as.numeric(coef(cx_os)) # Beta for arm effect
    nse     <- as.numeric(sqrt(vcov(cx_os))) # Standard error for beta
    nci     <- as.numeric(exp(confint(cx_os))) # Confidence interval for HR
    n_cover <- as.numeric(nci[1] <= hrd_s & hrd_s <= nci[2]) # Coverage indicator
    n_sig   <- as.numeric(nci[2] < 1) # Significance indicator






    ## 3. Naive Cox: PFS
    cx_pfs  <- coxph(Surv(obs_pfs_time, obs_pfs_event) ~ arm, data = d)
    pb      <- as.numeric(coef(cx_pfs)) # Beta for arm effect
    pci     <- as.numeric(exp(confint(cx_pfs))) # Confidence interval for HR
    p_cover <- as.numeric(pci[1] <= hrp_s & hrp_s <= pci[2]) # Coverage indicator
    p_sig   <- as.numeric(pci[2] < 1) # Significance indicator





    ## 4. Bayesian HID model
    sdata <- prepare_stan_data(d, vs)
    t0 <- proc.time()["elapsed"] # Measure time taken for Stan fit for reporting 
    fit <- tryCatch(
      sampling(mod, data = sdata, iter = STAN_ITER, warmup = STAN_WARMUP,
               chains = STAN_CHAINS, cores = 1, refresh = 0,
               control = list(adapt_delta = STAN_ADAPT,
                              max_treedepth = STAN_TREEDEPTH)),
      error = function(e) NULL
    )
    t_el <- as.numeric(proc.time()["elapsed"] - t0)

    if (is.null(fit)) { # Error handling for if Stan fails to fit the model
      out <- data.frame(
        rep = r, hr_death_true = hrd_s, hr_prog_true = hrp_s,
        Se_true = Se_s, Sp_true = Sp_s,
        naive_os_beta = nb, naive_os_se = nse,
        naive_os_hr = exp(nb), naive_os_ci_lo = nci[1], naive_os_ci_hi = nci[2],
        naive_os_width = log(nci[2]) - log(nci[1]),
        naive_os_cover = n_cover, naive_os_sig = n_sig,
        naive_pfs_beta = pb, naive_pfs_hr = exp(pb),
        naive_pfs_ci_lo = pci[1], naive_pfs_ci_hi = pci[2],
        naive_pfs_cover = p_cover, naive_pfs_sig = p_sig,
        bayes_hrd = NA, bayes_hrd_lo = NA, bayes_hrd_hi = NA,
        bayes_hrd_cover = NA, bayes_hrd_sig = NA, bayes_hrd_width = NA,
        bayes_hrp = NA, bayes_hrp_lo = NA, bayes_hrp_hi = NA,
        bayes_hrp_cover = NA, bayes_hrp_sig = NA,
        mean_prob_ambig = NA, mean_prob_prog = NA,
        ## PPC: latent state recovery
        ppc_brier = NA, ppc_disc_prog = NA, ppc_disc_noprog = NA,
        n_divergent = NA, max_rhat = NA, min_neff = NA,
        fit_time = t_el, stan_error = TRUE,
        pp_eff_power = NA, pp_cov_degrad = NA
      )
      for (p in CAL_PROBS) out[[paste0("bq_", p)]] <- NA
      return(out)
    }





    ## 5. Extract posteriors
    post <- rstan::extract(fit)
    bd <- post$beta_d; bp <- post$beta_p # Posterior draws for log(HR) for OS and PFS

    b_hrd    <- exp(median(bd)) # Posterior median HR for OS
    b_hrd_lo <- as.numeric(exp(quantile(bd, 0.025))) # Posterior 2.5% quantile for HR for OS
    b_hrd_hi <- as.numeric(exp(quantile(bd, 0.975))) # Posterior 97.5% quantile for HR for OS
    b_hrd_cv <- as.numeric(b_hrd_lo <= hrd_s & hrd_s <= b_hrd_hi) # Coverage indicator for OS
    b_hrd_sg <- as.numeric(b_hrd_hi < 1) # Significance indicator for OS
    b_hrd_w  <- log(b_hrd_hi) - log(b_hrd_lo) # Width of the posterior credible interval for OS

    b_hrp    <- exp(median(bp)) # Posterior median HR for PFS
    b_hrp_lo <- as.numeric(exp(quantile(bp, 0.025))) # Posterior 2.5% quantile for HR for PFS
    b_hrp_hi <- as.numeric(exp(quantile(bp, 0.975))) # Posterior 97.5% quantile for HR for PFS
    b_hrp_cv <- as.numeric(b_hrp_lo <= hrp_s & hrp_s <= b_hrp_hi) # Coverage indicator for PFS
    b_hrp_sg <- as.numeric(b_hrp_hi < 1) # Significance indicator for PFS

    # Assessing quality of w_B estimation in the dataset
    ma <- mean(colMeans(post$prob_ambiguous)) # Mean posterior P(ambiguous) across all patients and draws
    mp <- mean(colMeans(post$prob_progressed)) # Mean posterior P(progressed) across all patients and draws





    ## 6. PPC: Latent state recovery (reviewer R5)
    ## Mean posterior P(progressed) for truly progressed vs not
    post_prob_prog <- colMeans(post$prob_progressed)  # N-vector, averaged over draws
    truly_prog <- d$truly_progressed
    ppc_disc_prog   <- mean(post_prob_prog[truly_prog]) # Mean posterior P(progressed) for truly progressed patients
    ppc_disc_noprog <- mean(post_prob_prog[!truly_prog]) # Mean posterior P(progressed) for truly non-progressed patients
    ## Brier score for latent state recovery
    ppc_brier <- mean((post_prob_prog - as.numeric(truly_prog))^2)




    ## 7. PP effective power (200 posterior draws, simulate + naive Cox)
    n_pp <- 200L
    pp_idx <- sample.int(length(bd), n_pp, replace = FALSE)
    pp_sig <- logical(n_pp)
    pp_cov <- logical(n_pp)
    for (s in seq_len(n_pp)) {
      j <- pp_idx[s]
      pp_d <- simulate_trial(
        iter_idx = r * 10000L + s,
        total_patients = N_tot,
        sensitivity = Se_s, specificity = Sp_s,
        visit_times = vs,
        os_hr = exp(bd[j]), pfs_hr = exp(bp[j]),
        max_followup = MAX_FOLLOWUP
      )
      pp_cx <- tryCatch({
        cx <- coxph(Surv(os_time, os_event) ~ arm, data = pp_d)
        ci <- as.numeric(exp(confint(cx)))
        list(sig = ci[2] < 1, cov = ci[1] <= hrd_s & hrd_s <= ci[2])
      }, error = function(e) list(sig = FALSE, cov = TRUE))
      pp_sig[s] <- pp_cx$sig # Indicator of whether the naive Cox model on the simulated trial is significant
      pp_cov[s] <- pp_cx$cov # Indicator of whether the naive Cox model on the simulated trial covers the true HR
    }
    pp_power  <- mean(pp_sig) # Effective power: proportion of simulated trials that are significant
    pp_covdeg <- 0.95 - mean(pp_cov) # Coverage degradation: difference between nominal 95% coverage and actual coverage of the simulated trials

    ## 8. Diagnostics
    sp_diag <- get_sampler_params(fit, inc_warmup = FALSE)
    nd <- sum(sapply(sp_diag, function(x) sum(x[, "divergent__"])))
    fs <- summary(fit, pars = c("beta_d","beta_p","alpha_d","lambda_d",
                                 "alpha_p","lambda_p","lambda_q"))$summary
    mr <- max(fs[, "Rhat"], na.rm = TRUE)
    mn <- min(fs[, "n_eff"], na.rm = TRUE)

    ## 9. Calibration quantiles
    bq <- as.numeric(quantile(bd, probs = CAL_PROBS))

    ## 10. Output row
    out <- data.frame(
      rep = r, hr_death_true = hrd_s, hr_prog_true = hrp_s,
      Se_true = Se_s, Sp_true = Sp_s,
      naive_os_beta = nb, naive_os_se = nse,
      naive_os_hr = exp(nb), naive_os_ci_lo = nci[1], naive_os_ci_hi = nci[2],
      naive_os_width = log(nci[2]) - log(nci[1]),
      naive_os_cover = n_cover, naive_os_sig = n_sig,
      naive_pfs_beta = pb, naive_pfs_hr = exp(pb),
      naive_pfs_ci_lo = pci[1], naive_pfs_ci_hi = pci[2],
      naive_pfs_cover = p_cover, naive_pfs_sig = p_sig,
      bayes_hrd = b_hrd, bayes_hrd_lo = b_hrd_lo, bayes_hrd_hi = b_hrd_hi,
      bayes_hrd_cover = b_hrd_cv, bayes_hrd_sig = b_hrd_sg, bayes_hrd_width = b_hrd_w,
      bayes_hrp = b_hrp, bayes_hrp_lo = b_hrp_lo, bayes_hrp_hi = b_hrp_hi,
      bayes_hrp_cover = b_hrp_cv, bayes_hrp_sig = b_hrp_sg,
      mean_prob_ambig = ma, mean_prob_prog = mp,
      ppc_brier = ppc_brier, ppc_disc_prog = ppc_disc_prog,
      ppc_disc_noprog = ppc_disc_noprog,
      n_divergent = nd, max_rhat = mr, min_neff = mn,
      fit_time = t_el, stan_error = FALSE,
      pp_eff_power = pp_power, pp_cov_degrad = pp_covdeg,
      visit_schedule = vs[length(vs)]-vs[length(vs)-1] # How frequent the visits are
    )
    for (j in seq_along(CAL_PROBS)) out[[paste0("bq_", CAL_PROBS[j])]] <- bq[j]
    out
  }



  ## Console summary
  res$Se_true <- Se_s; res$Sp_true <- Sp_s
  valid <- !res$stan_error
  cat("  OS Coverage:  Bayes =",
      round(mean(res$bayes_hrd_cover[valid], na.rm=TRUE), 3),
      "  Naive =", round(mean(res$naive_os_cover, na.rm=TRUE), 3), "\n")
  cat("  OS Power:     Bayes =",
      round(mean(res$bayes_hrd_sig[valid], na.rm=TRUE), 3),
      "  Naive =", round(mean(res$naive_os_sig, na.rm=TRUE), 3), "\n")
  cat("  Median HR_d:  Bayes =",
      round(exp(mean(log(res$bayes_hrd[valid]), na.rm=TRUE)), 3),
      "  Naive =", round(exp(mean(log(res$naive_os_hr), na.rm=TRUE)), 3), "\n")
  cat("  Mean P(ambiguous):",
      round(mean(res$mean_prob_ambig[valid], na.rm=TRUE), 3),
      " (target:", W_B_DEFAULT, ")\n")
  cat("  PP eff power:", round(mean(res$pp_eff_power[valid], na.rm=TRUE), 3),
      "  Sim-truth:", round(mean(res$naive_os_sig, na.rm=TRUE), 3), "\n")
  cat("  PPC Brier:", round(mean(res$ppc_brier[valid], na.rm=TRUE), 3),
      "  Disc(prog):", round(mean(res$ppc_disc_prog[valid], na.rm=TRUE), 3),
      "  Disc(no-prog):", round(mean(res$ppc_disc_noprog[valid], na.rm=TRUE), 3), "\n")
  cat("  Fit time (s): median =",
      round(median(res$fit_time[valid], na.rm=TRUE), 1), "\n")
  cat("  Divergent:", sum(res$n_divergent[valid] > 0, na.rm=TRUE),
      "/", sum(valid), "  Errors:", sum(!valid), "\n")

  write.csv(res, csv_file, row.names = FALSE)
}

stopCluster(cl)
cat("\nPrimary simulation complete.\n")
