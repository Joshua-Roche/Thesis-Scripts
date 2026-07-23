## =============================================================================
## 03_run_sensitivity.R — Sensitivity analyses
## =============================================================================
## Varies w_B, Sp_fixed, max_delay, and Weibull shapes at a representative
## subset of Se/Sp scenarios. 
## 
## =============================================================================

### --------- Pre-amble --------- ###
# --- Load libraries --- #
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

# --- Build scenarios --- #
scenarios <- build_sensitivity_scenarios()
cat(nrow(scenarios), "sensitivity scenarios\n")

N_REP_SENS <- 300L  # Fewer reps for sensitivity (still precise enough)
dir.create("results/sensitivity", showWarnings = FALSE, recursive = TRUE)


## ------ Run scenarios ------ ##
for (sc in seq_len(nrow(scenarios))) {
  # --- Extract scenario parameters --- #
  s <- scenarios[sc, ]
  vs <- VISIT_SCHEDULES[[s$visit_schedule]]
  N_tot <- compute_sample_size(s$hr_death)$total_N

  # --- Check if results already exist --- #
  csv_file <- file.path("results/sensitivity", paste0(s$label, ".csv"))
  if (file.exists(csv_file)) {
    cat("[SKIP]", sc, "/", nrow(scenarios), ":", s$label, "\n"); next
  }
  cat("\n===== Sensitivity", sc, "/", nrow(scenarios), ":", s$label, "=====\n")

  # --- Run simulations --- #
  res <- foreach(
    r = 1:N_REP_SENS, .combine = rbind,
    .packages = c("rstan", "survival"),
    .export = c("simulate_trial", "prepare_stan_data", "mod",
                "LAMBDA_OS_BASE", "LAMBDA_PFS_BASE", "LAMBDA_2L", "LAMBDA_FAST",
                "MAX_FOLLOWUP", "INCLUDE_FP", "ESTIMATE_W_B", "ESTIMATE_SP",
                "STAN_ITER", "STAN_WARMUP", "STAN_CHAINS",
                "STAN_ADAPT", "STAN_TREEDEPTH"),
    .inorder = FALSE
  ) %dopar% {

    # --- Simulate trial --- #
    d <- simulate_trial(
      iter_idx = r, total_patients = N_tot,
      sensitivity = s$Se, specificity = s$Sp,
      visit_times = vs, os_hr = s$hr_death, pfs_hr = s$hr_prog,
      alpha_p_true = s$alpha_p, alpha_d_true = s$alpha_d,
      max_followup = MAX_FOLLOWUP
    )

    cx <- coxph(Surv(os_time, os_event) ~ arm, data = d) 
    nb <- as.numeric(coef(cx))
    nci <- as.numeric(exp(confint(cx)))

    sdata <- prepare_stan_data(d, vs,
                               w_B = s$w_B,
                               max_delay = s$max_delay,
                               Sp_fixed = s$Sp_fixed)
    t0 <- proc.time()["elapsed"]
    fit <- tryCatch(
      sampling(mod, data = sdata, iter = STAN_ITER, warmup = STAN_WARMUP,
               chains = STAN_CHAINS, cores = 1, refresh = 0,
               control = list(adapt_delta = STAN_ADAPT,
                              max_treedepth = STAN_TREEDEPTH)),
      error = function(e) NULL
    )
    t_el <- as.numeric(proc.time()["elapsed"] - t0)

    if (is.null(fit)) {
      return(data.frame(
        rep = r, Se = s$Se, Sp = s$Sp, hr_death = s$hr_death,
        w_B_used = s$w_B, max_delay_used = s$max_delay, Sp_fixed_used = s$Sp_fixed,
        alpha_p_dgp = s$alpha_p, alpha_d_dgp = s$alpha_d,
        naive_os_hr = exp(nb), naive_os_cover = as.numeric(nci[1]<=s$hr_death & s$hr_death<=nci[2]),
        bayes_hrd = NA, bayes_hrd_lo = NA, bayes_hrd_hi = NA,
        bayes_hrd_cover = NA, mean_prob_ambig = NA,
        n_divergent = NA, fit_time = t_el, stan_error = TRUE
      ))
    }

    post <- rstan::extract(fit)
    bd <- post$beta_d
    bhrd <- exp(median(bd))
    blo <- as.numeric(exp(quantile(bd, 0.025)))
    bhi <- as.numeric(exp(quantile(bd, 0.975)))
    bcv <- as.numeric(blo <= s$hr_death & s$hr_death <= bhi)
    ma <- mean(colMeans(post$prob_ambiguous))

    sp_d <- get_sampler_params(fit, inc_warmup = FALSE)
    nd <- sum(sapply(sp_d, function(x) sum(x[, "divergent__"])))

    data.frame(
      rep = r, Se = s$Se, Sp = s$Sp, hr_death = s$hr_death,
      w_B_used = s$w_B, max_delay_used = s$max_delay, Sp_fixed_used = s$Sp_fixed,
      alpha_p_dgp = s$alpha_p, alpha_d_dgp = s$alpha_d,
      naive_os_hr = exp(nb),
      naive_os_cover = as.numeric(nci[1] <= s$hr_death & s$hr_death <= nci[2]),
      bayes_hrd = bhrd, bayes_hrd_lo = blo, bayes_hrd_hi = bhi,
      bayes_hrd_cover = bcv, mean_prob_ambig = ma,
      n_divergent = nd, fit_time = t_el, stan_error = FALSE
    )
  }

  write.csv(res, csv_file, row.names = FALSE)
  valid <- !res$stan_error
  cat("  Bayes HR_d:", round(exp(mean(log(res$bayes_hrd[valid]),na.rm=TRUE)),3),
      "  Coverage:", round(mean(res$bayes_hrd_cover[valid],na.rm=TRUE),3),
      "  P(ambig):", round(mean(res$mean_prob_ambig[valid],na.rm=TRUE),3), "\n")
}

stopCluster(cl)
cat("\nSensitivity analyses complete.\n")
