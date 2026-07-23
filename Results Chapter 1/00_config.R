### ========================================================================
## 00_config.R
### ========================================================================
## Configuration file for the simulation study
### ========================================================================
library(data.table)

### ========================================================================
## DISEASE-MODEL CONSTANTS (control arm, mesothelioma)
### ========================================================================

LAMBDA_OS_BASE  <- log(2) / 14.5 # Median Survival ~ 14.5 months as in RioMeso
LAMBDA_PFS_BASE <- log(2) / 5.6 # Median PFS ~ 5.6 months 
LAMBDA_2L       <- log(2) / 8
LAMBDA_FAST     <- log(2) / 3.4

### ========================================================================
## DEFAULT MODEL CONFIGURATION
### ========================================================================

W_B_DEFAULT       <- 0.15 # Default probability of belonging to the "difficult" subpopulation
MAX_DELAY_DEFAULT <- 1L # Allow for up to XL visit delays 
SP_FIXED_DEFAULT  <- 0.90 # Default fixed specificity value 
ESTIMATE_W_B      <- 0L # Whether to estimate w_B (1) or fix it (0) in the model
INCLUDE_FP        <- 1L # Whether to include the false positive branch in the model (1) or not (0)
ESTIMATE_SP       <- 0L # Whether to estimate specificity (1) or fix it (0) in the model

### ========================================================================
## TRIAL DESIGN
### ========================================================================

MAX_FOLLOWUP <- 24 # Maximum follow-up time in months
VISIT_SCHEDULES <- list( # Visit schedules for protocolised imaging 
  "3mo" = seq(3, 24, by = 3),
  "2mo" = seq(2, 24, by = 2),
  "4.5mo" = seq(4.5, 22.5, by = 4.5)
)

### ========================================================================
## SIMULATION SETTINGS
### ========================================================================

N_REP         <- 500L # Number of replicates for Bayesian simulation
STAN_ITER     <- 3000L # Number of iterations for Stan sampling
STAN_WARMUP   <- 1500L # Number of warmup iterations for Stan sampling
STAN_CHAINS   <- 4L # Number of chains for Stan sampling
STAN_ADAPT    <- 0.92 # Adapt delta for Stan sampling
STAN_TREEDEPTH <- 12L # Maximum tree depth for Stan sampling

CAL_PROBS <- sort(unique(c(
  0.025, 0.05, seq(0.10, 0.90, by = 0.05), 0.95, 0.975
)))

### ========================================================================
### SCENARIO GRID — PRIMARY
### ========================================================================

SE_GRID <- c(0.70, 0.80, 0.90, 0.95) # Grid of sensitivity values for simulation scenarios
SP_GRID <- c(0.80, 0.90, 0.95, 1.00) # Grid of specificity values for simulation scenarios

build_primary_scenarios <- function() {
  # Primary scenario
  primary <- expand.grid(
    Se = SE_GRID, Sp = SP_GRID,
    hr_death = 0.70, pfs_factor = 0.9,
    visit_schedule = "2mo",
    alpha_p_true = 1.0, alpha_d_true = 1.0,
    stringsAsFactors = FALSE
  )
  primary$hr_prog <- primary$hr_death * primary$pfs_factor

  # Alternative scenario for perfect classification
  perfect <- data.frame(
    Se = 1.0, Sp = 1.0, hr_death = 0.7, pfs_factor = 0.9,
    visit_schedule = "2mo", alpha_p_true = 1.0, alpha_d_true = 1.0,
    stringsAsFactors = FALSE
  )
  perfect$hr_prog <- perfect$hr_death * perfect$pfs_factor

  # Alternative scenario for no effect (HR=1)
  null_sc <- expand.grid(
    Se = c(0.80, 0.90, 1.00), Sp = c(0.90, 0.95, 1.00),
    hr_death = 1.0, pfs_factor = 1.0,
    visit_schedule = "2mo", alpha_p_true = 1.0, alpha_d_true = 1.0,
    stringsAsFactors = FALSE
  )
  null_sc$hr_prog <- 1.0

  scenarios <- unique(rbind(primary, perfect, null_sc))
  scenarios$scenario_id <- seq_len(nrow(scenarios))
  scenarios$label <- with(scenarios, paste0("Se",Se,"_Sp",Sp,"_HR",hr_death, "_Visit", "2mo"))
  scenarios
}

### ========================================================================
## SCENARIO GRID — SENSITIVITY
### ========================================================================

# Sensitivity analysis scenarios to explore the impact of varying parameters on the model's performance

build_sensitivity_scenarios <- function() {
  base <- expand.grid(
    Se = c(0.80, 0.90), Sp = c(0.90, 0.95),
    hr_death = 0.7, stringsAsFactors = FALSE
  )
  base$hr_prog <- base$hr_death * 0.9

  # w_B sensitivity - varying the fixed value of w_B 
  wb_grid <- expand.grid(w_B = c(0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
                         max_delay = 1L, Sp_fixed = 0.90,
                         alpha_p = 1.0, alpha_d = 1.0,
                         stringsAsFactors = FALSE)
  sens_wb <- merge(base, wb_grid)
  sens_wb$sensitivity_type <- "w_B"

  # Sp_fixed sensitivity - varying the fixed value of Sp_fixed
  sp_grid <- expand.grid(w_B = 0.15, max_delay = 1L,
                         Sp_fixed = c(0.85, 0.90, 0.95, 1.0),
                         alpha_p = 1.0, alpha_d = 1.0,
                         stringsAsFactors = FALSE)
  sens_sp <- merge(base, sp_grid)
  sens_sp$sensitivity_type <- "Sp_fixed"

  # max_delay sensitivity - varying the time window for delayed observation of true response
  md_grid <- expand.grid(w_B = 0.15, max_delay = c(1L, 2L),
                         Sp_fixed = 0.90,
                         alpha_p = 1.0, alpha_d = 1.0,
                         stringsAsFactors = FALSE)
  sens_md <- merge(base, md_grid)
  sens_md$sensitivity_type <- "max_delay"

  # Non-exponential hazards - varying the shape parameters for the Weibull hazard functions
  shape_grid <- data.frame(
    w_B = 0.15, max_delay = 1L, Sp_fixed = 0.90,
    alpha_p = c(0.8, 1.3), alpha_d = c(1.2, 0.8),
    stringsAsFactors = FALSE
  )
  sens_shape <- merge(base, shape_grid)
  sens_shape$sensitivity_type <- "hazard_shape"

  # w_B from closed-form mapping: w_B = 2(1-Se)
  wb_mapped <- base
  wb_mapped$w_B <- 2 * (1 - wb_mapped$Se)
  wb_mapped$max_delay <- 1L
  wb_mapped$Sp_fixed <- 0.90
  wb_mapped$alpha_p <- 1.0
  wb_mapped$alpha_d <- 1.0
  wb_mapped$sensitivity_type <- "w_B_mapped"

  sens <- unique(rbind(sens_wb, sens_sp, sens_md, sens_shape, wb_mapped))
  sens$visit_schedule <- "3mo"
  sens$scenario_id <- seq_len(nrow(sens))
  sens$label <- with(sens, paste0(
    "Se",Se,"_Sp",Sp,"_wB",w_B,"_md",max_delay,
    "_SpF",Sp_fixed,"_aP",alpha_p,"_aD",alpha_d
  ))
  sens
}

### ========================================================================
## SCENARIO GRID — BIAS DECOMPOSITION (Reviewer R2)
### ========================================================================

# Four controlled configurations to isolate bias sources

build_decomposition_scenarios <- function() {
  base_se_sp <- expand.grid(
    Se = c(0.80, 0.90), Sp = c(0.90, 0.95),
    stringsAsFactors = FALSE
  )

  configs <- list()

  for (i in seq_len(nrow(base_se_sp))) {
    se <- base_se_sp$Se[i]
    sp <- base_se_sp$Sp[i]

    # Interval-censoring only: Se=Sp=1, lambda_2l = lambda_fast
    configs[[length(configs)+1]] <- data.frame(
      Se = 1.0, Sp = 1.0, hr_death = 0.7, hr_prog = 0.63,
      lambda_2l_use = LAMBDA_2L, lambda_fast_use = LAMBDA_2L,
      w_B_use = 0.0, config_label = "a_interval_only",
      ref_Se = se, ref_Sp = sp,
      stringsAsFactors = FALSE
    )

    # MC structure on perfect data: Se=Sp=1, lambda_2l = lambda_fast, w_B > 0
    configs[[length(configs)+1]] <- data.frame(
      Se = 1.0, Sp = 1.0, hr_death = 0.7, hr_prog = 0.63,
      lambda_2l_use = LAMBDA_2L, lambda_fast_use = LAMBDA_2L,
      w_B_use = W_B_DEFAULT, config_label = "b_mc_on_perfect",
      ref_Se = se, ref_Sp = sp,
      stringsAsFactors = FALSE
    )

    # MC + misclass, no death confound: Se/Sp < 1, lambda_2l = lambda_fast
    configs[[length(configs)+1]] <- data.frame(
      Se = se, Sp = sp, hr_death = 0.7, hr_prog = 0.63,
      lambda_2l_use = LAMBDA_2L, lambda_fast_use = LAMBDA_2L,
      w_B_use = W_B_DEFAULT, config_label = "c_mc_misclass_no_death_confound",
      ref_Se = se, ref_Sp = sp,
      stringsAsFactors = FALSE
    )

    # Full model: Se/Sp < 1, lambda_2l != lambda_fast
    configs[[length(configs)+1]] <- data.frame(
      Se = se, Sp = sp, hr_death = 0.7, hr_prog = 0.63,
      lambda_2l_use = LAMBDA_2L, lambda_fast_use = LAMBDA_FAST,
      w_B_use = W_B_DEFAULT, config_label = "d_full",
      ref_Se = se, ref_Sp = sp,
      stringsAsFactors = FALSE
    )
  }

  decomp <- do.call(rbind, configs)
  decomp$scenario_id <- seq_len(nrow(decomp))
  decomp$label <- with(decomp, paste0(
    config_label, "_Se", Se, "_Sp", Sp, "_refSe", ref_Se, "_refSp", ref_Sp
  ))
  decomp
}

### ========================================================================
## SAMPLE SIZE
### ========================================================================

# Using Schoenfeld's formula for sample size calculation in survival analysis

compute_sample_size <- function(hr, alpha = 0.05, power = 0.80,
                                max_followup = MAX_FOLLOWUP) {
  if (hr >= 1) hr <- 0.8
  z_a <- qnorm(1 - alpha / 2)
  z_b <- qnorm(power)
  d   <- ceiling(4 * (z_a + z_b)^2 / log(hr)^2)
  p_ctrl <- 1 - exp(-LAMBDA_OS_BASE * max_followup)
  p_trt  <- 1 - exp(-LAMBDA_OS_BASE * hr * max_followup)
  ep <- (p_ctrl + p_trt) / 2
  N  <- ceiling(d / ep)
  if (N %% 2 != 0) N <- N + 1
  list(total_N = N, required_events = d, event_prob = round(ep, 3))
}

### ========================================================================
## STAN DATA PREPARATION
### ========================================================================

# Stan expects data in a specific format and I probably could have written the original data frame to output it like this, but I didn't

prepare_stan_data <- function(d, visit_times,
                              w_B = W_B_DEFAULT,
                              max_delay = MAX_DELAY_DEFAULT,
                              Sp_fixed = SP_FIXED_DEFAULT,
                              include_fp = INCLUDE_FP,
                              estimate_w_B = ESTIMATE_W_B,
                              estimate_Sp = ESTIMATE_SP) {
  K <- length(visit_times)
  k_obs <- integer(nrow(d))
  for (i in seq_len(nrow(d))) {
    if (is.finite(d$obs_prog_time[i])) {
      m <- which(abs(visit_times - d$obs_prog_time[i]) < 0.001)
      if (length(m) == 1) k_obs[i] <- m
      else stop("obs_prog_time[", i, "] doesn't match any visit.")
    }
  }

  list(
    N = nrow(d), Z = d$arm,
    Tp_obs = d$obs_prog_time, Ts_obs = d$os_time,
    delta_os = d$os_event,
    K = K, visits = visit_times,
    estimate_w_B = as.integer(estimate_w_B),
    w_B_fixed = w_B,
    max_delay = as.integer(max_delay),
    include_fp = as.integer(include_fp),
    estimate_Sp = as.integer(estimate_Sp),
    Sp_fixed = Sp_fixed,
    Sp_alpha = 19, Sp_beta = 1
  )
}

cat("Configuration loaded.\n")
