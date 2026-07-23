### ========================================================================
## 01_datasets.R
### ========================================================================
## Generates a single two-arm RCT with scan-level misclassification.  
## Processes interval-by-interval through the visit schedule.
### ========================================================================

source("scripts/00_config.R")

## -------------------------------------------------------------------------
## Helper: draw a conditional Weibull event time
## Given survival to t0, returns absolute event time from Weibull(alpha, sigma)
## Uses inversion of the survival function (T = sigma * (-log(U) + (prev_t/sigma)^alpha)^{1/alpha})
## Reduces to t0 + Exp(rate = 1/sigma) when alpha = 1 via memorylessness
## -------------------------------------------------------------------------
rweibull_conditional <- function(n, alpha, sigma, t0) {
  if (alpha == 1.0) {
    return(t0 + rexp(n, rate = 1.0 / sigma))
  }
  u <- runif(n)
  cum_haz_t0 <- (t0 / sigma)^alpha
  sigma * (cum_haz_t0 - log(u))^(1.0 / alpha)
}

## -------------------------------------------------------------------------
## Helper: Weibull scale matching an exponential rate's median
## Exp(lambda) has median log(2)/lambda.
## Weibull(alpha, sigma) has median sigma * log(2)^{1/alpha}.
## Equating: sigma = log(2)^{1 - 1/alpha} / lambda.
## When alpha=1: sigma = 1/lambda.
## -------------------------------------------------------------------------
weibull_scale_from_rate <- function(lambda, alpha) {
  log(2)^(1.0 - 1.0 / alpha) / lambda
}


simulate_trial <- function(
    iter_idx,
    total_patients,
    sensitivity    = 0.80,
    specificity    = 0.95,
    visit_times    = VISIT_SCHEDULES[["3mo"]],
    os_hr          = 0.70,
    pfs_hr         = 0.63,
    alpha_p_true   = 1.0,     # Weibull shape for progression
    alpha_d_true   = 1.0,     # Weibull shape for pre-progression death
    lambda_os_base = LAMBDA_OS_BASE,
    lambda_pfs_base = LAMBDA_PFS_BASE,
    lambda_2l      = LAMBDA_2L,
    lambda_fast    = LAMBDA_FAST,
    max_followup   = NULL
) {
  set.seed(42 + iter_idx)
  if (is.null(max_followup)) max_followup <- max(visit_times)
  n <- total_patients

  ## --- 1:1 randomisation ---
  arm <- c(rep(0L, n %/% 2), rep(1L, n - n %/% 2))

  ## --- Weibull scales ---
  sigma_pfs_ctrl <- weibull_scale_from_rate(lambda_pfs_base, alpha_p_true)
  sigma_os_ctrl  <- weibull_scale_from_rate(lambda_os_base, alpha_d_true)

  ## PH parameterisation: sigma_trt = sigma_ctrl / HR^{1/alpha}
  sigma_pfs_trt <- sigma_pfs_ctrl / pfs_hr^(1.0 / alpha_p_true)
  sigma_os_trt  <- sigma_os_ctrl  / os_hr^(1.0 / alpha_d_true)

  ## Per-patient scales
  sigma_pfs <- ifelse(arm == 1, sigma_pfs_trt, sigma_pfs_ctrl)
  sigma_os  <- ifelse(arm == 1, sigma_os_trt, sigma_os_ctrl)

  ## --- Pre-allocating variables for state tracking ---
  alive            <- rep(TRUE, n)
  truly_progressed <- rep(FALSE, n)
  detected         <- rep(FALSE, n)
  death_time       <- rep(NA_real_, n)
  death_event      <- rep(NA_integer_, n)
  obs_prog_time    <- rep(Inf, n)
  n_scans_attended <- rep(0L, n)
  true_prog_time   <- rep(Inf, n)

  n_tp <- 0L; n_fn <- 0L; n_tn <- 0L; n_fp <- 0L
  prev_t <- 0.0

  # Iterate through each scheduled visit time, simulating deaths, progressions, and scan detections
  for (k in seq_along(visit_times)) {
    vt <- visit_times[k]
    idx <- which(alive & !detected)
    if (length(idx) == 0) break

    ## -- Deaths in (prev_t, vt] --
    prog_mask <- truly_progressed[idx]
    death_times_cond <- rep(Inf, length(idx))

    ## Progressed & undetected: exponential from progression time (memoryless)
    if (any(prog_mask)) {
      pi <- which(prog_mask)
      death_times_cond[pi] <- prev_t + rexp(length(pi), rate = lambda_fast)
    }
    ## Not progressed: Weibull death conditional on T > prev_t
    if (any(!prog_mask)) {
      npi <- which(!prog_mask)
      pat_idx <- idx[npi]
      death_times_cond[npi] <- rweibull_conditional(
        length(npi), alpha_d_true, sigma_os[pat_idx], prev_t
      )
    }

    died <- death_times_cond <= vt
    if (any(died)) {
      di <- idx[died]
      death_time[di]  <- death_times_cond[died]
      death_event[di] <- 1L
      alive[di]       <- FALSE
    }

    ## -- New true progressions in (prev_t, vt] --
    idx <- which(alive & !detected)
    if (length(idx) == 0) { prev_t <- vt; next }
    np <- !truly_progressed[idx]
    if (any(np)) {
      inp <- idx[np]
      cond_prog <- rweibull_conditional(
        length(inp), alpha_p_true, sigma_pfs[inp], prev_t
      )
      pnow <- cond_prog <= vt
      if (any(pnow)) {
        ni <- inp[pnow]
        truly_progressed[ni] <- TRUE
        true_prog_time[ni]   <- cond_prog[pnow]
      }
    }

    ## -- Scan at time vt --
    idx <- which(alive & !detected)
    if (length(idx) == 0) { prev_t <- vt; next }
    n_scans_attended[idx] <- n_scans_attended[idx] + 1L
    tp_mask <- truly_progressed[idx]
    scan_pos <- logical(length(idx))
    if (any(tp_mask)) {
      det <- runif(sum(tp_mask)) < sensitivity
      scan_pos[tp_mask] <- det
      n_tp <- n_tp + sum(det)
      n_fn <- n_fn + sum(!det)
    }
    if (any(!tp_mask)) {
      fa <- runif(sum(!tp_mask)) < (1 - specificity)
      scan_pos[!tp_mask] <- fa
      n_fp <- n_fp + sum(fa)
      n_tn <- n_tn + sum(!fa)
    }

    ## -- Handle observed progressions --
    oi <- idx[scan_pos]
    if (length(oi) > 0) {
      obs_prog_time[oi] <- vt
      detected[oi] <- TRUE
      ps <- rexp(length(oi), rate = lambda_2l)
      rd <- vt + ps
      cens <- rd > max_followup
      death_time[oi[!cens]]  <- rd[!cens]
      death_event[oi[!cens]] <- 1L
      death_time[oi[cens]]   <- max_followup
      death_event[oi[cens]]  <- 0L
      alive[oi] <- FALSE
    }
    prev_t <- vt
  }

  ## Admin censoring
  sa <- which(alive)
  if (length(sa) > 0) {
    death_time[sa]  <- max_followup
    death_event[sa] <- 0L
  }
  stopifnot(!anyNA(death_time), !anyNA(death_event))

  ## Observed PFS
  obs_pfs_time  <- pmin(obs_prog_time, death_time)
  obs_pfs_event <- ifelse(obs_prog_time <= death_time, 1L, death_event)

  data.frame(
    patient_id = 1:n, arm = arm,
    obs_prog_time = obs_prog_time,
    obs_pfs_time = obs_pfs_time, obs_pfs_event = obs_pfs_event,
    os_time = death_time, os_event = death_event,
    n_scans = n_scans_attended,
    true_prog_time = true_prog_time,
    truly_progressed = truly_progressed, detected = detected,
    n_true_pos = rep(n_tp, n), n_false_neg = rep(n_fn, n),
    n_true_neg = rep(n_tn, n), n_false_pos = rep(n_fp, n)
  )
}
