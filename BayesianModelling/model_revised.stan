// =============================================================================
// HIDDEN ILLNESS-DEATH MODEL
// =============================================================================
//
// MODEL STRUCTURE
// ---------------
// States: 0 = alive, progression-free
//         1 = alive, truly progressed
//         2 = dead
//
// Transitions (all Weibull PH):
//   0 -> 1: progression, h_p(t | Z) with treatment effect beta_p
//   0 -> 2: pre-progression death, h_d(t | Z) with treatment effect beta_d
//   1 -> 2: post-progression death, h_q(u) with clock-reset, no treatment effect
//
// Observation model:
//   Detection at visit v_k implies true progression in:
//     Class A (fraction w_A): (v_{k-1}, v_k] 
//     Class B (fraction w_B): (v_{k-1-D}, v_k] 
//   OR: false positive with probability (1 - Sp_eff)
// =============================================================================

functions {
  // -------------------------------------------------------------------------
  // Log survival likelihood given true progression at t_prog.
  //
  // If t_prog >= t_surv: patient died/censored before progressing.
  //   Entire follow-up under pre-progression hazard.
  // If t_prog < t_surv: patient survived past true progression.
  //   Pre-progression piece (0, t_prog) + post-progression piece
  //   (t_prog, t_surv) with clock reset.
  //
  // Post-progression death shares shape alpha_d with pre-progression death
  // but uses baseline scale lambda_q (no treatment effect).
  // -------------------------------------------------------------------------
  real log_surv_lik(real t_prog, real t_surv, int delta,
                    real alpha_d, real scale_pre, real lambda_q) {
    if (t_prog >= t_surv) {
      if (delta == 1) {
        return log(alpha_d) - alpha_d * log(scale_pre)
               + (alpha_d - 1) * log(t_surv)
               - pow(t_surv / scale_pre, alpha_d);
      } else {
        return -pow(t_surv / scale_pre, alpha_d);
      }
    } else {
      real log_S_pre = -pow(t_prog / scale_pre, alpha_d);
      real t_post = t_surv - t_prog;
      if (delta == 1) {
        return log_S_pre
               + log(alpha_d) - alpha_d * log(lambda_q)
               + (alpha_d - 1) * log(t_post)
               - pow(t_post / lambda_q, alpha_d);
      } else {
        return log_S_pre - pow(t_post / lambda_q, alpha_d);
      }
    }
  }

  // -------------------------------------------------------------------------
  // 5-point Gauss-Legendre quadrature for:
  //   log int_a^b f_Weibull(t | alpha_p, lp_arm) * L_surv(t, ...) dt
  // -------------------------------------------------------------------------
  real log_gl_integral(real a, real b,
                       real alpha_p, real lp_arm,
                       real t_surv, int delta,
                       real alpha_d, real scale_pre, real lambda_q) {
    if (b <= a + 1e-8) return negative_infinity();

    array[5] real nd = {-0.9061798459386640, -0.5384693101056831, 0.0,
                          0.5384693101056831,  0.9061798459386640};
    array[5] real wt = { 0.2369268850561891,  0.4786286704993665,
                          0.5688888888888889,
                          0.4786286704993665,  0.2369268850561891};

    real hw  = (b - a) / 2.0;
    real mid = (a + b) / 2.0;

    vector[5] log_terms;
    for (q in 1:5) {
      real t_q = fmax(hw * nd[q] + mid, 1e-8);
      t_q = fmin(t_q, t_surv - 1e-8);

      log_terms[q] = log(wt[q])
                   + weibull_lpdf(t_q | alpha_p, lp_arm)
                   + log_surv_lik(t_q, t_surv, delta,
                                  alpha_d, scale_pre, lambda_q);
    }
    return log(hw) + log_sum_exp(log_terms);
  }
}


data {
  int<lower=1> N;
  array[N] int<lower=0, upper=1> Z;        // Treatment arm (0=control, 1=active)
  vector<lower=0>[N] Tp_obs;               // Observed progression time (visit time or +Inf)
  vector<lower=0>[N] Ts_obs;               // Observed death/censoring time
  array[N] int<lower=0, upper=1> delta_os; // 1=died, 0=censored
  int<lower=1> K;                          // Number of scheduled assessment visits
  vector<lower=0>[K] visits;               // Visit times (excludes time 0)

  // --- Measurement-class configuration ---
  int<lower=0, upper=1> estimate_w_B;      // 0=fix w_B, 1=estimate
  real<lower=0, upper=1> w_B_fixed;        // Fixed value when estimate_w_B=0
  int<lower=1, upper=3> max_delay;         // Max visit delay for ambiguous class

  // --- False-positive extension ---
  int<lower=0, upper=1> include_fp;        // 0=no FP modelling, 1=include FP branch
  int<lower=0, upper=1> estimate_Sp;       // 0=fix Sp, 1=estimate
  real<lower=0, upper=1> Sp_fixed;         // Fixed Sp value (used when estimate_Sp=0)
  real<lower=0> Sp_alpha;                  // Beta prior shape1 for Sp
  real<lower=0> Sp_beta;                   // Beta prior shape2 for Sp
}


transformed data {
  // Map each Tp_obs to visit index
  array[N] int<lower=0, upper=K> k_obs;
  for (i in 1:N) {
    k_obs[i] = 0;
    if (Tp_obs[i] < positive_infinity()) {
      for (k in 1:K) {
        if (fabs(Tp_obs[i] - visits[k]) < 0.001) {
          k_obs[i] = k;
          break;
        }
      }
    }
  }
  for (i in 1:N) {
    if (Tp_obs[i] < positive_infinity() && k_obs[i] == 0)
      reject("Tp_obs[", i, "] = ", Tp_obs[i], " does not match any visit.");
  }

  // For never-detected patients: last visit at or before death/censoring
  array[N] int<lower=0, upper=K> k_last_neg;
  for (i in 1:N) {
    k_last_neg[i] = 0;
    if (k_obs[i] == 0) {
      for (k in 1:K) {
        if (visits[k] <= Ts_obs[i])
          k_last_neg[i] = k;
      }
    }
  }
}


parameters {
  real<lower=0> alpha_p;     // Weibull shape, progression
  real<lower=0> lambda_p;    // Weibull scale, progression (control baseline)
  real beta_p;               // Log HR for treatment on progression

  real<lower=0> alpha_d;     // Weibull shape, death (shared pre- and post-prog)
  real<lower=0> lambda_d;    // Weibull scale, death (baseline pre-progression)
  real beta_d;               // Log HR for treatment on pre-progression death

  real<lower=0> lambda_q;    // Weibull scale, post-progression death (no trt effect)
                             // Shape shared with alpha_d

  real<lower=0, upper=1> w_B_param;  // Used only when estimate_w_B=1
  real<lower=0, upper=1> Sp_param;   // Used only when estimate_Sp=1
}


transformed parameters {
  real w_B = estimate_w_B ? w_B_param : w_B_fixed;
  real w_A = 1.0 - w_B;
  real Sp_eff = (include_fp == 0) ? 1.0 : (estimate_Sp ? Sp_param : Sp_fixed);
}


model {
  // =========================================================================
  // PRIORS
  // =========================================================================

  // Weibull shapes
  alpha_p ~ lognormal(0, 0.5);    // Median 1, 95% CI (0.37, 2.7)
  alpha_d ~ lognormal(0, 0.5);

  // Weibull scales — weakly informative, calibrated to mesothelioma
  lambda_p ~ lognormal(log(6), 0.5);     // PFS ~ 5.6 months
  lambda_d ~ lognormal(log(15), 0.5);    // OS ~ 14.5 months
  lambda_q ~ lognormal(log(8), 0.5);     // Post-progression ~ 8 months

  // Treatment effects — centred on no effect
  beta_p ~ normal(0, 0.5);               // 95% prior HR in (0.37, 2.7)
  beta_d ~ normal(0, 0.5);

  // Measurement-class mixture weight
  w_B_param ~ beta(6, 14);               // Mean 0.30, sd ~0.10

  // Specificity (for false-positive extension)
  Sp_param ~ beta(Sp_alpha, Sp_beta);    // Informative; default Beta(19,1) → mean 0.95


  // =========================================================================
  // LIKELIHOOD
  // =========================================================================

  real lw_A = log(w_A);
  real lw_B = log(w_B);
  real log_Sp_eff = log(Sp_eff);
  real log_1mSp = log1m(Sp_eff);

  for (i in 1:N) {
    real lp_arm = lambda_p * exp(-beta_p * Z[i] / alpha_p);
    real hr_d   = (Z[i] == 1) ? exp(beta_d) : 1.0;
    real sc_pre = lambda_d / pow(hr_d, 1.0 / alpha_d);


    if (k_obs[i] > 0) {
      // =================================================================
      // CASE 1: PROGRESSION DETECTED AT VISIT k_obs[i]
      // =================================================================
      //
      // Two branches:
      //   (a) TRUE POSITIVE: patient truly progressed, detected.
      //       Class A: T_p in (v_{k-1}, v_k]
      //       Class B: T_p in (v_{k-D}, v_k]
      //       Weight: Sp_eff (detection was a true finding, not a false alarm)
      //
      //   (b) FALSE POSITIVE: patient NOT truly progressed, spurious detection.
      //       T_p > v_k (or T_p > Ts if never progressed)
      //       Weight: (1 - Sp_eff)

      int kd = k_obs[i];

      // --- True positive: Class A interval ---
      real a_A = (kd == 1) ? 1e-8 : visits[kd - 1];
      real b_A = visits[kd];

      // --- True positive: Class B interval ---
      int delay = min(max_delay, kd - 1);
      int k_start = kd - delay;
      real a_B = (k_start <= 1) ? 1e-8 : visits[k_start - 1];

      real log_int_A = log_gl_integral(a_A, b_A, alpha_p, lp_arm,
                                       Ts_obs[i], delta_os[i],
                                       alpha_d, sc_pre, lambda_q);

      real log_int_B;
      if (delay > 0) {
        log_int_B = log_gl_integral(a_B, b_A, alpha_p, lp_arm,
                                    Ts_obs[i], delta_os[i],
                                    alpha_d, sc_pre, lambda_q);
      } else {
        log_int_B = log_int_A;
      }

      // True positive branch (weighted by Sp_eff)
      real log_tp = log_sum_exp(lw_A + log_int_A, lw_B + log_int_B);
      if (include_fp) log_tp += log_Sp_eff;

      if (include_fp) {
        // --- False positive branch ---
        // Patient never truly progressed (or progressed after detection).
        // Contribution: f(no progression by Ts) * L_surv(Ts | never progressed)
        //             + integral_{v_kd}^{Ts} f_p(t) * L_surv(Ts | t) dt
        //
        // Simplification: we approximate by just the "never progressed" tail,
        // since the FP rate is small and the "progressed after detection"
        // integral is a second-order correction.

        real log_Sp_prog = weibull_lccdf(Ts_obs[i] | alpha_p, lp_arm);
        real log_Ls_pre;
        if (delta_os[i] == 1) {
          log_Ls_pre = log(alpha_d) - alpha_d * log(sc_pre)
                     + (alpha_d - 1) * log(Ts_obs[i])
                     - pow(Ts_obs[i] / sc_pre, alpha_d);
        } else {
          log_Ls_pre = -pow(Ts_obs[i] / sc_pre, alpha_d);
        }
        real log_fp = log_1mSp + log_Sp_prog + log_Ls_pre;

        target += log_sum_exp(log_tp, log_fp);
      } else {
        target += log_tp;
      }


    } else {
      // =================================================================
      // CASE 2: PROGRESSION NEVER DETECTED
      // =================================================================
      // body (progressed after last negative scan) + tail (never progressed).
      //
      // Class A: last reliable negative at v_L
      // Class B: last reliable negative at v_{L-D}

      int kL = k_last_neg[i];
      real v_L = (kL > 0) ? visits[kL] : 0.0;

      // --- TAIL: T_p >= Ts (never progressed before death/censoring) ---
      real log_Sp_prog = weibull_lccdf(Ts_obs[i] | alpha_p, lp_arm);
      real log_Ls_pre;
      if (delta_os[i] == 1) {
        log_Ls_pre = log(alpha_d) - alpha_d * log(sc_pre)
                   + (alpha_d - 1) * log(Ts_obs[i])
                   - pow(Ts_obs[i] / sc_pre, alpha_d);
      } else {
        log_Ls_pre = -pow(Ts_obs[i] / sc_pre, alpha_d);
      }
      real log_tail = log_Sp_prog + log_Ls_pre;

      // --- BODY Class A: (v_L, Ts) ---
      real a_A = fmax(v_L, 1e-8);
      real log_body_A = log_gl_integral(a_A, Ts_obs[i], alpha_p, lp_arm,
                                        Ts_obs[i], delta_os[i],
                                        alpha_d, sc_pre, lambda_q);
      real log_piece_A = log_sum_exp(log_body_A, log_tail);

      // --- BODY Class B: (v_{L-D}, Ts) ---
      int delay_B = min(max_delay, max(kL, 0));
      int kL_eff = kL - delay_B;
      real v_L_eff = (kL_eff > 0) ? visits[kL_eff] : 0.0;
      real a_B = fmax(v_L_eff, 1e-8);

      real log_body_B = log_gl_integral(a_B, Ts_obs[i], alpha_p, lp_arm,
                                        Ts_obs[i], delta_os[i],
                                        alpha_d, sc_pre, lambda_q);
      real log_piece_B = log_sum_exp(log_body_B, log_tail);

      target += log_sum_exp(lw_A + log_piece_A, lw_B + log_piece_B);
    }
  }
}


generated quantities {
  // --- Derived hazard ratios ---
  real HR_progression = exp(beta_p);
  real HR_death       = exp(beta_d);
  real w_B_out        = w_B;
  real Sp_out         = Sp_eff;

  // --- Per-patient quantities ---
  vector[N] log_lik;
  vector[N] prob_ambiguous;
  vector[N] prob_progressed;
  vector[N] prob_false_positive;
  vector[N] Ts_obs_rep;   // Replicated observed OS/PFS time
  int<lower=0,upper=1> delta_os_rep[N]; // Replicated OS censoring indicator
  int<lower=0> k_obs_rep[N]; // Replicated visit index of progression

  {
    real lw_A = log(w_A);
    real lw_B = log(w_B);
    real log_Sp_eff = log(Sp_eff);
    real log_1mSp = log1m(Sp_eff);

    for (i in 1:N) {
      real lp_arm = lambda_p * exp(-beta_p * Z[i] / alpha_p);
      real hr_d   = (Z[i] == 1) ? exp(beta_d) : 1.0;
      real sc_pre = lambda_d / pow(hr_d, 1.0 / alpha_d);

      // Simulate "True" latent times
      real T_p_true = weibull_rng(alpha_p, lp_arm);
      real T_d_true = weibull_rng(alpha_d, sc_pre);

      // Apply censoring 
      Ts_obs_rep[i] = fmin(T_d_true, 24.0);
      delta_os_rep[i] = (T_d_true <= 24.0) ? 1 : 0;

      k_obs_rep[i] = 0;
      for (k in 1:8) {
        if (visits[k] > Ts_obs_rep[i]) break;
        if (visits[k] >= T_p_true) {
          if (bernoulli_rng(0.90)) {
            k_obs_rep[i] = k;
            break;
          }
        } else {
          if (bernoulli_rng(1.0 - Sp_eff)) {
            k_obs_rep[i] = k;
            break;
          }
        }
      }


      if (k_obs[i] > 0) {
        int kd = k_obs[i];
        real a_A = (kd == 1) ? 1e-8 : visits[kd - 1];
        real b_A = visits[kd];
        int delay = min(max_delay, kd - 1);
        int k_start = kd - delay;
        real a_B = (k_start <= 1) ? 1e-8 : visits[k_start - 1];

        real log_int_A = log_gl_integral(a_A, b_A, alpha_p, lp_arm,
                                         Ts_obs[i], delta_os[i],
                                         alpha_d, sc_pre, lambda_q);
        real log_int_B = (delay > 0)
          ? log_gl_integral(a_B, b_A, alpha_p, lp_arm,
                            Ts_obs[i], delta_os[i],
                            alpha_d, sc_pre, lambda_q)
          : log_int_A;

        real lt_A = lw_A + log_int_A;
        real lt_B = lw_B + log_int_B;
        real log_tp = log_sum_exp(lt_A, lt_B);

        if (include_fp) {
          log_tp += log_Sp_eff;

          real log_Sp_prog = weibull_lccdf(Ts_obs[i] | alpha_p, lp_arm);
          real log_Ls_pre;
          if (delta_os[i] == 1) {
            log_Ls_pre = log(alpha_d) - alpha_d * log(sc_pre)
                       + (alpha_d - 1) * log(Ts_obs[i])
                       - pow(Ts_obs[i] / sc_pre, alpha_d);
          } else {
            log_Ls_pre = -pow(Ts_obs[i] / sc_pre, alpha_d);
          }
          real log_fp = log_1mSp + log_Sp_prog + log_Ls_pre;

          log_lik[i] = log_sum_exp(log_tp, log_fp);
          prob_false_positive[i] = exp(log_fp - log_lik[i]);
          prob_progressed[i] = 1.0 - prob_false_positive[i];
        } else {
          log_lik[i] = log_tp;
          prob_false_positive[i] = 0.0;
          prob_progressed[i] = 1.0;
        }

        // Class assignment (among true positives)
        prob_ambiguous[i] = exp(lt_B - log_sum_exp(lt_A, lt_B));

      } else {
        int kL = k_last_neg[i];
        real v_L = (kL > 0) ? visits[kL] : 0.0;

        real log_Sp_prog = weibull_lccdf(Ts_obs[i] | alpha_p, lp_arm);
        real log_Ls_pre;
        if (delta_os[i] == 1) {
          log_Ls_pre = log(alpha_d) - alpha_d * log(sc_pre)
                     + (alpha_d - 1) * log(Ts_obs[i])
                     - pow(Ts_obs[i] / sc_pre, alpha_d);
        } else {
          log_Ls_pre = -pow(Ts_obs[i] / sc_pre, alpha_d);
        }
        real log_tail = log_Sp_prog + log_Ls_pre;

        real a_A = fmax(v_L, 1e-8);
        real log_body_A = log_gl_integral(a_A, Ts_obs[i], alpha_p, lp_arm,
                                          Ts_obs[i], delta_os[i],
                                          alpha_d, sc_pre, lambda_q);
        real log_piece_A = log_sum_exp(log_body_A, log_tail);

        int delay_B = min(max_delay, max(kL, 0));
        int kL_eff = kL - delay_B;
        real v_L_eff = (kL_eff > 0) ? visits[kL_eff] : 0.0;
        real a_B = fmax(v_L_eff, 1e-8);

        real log_body_B = log_gl_integral(a_B, Ts_obs[i], alpha_p, lp_arm,
                                          Ts_obs[i], delta_os[i],
                                          alpha_d, sc_pre, lambda_q);
        real log_piece_B = log_sum_exp(log_body_B, log_tail);

        real lt_A = lw_A + log_piece_A;
        real lt_B = lw_B + log_piece_B;
        log_lik[i] = log_sum_exp(lt_A, lt_B);
        prob_ambiguous[i] = exp(lt_B - log_lik[i]);

        real log_body_mix = log_sum_exp(lw_A + log_body_A, lw_B + log_body_B);
        real log_total = log_sum_exp(log_body_mix,
                                     log_sum_exp(lw_A, lw_B) + log_tail);
        prob_progressed[i] = exp(log_body_mix - log_total);
        prob_false_positive[i] = 0.0;  // Never detected → can't be FP
      }
    }
  }
}
