#include "vol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== EWMA ==================== */

double *vol_ewma(const double *returns, int n_returns, double lambda) {
    double *sigma2;
    int i;

    if (n_returns <= 0) return NULL;
    sigma2 = (double *)utils_safe_malloc((size_t)n_returns * sizeof(double));

    sigma2[0] = returns[0] * returns[0];
    for (i = 1; i < n_returns; i++) {
        sigma2[i] = lambda * sigma2[i - 1] + (1.0 - lambda) * returns[i - 1] * returns[i - 1];
    }

    return sigma2;
}

double vol_ewma_next(const double *returns, int n_returns, double lambda) {
    double *sigma2;
    double result;

    if (n_returns <= 0) return 0.0;
    sigma2 = vol_ewma(returns, n_returns, lambda);
    result = lambda * sigma2[n_returns - 1] + (1.0 - lambda) * returns[n_returns - 1] * returns[n_returns - 1];
    free(sigma2);
    return result;
}

/* ==================== GARCH(1,1) Nelder-Mead ==================== */

/* Negative log-likelihood for GARCH(1,1) */
static double garch_neg_loglik(const double *params, const double *returns, int n) {
    double omega, alpha, beta;
    double sigma2, ll;
    int i;

    omega = params[0];
    alpha = params[1];
    beta = params[2];

    /* Constraint check: penalize violations */
    if (omega <= 0.0 || alpha < 0.0 || beta < 0.0 || (alpha + beta) >= 0.999) {
        return 1e10;
    }

    /* Initial variance = unconditional variance = omega / (1 - alpha - beta) */
    sigma2 = omega / (1.0 - alpha - beta);
    if (sigma2 <= 0.0) return 1e10;

    ll = 0.0;
    for (i = 0; i < n; i++) {
        /* Likelihood of r_i uses σ²_i (conditional on info up to i-1).
         * Must compute BEFORE updating sigma2 to σ²_{i+1}. */
        if (sigma2 <= 1e-20) sigma2 = 1e-20;
        ll += 0.5 * (log(sigma2) + returns[i] * returns[i] / sigma2);
        sigma2 = omega + alpha * returns[i] * returns[i] + beta * sigma2;
    }

    return ll;
}

/* Nelder-Mead simplex optimizer (3D) */
static void nelder_mead_3d(
    double (*objective)(const double *, const double *, int),
    const double *returns, int n,
    double *start, int max_iter, double tol,
    double *out_params)
{
    double simplex[4][3];
    double fval[4];
    double centroid[3];
    double xr[3], xe[3], xc[3];
    double fr, fe, fc;
    int i, j, iter;
    int worst, second_worst, best;

    /* Initialize simplex around start point */
    for (i = 0; i < 3; i++) {
        simplex[0][i] = start[i];
    }
    /* Perturbations for each vertex */
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            simplex[i + 1][j] = start[j];
        }
        simplex[i + 1][i] *= (simplex[i + 1][i] > 0.0) ? 1.1 : 0.9;
        if (simplex[i + 1][i] == 0.0) simplex[i + 1][i] = 1e-6;
    }

    /* Evaluate objective at each vertex */
    for (i = 0; i < 4; i++) {
        fval[i] = objective(simplex[i], returns, n);
    }

    for (iter = 0; iter < max_iter; iter++) {
        /* Sort vertices by function value (simple sort for 4 elements) */
        {
            double tmp_f;
            double tmp_v[3];
            int a, b;
            for (a = 0; a < 4; a++) {
                for (b = a + 1; b < 4; b++) {
                    if (fval[b] < fval[a]) {
                        tmp_f = fval[a]; fval[a] = fval[b]; fval[b] = tmp_f;
                        memcpy(tmp_v, simplex[a], sizeof(tmp_v));
                        memcpy(simplex[a], simplex[b], sizeof(tmp_v));
                        memcpy(simplex[b], tmp_v, sizeof(tmp_v));
                    }
                }
            }
        }

        best = 0; second_worst = 2; worst = 3;

        /* Convergence check: simplex size */
        {
            double max_dist = 0.0;
            for (i = 1; i < 4; i++) {
                for (j = 0; j < 3; j++) {
                    double d = fabs(simplex[i][j] - simplex[0][j]);
                    if (d > max_dist) max_dist = d;
                }
            }
            if (max_dist < tol) break;
        }

        /* Compute centroid of all but worst */
        for (j = 0; j < 3; j++) {
            centroid[j] = 0.0;
            for (i = 0; i < 3; i++) /* 0,1,2 (not worst=3) */
                centroid[j] += simplex[i][j];
            centroid[j] /= 3.0;
        }

        /* Reflection */
        for (j = 0; j < 3; j++)
            xr[j] = centroid[j] + 1.0 * (centroid[j] - simplex[worst][j]);
        fr = objective(xr, returns, n);

        if (fr < fval[best]) {
            /* Expansion */
            for (j = 0; j < 3; j++)
                xe[j] = centroid[j] + 2.0 * (centroid[j] - simplex[worst][j]);
            fe = objective(xe, returns, n);
            if (fe < fr) {
                memcpy(simplex[worst], xe, sizeof(xe));
                fval[worst] = fe;
            } else {
                memcpy(simplex[worst], xr, sizeof(xr));
                fval[worst] = fr;
            }
        } else if (fr < fval[second_worst]) {
            memcpy(simplex[worst], xr, sizeof(xr));
            fval[worst] = fr;
        } else {
            /* Contraction */
            if (fr < fval[worst]) {
                for (j = 0; j < 3; j++)
                    xc[j] = centroid[j] + 0.5 * (xr[j] - centroid[j]);
                fc = objective(xc, returns, n);
                if (fc < fr) {
                    memcpy(simplex[worst], xc, sizeof(xc));
                    fval[worst] = fc;
                } else {
                    /* Shrink */
                    for (i = 1; i < 4; i++) {
                        for (j = 0; j < 3; j++)
                            simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                        fval[i] = objective(simplex[i], returns, n);
                    }
                }
            } else {
                /* Contraction towards best */
                for (j = 0; j < 3; j++)
                    xc[j] = centroid[j] - 0.5 * (centroid[j] - simplex[worst][j]);
                fc = objective(xc, returns, n);
                if (fc < fval[worst]) {
                    memcpy(simplex[worst], xc, sizeof(xc));
                    fval[worst] = fc;
                } else {
                    /* Shrink */
                    for (i = 1; i < 4; i++) {
                        for (j = 0; j < 3; j++)
                            simplex[i][j] = simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]);
                        fval[i] = objective(simplex[i], returns, n);
                    }
                }
            }
        }
    }

    /* Return best vertex */
    {
        double min_f = fval[0];
        int min_idx = 0;
        for (i = 1; i < 4; i++) {
            if (fval[i] < min_f) { min_f = fval[i]; min_idx = i; }
        }
        memcpy(out_params, simplex[min_idx], 3 * sizeof(double));
    }
}

GarchParams vol_garch_fit(const double *returns, int n_returns, int max_iter) {
    GarchParams gp;
    double sample_var;
    double start[3];
    double opt_params[3];
    int i;

    if (n_returns <= 2) {
        gp.omega = 1e-6; gp.alpha = 0.1; gp.beta = 0.85; gp.sigma2_init = 1e-6;
        return gp;
    }

    /* Compute sample variance as initial estimate */
    sample_var = 0.0;
    for (i = 0; i < n_returns; i++) {
        sample_var += returns[i] * returns[i];
    }
    sample_var /= (double)n_returns;
    if (sample_var < 1e-20) sample_var = 1e-20;

    /* Initial guess: omega = var*(1-alpha-beta), alpha=0.1, beta=0.85 */
    start[0] = sample_var * (1.0 - 0.1 - 0.85);
    start[1] = 0.1;
    start[2] = 0.85;

    if (start[0] <= 0.0) start[0] = sample_var * 0.01;

    nelder_mead_3d(garch_neg_loglik, returns, n_returns, start, max_iter, 1e-8, opt_params);

    /* Ensure constraints */
    if (opt_params[0] <= 0.0) opt_params[0] = sample_var * 0.001;
    if (opt_params[1] < 0.0) opt_params[1] = 0.0;
    if (opt_params[2] < 0.0) opt_params[2] = 0.0;
    if (opt_params[1] + opt_params[2] >= 0.999) {
        /* Scale α,β to satisfy stationarity AND adjust ω to preserve
         * unconditional variance ω/(1-α-β). Without ω adjustment, the
         * post-hoc scaling moves away from the likelihood optimum. */
        double old_uncond_var = opt_params[0] / (1.0 - opt_params[1] - opt_params[2]);
        double scale = 0.999 / (opt_params[1] + opt_params[2]);
        opt_params[1] *= scale;
        opt_params[2] *= scale;
        opt_params[0] = old_uncond_var * (1.0 - opt_params[1] - opt_params[2]);
    }

    gp.omega = opt_params[0];
    gp.alpha = opt_params[1];
    gp.beta = opt_params[2];
    gp.sigma2_init = gp.omega / (1.0 - gp.alpha - gp.beta);

    return gp;
}

double *vol_garch_series(const double *returns, int n_returns, const GarchParams *params) {
    double *sigma2;
    int i;

    if (n_returns <= 0) return NULL;
    sigma2 = (double *)utils_safe_malloc((size_t)n_returns * sizeof(double));

    sigma2[0] = params->sigma2_init;
    for (i = 1; i < n_returns; i++) {
        sigma2[i] = params->omega + params->alpha * returns[i - 1] * returns[i - 1]
                    + params->beta * sigma2[i - 1];
    }

    return sigma2;
}

double vol_garch_next(const double *returns, int n_returns, const GarchParams *params) {
    double *sigma2;
    double result;

    if (n_returns <= 0) return params->sigma2_init;
    sigma2 = vol_garch_series(returns, n_returns, params);
    result = params->omega + params->alpha * returns[n_returns - 1] * returns[n_returns - 1]
             + params->beta * sigma2[n_returns - 1];
    free(sigma2);
    return result;
}

/* ==================== Unified Interface ==================== */

double *vol_predict(const double *returns, int n_returns,
                    int fit_end, int predict_start, int predict_end,
                    int vol_mode, double ewma_lambda,
                    GarchParams *out_garch) {
    double *result;
    int result_len;
    int i;

    if (predict_end <= predict_start || n_returns <= 0) return NULL;

    result_len = predict_end - predict_start;
    result = (double *)utils_safe_malloc((size_t)result_len * sizeof(double));

    if (vol_mode == VOL_MODE_GARCH) {
        double *sigma2;
        GarchParams gp;

        gp = vol_garch_fit(returns, fit_end, 300);
        if (out_garch) *out_garch = gp;

        sigma2 = vol_garch_series(returns, n_returns, &gp);

        for (i = 0; i < result_len; i++) {
            int idx = predict_start + i;
            if (idx >= 0 && idx < n_returns) {
                result[i] = (sigma2[idx] > 1e-12) ? sqrt(sigma2[idx]) : 1e-6;
            } else {
                result[i] = (gp.sigma2_init > 1e-12) ? sqrt(gp.sigma2_init) : 1e-6;
            }
        }
        free(sigma2);
    } else {
        /* EWMA */
        double *sigma2;
        sigma2 = vol_ewma(returns, n_returns, ewma_lambda);

        for (i = 0; i < result_len; i++) {
            int idx = predict_start + i;
            if (idx >= 0 && idx < n_returns) {
                result[i] = (sigma2[idx] > 1e-12) ? sqrt(sigma2[idx]) : 1e-6;
            } else {
                result[i] = (sigma2[n_returns - 1] > 1e-12) ? sqrt(sigma2[n_returns - 1]) : 1e-6;
            }
        }
        free(sigma2);

        if (out_garch) {
            out_garch->omega = 0.0;
            out_garch->alpha = 1.0 - ewma_lambda;
            out_garch->beta = ewma_lambda;
            out_garch->sigma2_init = returns[0] * returns[0];
        }
    }

    return result;
}

void vol_free(double *vol_array) {
    free(vol_array);
}
