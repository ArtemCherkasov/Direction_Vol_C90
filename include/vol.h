#ifndef VOL_H
#define VOL_H

#include "data.h"

/* ==================== Volatility Model Types ==================== */
#define VOL_MODE_GARCH 0
#define VOL_MODE_EWMA  1

/* GARCH(1,1) parameters: sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1} */
typedef struct {
    double omega;
    double alpha;
    double beta;
    double sigma2_init;  /* initial variance (long-run or first sample) */
} GarchParams;

/* ==================== EWMA ==================== */

/* Compute EWMA volatility series.
 * sigma2[i] = lambda * sigma2[i-1] + (1-lambda) * r^2[i-1]
 * sigma2[0] = r^2[0] (seed with first squared return)
 * Caller frees the returned array. */
double *vol_ewma(const double *returns, int n_returns, double lambda);

/* Predict next-bar EWMA variance. */
double vol_ewma_next(const double *returns, int n_returns, double lambda);

/* ==================== GARCH(1,1) ==================== */

/* Fit GARCH(1,1) parameters via Nelder-Mead optimization on log-likelihood.
 * max_iter: 200-400 recommended.
 * Constraint: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 */
GarchParams vol_garch_fit(const double *returns, int n_returns, int max_iter);

/* Compute GARCH variance series given fitted params.
 * Returns array of length n_returns. Caller frees. */
double *vol_garch_series(const double *returns, int n_returns, const GarchParams *params);

/* Predict next-bar GARCH variance. */
double vol_garch_next(const double *returns, int n_returns, const GarchParams *params);

/* ==================== Unified Interface ==================== */

/* Compute predictive volatility for each bar in a range.
 * Fits on returns[0..fit_end], predicts vol for [predict_start..predict_end).
 * For GARCH: params from train are used (no re-fit on test).
 * Returns array of length (predict_end - predict_start) of sigma (std dev).
 * Caller frees. */
double *vol_predict(const double *returns, int n_returns,
                    int fit_end, int predict_start, int predict_end,
                    int vol_mode, double ewma_lambda,
                    GarchParams *out_garch);

void vol_free(double *vol_array);

#endif /* VOL_H */
