#ifndef BACKTEST_H
#define BACKTEST_H

#include "lstm.h"
#include "data.h"
#include "metrics.h"
#include "vol.h"

/* ==================== Backtest Config ==================== */
typedef struct {
    double cost_spread;      /* fraction per trade, e.g. 0.0001 = 1bp */
    double cost_commission;  /* fraction per side, e.g. 0.0 */
    double pip_value;        /* pip size (legacy, kept for API compat) */
    double threshold_k;      /* k in threshold_offset = cost / (k * sigma) */
    int hold_bars;           /* h: bars to hold (prediction_horizon) */
    double max_position_size;
    double target_risk;      /* constant expected risk per trade */
    double prediction_threshold;  /* threshold for long/short decision (default 0.5) */
} BacktestConfig;

/* ==================== Backtest Result ==================== */
typedef struct {
    Trade *trades;
    int num_trades;
    TradingMetrics trading;
    DiagnosticMetrics diagnostic;
    /* Filter funnel: how test bars become trades */
    int n_test_bars;      /* total test samples */
    int n_skip_overlap;   /* skipped by non-overlap filter */
    int n_skip_vol;       /* skipped by zero/near-zero vol */
    int n_skip_cost;      /* skipped by vol cost filter */
    int n_skip_oob;       /* skipped by exit_bar out of bounds */
} BacktestResult;

BacktestResult backtest_run(LSTMNetwork *net,
                             Dataset *test_ds,
                             DataRow *raw, int num_raw,
                             int raw_offset,
                             const double *vol_predictions,
                             const BacktestConfig *config);

void backtest_free_result(BacktestResult *result);

#endif /* BACKTEST_H */
