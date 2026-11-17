#ifndef BACKTEST_H
#define BACKTEST_H

#include "lstm.h"
#include "data.h"
#include "metrics.h"
#include "vol.h"

/* ==================== Backtest Config ==================== */
typedef struct {
    double cost_spread;      /* pips, e.g. 1.0 for EURUSD */
    double cost_commission;  /* pips per side, e.g. 0.2 */
    double pip_value;        /* pip size, e.g. 0.0001 for EURUSD */
    double threshold_k;      /* k in threshold_offset = cost / (k * sigma) */
    int hold_bars;           /* h: bars to hold (prediction_horizon) */
    double max_position_size;
    double target_risk;      /* constant expected risk per trade */
} BacktestConfig;

/* ==================== Backtest Result ==================== */
typedef struct {
    Trade *trades;
    int num_trades;
    TradingMetrics trading;
    DiagnosticMetrics diagnostic;
} BacktestResult;

BacktestResult backtest_run(LSTMNetwork *net,
                             Dataset *test_ds,
                             DataRow *raw, int num_raw,
                             int raw_offset,
                             const double *vol_predictions,
                             const BacktestConfig *config);

void backtest_free_result(BacktestResult *result);

#endif /* BACKTEST_H */
