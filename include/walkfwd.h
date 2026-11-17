#ifndef WALKFWD_H
#define WALKFWD_H

#include "data.h"
#include "params.h"
#include "train.h"
#include "backtest.h"
#include "metrics.h"
#include "vol.h"

typedef struct {
    int window_num;
    char train_start[32];
    char train_end[32];
    char test_start[32];
    char test_end[32];
    TrainResult train_result;
    BacktestResult backtest_result;
} WindowResult;

typedef struct {
    WindowResult *windows;
    int num_windows;
    TradingMetrics aggregate;
    TradingMetrics worst;
    int criteria_met;
} WalkFwdResult;

WalkFwdResult walkfwd_run(DataRow *raw, int num_raw,
                          DataRow *sec_raw, int sec_num_raw,
                          const ParamsFile *params);
void walkfwd_free(WalkFwdResult *result);
void walkfwd_print_report(const WalkFwdResult *result);

#endif /* WALKFWD_H */
