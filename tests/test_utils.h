#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "lstm.h"
#include "data.h"
#include "types.h"
#include "vol.h"
#include "metrics.h"
#include "train.h"
#include "backtest.h"

/* Test functions return 1=pass, 0=fail */
int test_mtf_no_lookahead(void);
int test_target_consecutive_future(void);
int test_normalization_train_only(void);
int test_val_split_chronological(void);
int test_date_boundary(void);
int test_add_months(void);

int test_vol_ewma_series(void);
int test_vol_ewma_next(void);
int test_vol_garch_fit(void);
int test_vol_garch_constraints(void);

int test_metrics_profit_factor(void);
int test_metrics_sharpe(void);
int test_metrics_drawdown(void);
int test_metrics_brier(void);
int test_metrics_verdict(void);

int test_params_new_keys(void);
int test_params_patience_default(void);

int test_backtest_one_signal_per_bar(void);
int test_backtest_costs(void);

int test_model_garch_roundtrip(void);

#endif /* TEST_UTILS_H */
