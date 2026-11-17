/*
 * T-B1: Perfect signal — CRITICAL test for D2.
 * Synthetic 30 bars: open[t] = close[t-1], prices rise every bar.
 * Net outputs p ≈ 1.0 (all long). Zero costs. High vol to pass filter.
 * Assert: all trades win, WR=100%, P&L > 0.
 */
#include "test_common.h"

/* Create a network that always outputs sigmoid(bias) regardless of input */
/* make_constant_net is now in test_common.h */

int test_tb1_run(void) {
    int num_raw = 30, seq = 5, h = 1;
    DataRow raw[30];
    Dataset *ds;
    LSTMNetwork *net;
    BacktestConfig cfg;
    BacktestResult br;
    double *vol_preds;
    int i;

    /* Build synthetic: open[t] = close[t-1], prices rise every bar */
    raw[0].close = 1.0; raw[0].open = 1.0;
    strcpy(raw[0].date, "2020.01.01"); strcpy(raw[0].time_, "00:00");
    raw[0].high = 1.01; raw[0].low = 0.99; raw[0].tickvol = 1000; raw[0].vol = 500; raw[0].spread = 10;
    for (i = 1; i < num_raw; i++) {
        double prev_close = raw[i-1].close;
        double close = prev_close + 0.01;  /* always up */
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open  = prev_close;   /* open[t] = close[t-1] */
        raw[i].close = close;
        raw[i].high  = close + 0.005;
        raw[i].low   = prev_close - 0.005;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    /* Net that always outputs ≈1.0 (long) */
    net = make_constant_net(ds->num_features, 20.0);

    /* Vol predictions: high enough to pass filter */
    vol_preds = (double *)malloc(ds->num_samples * sizeof(double));
    for (i = 0; i < ds->num_samples; i++) vol_preds[i] = 0.01;

    /* Config: zero costs, hold 1 bar */
    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.0;
    cfg.cost_commission = 0.0;
    cfg.pip_value = 0.0001;
    cfg.threshold_k = 2.0;
    cfg.hold_bars = h;
    cfg.max_position_size = 1.0;
    cfg.target_risk = 0.02;

    br = backtest_run(net, ds, raw, num_raw, 0, vol_preds, &cfg);

    printf("  trades=%d WR=%.1f%% P&L=%.6f\n",
           br.num_trades, br.trading.win_rate * 100.0, br.trading.total_pnl);

    /* All trades should be long (direction=1) */
    for (i = 0; i < br.num_trades; i++) {
        ASSERT_INT(br.trades[i].direction, 1);
        ASSERT_TRUE(br.trades[i].pnl_fraction > 0.0);
    }

    ASSERT_TRUE(br.num_trades > 0);
    ASSERT_APPROX(br.trading.win_rate, 1.0, 1e-9);
    ASSERT_TRUE(br.trading.total_pnl > 0.0);

    backtest_free_result(&br);
    free(vol_preds);
    lstm_free(net);
    data_free_dataset(ds);

    printf("T-B1 PASS: perfect signal → WR=100%%, P&L>0, all trades win\n");
    return 0;
}
