/*
 * T-B3: Costs.
 * Flat prices (open=close=high=low=1.0), p ≈ 1.0 (long).
 * Every trade loses exactly cost_fraction * position_size.
 * P&L = -n * cost_fraction * position_size, PF=0.
 */
#include "test_common.h"

int test_tb3_run(void) {
    int num_raw = 30, seq = 5, h = 1;
    DataRow raw[30];
    Dataset *ds;
    LSTMNetwork *net;
    BacktestConfig cfg;
    BacktestResult br;
    double *vol_preds;
    int i;
    double cost_pips, cost_fraction, pos_size, exp_pnl_per_trade;

    /* Flat prices */
    for (i = 0; i < num_raw; i++) {
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = raw[i].high = raw[i].low = raw[i].close = 1.0;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    net = make_constant_net(ds->num_features, 20.0);  /* p ≈ 1.0 */

    vol_preds = (double *)malloc(ds->num_samples * sizeof(double));
    for (i = 0; i < ds->num_samples; i++) vol_preds[i] = 0.01;

    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.00008; cfg.cost_commission = 0.00003;
    cfg.pip_value = 0.0001; cfg.threshold_k = 2.0;
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.02;
    cfg.prediction_threshold = 0.5;

    br = backtest_run(net, ds, raw, num_raw, 0, vol_preds, &cfg);

    /* Expected: cost_fraction = spread + 2*commission = 0.00008 + 2*0.00003 = 0.00014 */
    cost_pips = cfg.cost_spread + 2.0 * cfg.cost_commission;
    cost_fraction = cost_pips;  /* direct fraction, no pip_value/price conversion */
    pos_size = cfg.target_risk / 0.01;  /* 2.0 */
    if (pos_size > cfg.max_position_size) pos_size = cfg.max_position_size;  /* 1.0 */
    exp_pnl_per_trade = -cost_fraction * pos_size;

    printf("  trades=%d cost_fraction=%.6f pos_size=%.4f exp_pnl=%.8f\n",
           br.num_trades, cost_fraction, pos_size, exp_pnl_per_trade);
    printf("  total_pnl=%.8f WR=%.1f%% PF=%.2f\n",
           br.trading.total_pnl, br.trading.win_rate * 100.0, br.trading.profit_factor);

    ASSERT_TRUE(br.num_trades > 0);

    /* Each trade loses exactly cost_fraction * pos_size */
    for (i = 0; i < br.num_trades; i++) {
        ASSERT_APPROX(br.trades[i].pnl_fraction, exp_pnl_per_trade, 1e-10);
        ASSERT_TRUE(br.trades[i].pnl_fraction < 0.0);
    }

    /* Total P&L = n * cost_fraction * pos_size (negative) */
    ASSERT_APPROX(br.trading.total_pnl, br.num_trades * exp_pnl_per_trade, 1e-10);

    /* PF = 0 (no wins) */
    ASSERT_APPROX(br.trading.profit_factor, 0.0, 1e-9);

    backtest_free_result(&br);
    free(vol_preds);
    lstm_free(net);
    data_free_dataset(ds);
    printf("T-B3 PASS: flat prices → each trade loses cost, P&L=-n*cost, PF=0\n");
    return 0;
}
