/*
 * T-B4b: Dynamic vol threshold — verify the gate CAN block trades.
 * With high threshold_k and low vol, the vol filter should skip trades
 * where k*sigma < cost_fraction.
 * Config: cost_spread=1.0, commission=0.2, pip_value=0.0001, price~1.0
 *   cost_fraction = (1.0 + 0.4) * 0.0001 / 1.0 = 0.00014
 * With vol=0.001 and threshold_k=100: k*sigma = 0.1 >> cost → trades
 * With vol=0.001 and threshold_k=0.1: k*sigma = 0.0001 < cost → blocked
 */
#include "test_common.h"

int test_tb4b_run(void) {
    int num_raw = 30, seq = 5, h = 1;
    DataRow raw[30];
    Dataset *ds;
    LSTMNetwork *net;
    BacktestConfig cfg;
    BacktestResult br;
    double *vol_preds;
    int i;

    for (i = 0; i < num_raw; i++) {
        double c = 1.0 + 0.01 * i;
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = c; raw[i].high = c + 0.005; raw[i].low = c - 0.005; raw[i].close = c;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    /* p ≈ 0.8 (sigmoid(1.4) ≈ 0.80) — should trade if vol filter passes */
    net = make_constant_net(ds->num_features, 1.4);

    vol_preds = (double *)malloc(ds->num_samples * sizeof(double));
    for (i = 0; i < ds->num_samples; i++) vol_preds[i] = 0.001;  /* low vol */

    /* --- Case 1: threshold_k high → k*sigma >> cost → all trade --- */
    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 1.0; cfg.cost_commission = 0.2;
    cfg.pip_value = 0.0001; cfg.threshold_k = 100.0;  /* 100 * 0.001 = 0.1 >> 0.00014 */
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.02;

    br = backtest_run(net, ds, raw, num_raw, 0, vol_preds, &cfg);
    printf("  Case 1 (k=100, vol=0.001): k*sigma=%.4f >> cost=%.6f → trades=%d\n",
           cfg.threshold_k * 0.001, (1.0 + 0.4) * 0.0001 / 1.0, br.num_trades);
    ASSERT_TRUE(br.num_trades > 0);  /* gate open */
    backtest_free_result(&br);

    /* --- Case 2: threshold_k low → k*sigma < cost → all blocked --- */
    cfg.threshold_k = 0.1;  /* 0.1 * 0.001 = 0.0001 < 0.00014 */

    br = backtest_run(net, ds, raw, num_raw, 0, vol_preds, &cfg);
    printf("  Case 2 (k=0.1, vol=0.001): k*sigma=%.6f < cost=%.6f → trades=%d\n",
           cfg.threshold_k * 0.001, (1.0 + 0.4) * 0.0001 / 1.0, br.num_trades);
    ASSERT_INT(br.num_trades, 0);  /* gate closed */
    backtest_free_result(&br);

    /* --- Case 3: zero costs → gate always open --- */
    cfg.cost_spread = 0.0; cfg.cost_commission = 0.0;
    cfg.threshold_k = 0.1;  /* k*sigma = 0.0001, cost = 0 → 0.0001 > 0 */

    br = backtest_run(net, ds, raw, num_raw, 0, vol_preds, &cfg);
    printf("  Case 3 (k=0.1, zero cost): k*sigma=%.6f > cost=0 → trades=%d\n",
           cfg.threshold_k * 0.001, br.num_trades);
    ASSERT_TRUE(br.num_trades > 0);  /* gate open (zero cost) */
    backtest_free_result(&br);

    free(vol_preds);
    lstm_free(net);
    data_free_dataset(ds);
    printf("T-B4b PASS: vol filter blocks trades when k*sigma < cost, opens when sufficient\n");
    return 0;
}
