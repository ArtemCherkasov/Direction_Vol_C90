/*
 * T-B4: Threshold — document that there is no "no trade" zone.
 * With p=0.5: code says p >= 0.5 → long, else short. Every bar trades.
 * Assert: num_trades == num_samples (when vol filter doesn't skip).
 */
#include "test_common.h"

int test_tb4_run(void) {
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

    /* p = 0.5 (sigmoid(0) = 0.5) */
    net = make_constant_net(ds->num_features, 0.0);

    vol_preds = (double *)malloc(ds->num_samples * sizeof(double));
    for (i = 0; i < ds->num_samples; i++) vol_preds[i] = 0.01;

    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.0; cfg.cost_commission = 0.0;  /* zero cost → vol filter always passes */
    cfg.pip_value = 0.0001; cfg.threshold_k = 0.0;     /* threshold_k=0 → 0 < 0 false → never skip */
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.02;
    cfg.prediction_threshold = 0.5;

    br = backtest_run(net, ds, raw, num_raw, 0, vol_preds, &cfg);

    printf("  num_samples=%d num_trades=%d\n", ds->num_samples, br.num_trades);

    /* Every bar trades — there is no threshold gap */
    ASSERT_INT(br.num_trades, ds->num_samples);

    /* All trades are long (p=0.5, p >= 0.5 → long) */
    for (i = 0; i < br.num_trades; i++) {
        ASSERT_INT(br.trades[i].direction, 1);
    }

    backtest_free_result(&br);
    free(vol_preds);
    lstm_free(net);
    data_free_dataset(ds);
    printf("T-B4 PASS: p=0.5 → all %d bars trade (no threshold gap, p>=0.5 → long)\n", seq);
    return 0;
}
