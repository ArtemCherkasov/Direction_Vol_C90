/*
 * T-G2: Golden — perfect monetization.
 * Monotonic increasing series + oracle net (p ≈ 1.0) → WR=100%, P&L > 0.
 */
#include "test_common.h"

int test_tg2_run(void) {
    int num_raw = 100, seq = 5, h = 1;
    DataRow raw[100];
    Dataset *ds;
    DatasetSplit split;
    LSTMNetwork *net;
    BacktestConfig cfg;
    BacktestResult br;
    double *vol_preds;
    int i;

    /* Monotonic increasing: open[t] = close[t-1], always up */
    raw[0].close = 1.0; raw[0].open = 1.0;
    strcpy(raw[0].date, "2020.01.01"); strcpy(raw[0].time_, "00:00");
    raw[0].high = 1.01; raw[0].low = 0.99; raw[0].tickvol = 1000; raw[0].vol = 500; raw[0].spread = 10;
    for (i = 1; i < num_raw; i++) {
        double pc = raw[i-1].close;
        double c = pc + 0.01;
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = pc; raw[i].close = c;
        raw[i].high = c + 0.005; raw[i].low = pc - 0.005;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    split = data_split_train_test(ds, 0.5);  /* 50/50 split */
    if (!split.train || !split.test) { fprintf(stderr, "  FAIL: split\n"); data_free_dataset(ds); return 1; }

    /* Oracle: always long */
    net = make_constant_net(ds->num_features, 20.0);

    vol_preds = (double *)malloc(split.test->num_samples * sizeof(double));
    for (i = 0; i < split.test->num_samples; i++) vol_preds[i] = 0.01;

    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.0; cfg.cost_commission = 0.0;
    cfg.pip_value = 0.0001; cfg.threshold_k = 0.0;
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.02;
    cfg.prediction_threshold = 0.5;

    br = backtest_run(net, split.test, raw, num_raw,
                      split.test_start_sample, vol_preds, &cfg);

    printf("  trades=%d WR=%.1f%% P&L=%.6f\n",
           br.num_trades, br.trading.win_rate * 100.0, br.trading.total_pnl);

    ASSERT_TRUE(br.num_trades > 0);
    ASSERT_APPROX(br.trading.win_rate, 1.0, 1e-9);
    ASSERT_TRUE(br.trading.total_pnl > 0.0);

    backtest_free_result(&br);
    free(vol_preds);
    lstm_free(net);
    data_free_split(&split);
    data_free_dataset(ds);
    printf("T-G2 PASS: monotonic + oracle → WR=100%%, P&L>0\n");
    return 0;
}
