/*
 * T-B6: Label–PnL consistency (close-to-close).
 * With cost=0 and hold_bars=1, the label exactly predicts trade direction:
 *   label = (close[entry_bar] > close[entry_bar-1]) ? 1 : 0
 *   pnl   = direction * (close[entry_bar] - close[entry_bar-1]) / close[entry_bar-1]
 * For a long (direction=1): pnl > 0 iff label == 1.
 * For a short (direction=-1): pnl > 0 iff label == 0.
 * This test would FAIL with open-to-open execution — that's the point.
 */
#include "test_common.h"

int test_tb6_run(void) {
    int num_raw = 50, seq = 5, h = 1, raw_offset = 0;
    DataRow raw[50];
    Dataset *ds;
    LSTMNetwork *net_long, *net_short;
    BacktestConfig cfg;
    BacktestResult br;
    double *vol_preds;
    int i;

    /* Alternating up/down: even bars close up, odd bars close down */
    raw[0].close = 1.0; raw[0].open = 1.0;
    strcpy(raw[0].date, "2020.01.01"); strcpy(raw[0].time_, "00:00");
    raw[0].high = 1.01; raw[0].low = 0.99; raw[0].tickvol = 1000; raw[0].vol = 500; raw[0].spread = 10;
    for (i = 1; i < num_raw; i++) {
        double prev_close = raw[i-1].close;
        double c = (i % 2 == 0) ? prev_close + 0.01 : prev_close - 0.01;
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = prev_close;  /* open = previous close (gap-free) */
        raw[i].close = c;
        raw[i].high = (c > prev_close) ? c + 0.005 : prev_close + 0.005;
        raw[i].low = (c < prev_close) ? c - 0.005 : prev_close - 0.005;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    vol_preds = (double *)malloc(ds->num_samples * sizeof(double));
    for (i = 0; i < ds->num_samples; i++) vol_preds[i] = 0.01;

    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.0; cfg.cost_commission = 0.0;
    cfg.pip_value = 0.0001; cfg.threshold_k = 0.0;
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.02;

    /* Test 1: always-long oracle (p ≈ 1.0) */
    net_long = make_constant_net(ds->num_features, 20.0);
    br = backtest_run(net_long, ds, raw, num_raw, raw_offset, vol_preds, &cfg);

    printf("  Long oracle: %d trades\n", br.num_trades);
    for (i = 0; i < br.num_trades; i++) {
        int ebar = br.trades[i].entry_bar;
        int label = (raw[ebar].close > raw[ebar - 1].close) ? 1 : 0;
        double pnl = br.trades[i].pnl_fraction;
        if (label == 1 && pnl <= 0.0) {
            fprintf(stderr, "  FAIL: trade %d label=1 but pnl=%.6f (close %.5f vs %.5f)\n",
                    i, pnl, raw[ebar].close, raw[ebar-1].close);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_long); data_free_dataset(ds);
            return 1;
        }
        if (label == 0 && pnl >= 0.0) {
            fprintf(stderr, "  FAIL: trade %d label=0 but pnl=%.6f (close %.5f vs %.5f)\n",
                    i, pnl, raw[ebar].close, raw[ebar-1].close);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_long); data_free_dataset(ds);
            return 1;
        }
    }
    backtest_free_result(&br);
    lstm_free(net_long);
    printf("  Long oracle: pnl > 0 iff label == 1 — OK\n");

    /* Test 2: always-short oracle (p ≈ 0.0) */
    net_short = make_constant_net(ds->num_features, -20.0);
    br = backtest_run(net_short, ds, raw, num_raw, raw_offset, vol_preds, &cfg);

    printf("  Short oracle: %d trades\n", br.num_trades);
    for (i = 0; i < br.num_trades; i++) {
        int ebar = br.trades[i].entry_bar;
        int label = (raw[ebar].close > raw[ebar - 1].close) ? 1 : 0;
        double pnl = br.trades[i].pnl_fraction;
        /* Short: direction = -1, pnl > 0 iff label == 0 */
        if (label == 0 && pnl <= 0.0) {
            fprintf(stderr, "  FAIL: short trade %d label=0 but pnl=%.6f\n", i, pnl);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_short); data_free_dataset(ds);
            return 1;
        }
        if (label == 1 && pnl >= 0.0) {
            fprintf(stderr, "  FAIL: short trade %d label=1 but pnl=%.6f\n", i, pnl);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_short); data_free_dataset(ds);
            return 1;
        }
    }
    backtest_free_result(&br);
    lstm_free(net_short);
    printf("  Short oracle: pnl > 0 iff label == 0 — OK\n");

    free(vol_preds);
    data_free_dataset(ds);
    printf("T-B6 PASS: label–PnL consistency — close-to-close matches label exactly\n");
    return 0;
}
