/*
 * T-B6: Label–PnL consistency (close-to-close).
 * With cost=0, the label at output k=hold_bars-1 exactly predicts trade direction:
 *   label = (close[entry_bar + hold_bars] > close[entry_bar]) ? 1 : 0
 *   pnl   = direction * (close[exit_bar] - close[entry_bar]) / close[entry_bar]
 * For a long (direction=1): pnl > 0 iff label == 1.
 * For a short (direction=-1): pnl > 0 iff label == 0.
 * Tested for h=1 (single output) and h=3 (three outputs, k=2 used).
 */
#include "test_common.h"

/* Build alternating up/down raw data */
static void build_alt_data(DataRow *raw, int num_raw) {
    int i;
    raw[0].close = 1.0; raw[0].open = 1.0;
    strcpy(raw[0].date, "2020.01.01"); strcpy(raw[0].time_, "00:00");
    raw[0].high = 1.01; raw[0].low = 0.99; raw[0].tickvol = 1000; raw[0].vol = 500; raw[0].spread = 10;
    for (i = 1; i < num_raw; i++) {
        double prev_close = raw[i-1].close;
        double c = (i % 2 == 0) ? prev_close + 0.01 : prev_close - 0.01;
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = prev_close;
        raw[i].close = c;
        raw[i].high = (c > prev_close) ? c + 0.005 : prev_close + 0.005;
        raw[i].low = (c < prev_close) ? c - 0.005 : prev_close - 0.005;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }
}

/* Run oracle test for a given horizon h */
static int run_oracle_test(int h, int seq, int num_raw) {
    DataRow *raw;
    Dataset *ds;
    LSTMNetwork *net_long, *net_short;
    BacktestConfig cfg;
    BacktestResult br;
    double *vol_preds;
    int i;

    raw = (DataRow *)malloc(num_raw * sizeof(DataRow));
    build_alt_data(raw, num_raw);

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL(h=%d): cannot build dataset\n", h); free(raw); return 1; }

    vol_preds = (double *)malloc(ds->num_samples * sizeof(double));
    for (i = 0; i < ds->num_samples; i++) vol_preds[i] = 0.01;

    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.0; cfg.cost_commission = 0.0;
    cfg.pip_value = 0.0001; cfg.threshold_k = 0.0;
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.02;
    cfg.prediction_threshold = 0.5;

    /* Test 1: always-long oracle (p ~ 1.0 at output k=h-1) */
    net_long = make_constant_net_ext(ds->num_features, 20.0, h);
    br = backtest_run(net_long, ds, raw, num_raw, 0, vol_preds, &cfg);

    printf("  h=%d Long oracle: %d trades\n", h, br.num_trades);
    for (i = 0; i < br.num_trades; i++) {
        int ebar = br.trades[i].entry_bar;
        int label = (raw[ebar + h].close > raw[ebar].close) ? 1 : 0;
        double pnl = br.trades[i].pnl_fraction;
        if (label == 1 && pnl <= 0.0) {
            fprintf(stderr, "  FAIL(h=%d): trade %d label=1 but pnl=%.6f (close %.5f vs %.5f)\n",
                    h, i, pnl, raw[ebar + h].close, raw[ebar].close);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_long); data_free_dataset(ds); free(raw);
            return 1;
        }
        if (label == 0 && pnl >= 0.0) {
            fprintf(stderr, "  FAIL(h=%d): trade %d label=0 but pnl=%.6f (close %.5f vs %.5f)\n",
                    h, i, pnl, raw[ebar + h].close, raw[ebar].close);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_long); data_free_dataset(ds); free(raw);
            return 1;
        }
    }
    backtest_free_result(&br);
    lstm_free(net_long);
    printf("  h=%d Long oracle: pnl > 0 iff label == 1 — OK\n", h);

    /* Test 2: always-short oracle (p ~ 0.0 at output k=h-1) */
    net_short = make_constant_net_ext(ds->num_features, -20.0, h);
    br = backtest_run(net_short, ds, raw, num_raw, 0, vol_preds, &cfg);

    printf("  h=%d Short oracle: %d trades\n", h, br.num_trades);
    for (i = 0; i < br.num_trades; i++) {
        int ebar = br.trades[i].entry_bar;
        int label = (raw[ebar + h].close > raw[ebar].close) ? 1 : 0;
        double pnl = br.trades[i].pnl_fraction;
        if (label == 0 && pnl <= 0.0) {
            fprintf(stderr, "  FAIL(h=%d): short trade %d label=0 but pnl=%.6f\n", h, i, pnl);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_short); data_free_dataset(ds); free(raw);
            return 1;
        }
        if (label == 1 && pnl >= 0.0) {
            fprintf(stderr, "  FAIL(h=%d): short trade %d label=1 but pnl=%.6f\n", h, i, pnl);
            backtest_free_result(&br); free(vol_preds); lstm_free(net_short); data_free_dataset(ds); free(raw);
            return 1;
        }
    }
    backtest_free_result(&br);
    lstm_free(net_short);
    printf("  h=%d Short oracle: pnl > 0 iff label == 0 — OK\n", h);

    free(vol_preds);
    data_free_dataset(ds);
    free(raw);
    return 0;
}

int test_tb6_run(void) {
    /* h=1: single output, basic close-to-close consistency */
    if (run_oracle_test(1, 5, 50) != 0) return 1;

    /* h=3: three outputs, backtest uses k=2 (hold_bars-1) */
    if (run_oracle_test(3, 5, 60) != 0) return 1;

    printf("T-B6 PASS: label–PnL consistency for h=1 and h=3 — close-to-close matches label\n");
    return 0;
}
