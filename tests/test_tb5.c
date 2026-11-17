/*
 * T-B5: Entry/exit alignment.
 * Assert: entry_price == raw[entry_bar].open,
 *         exit_price == raw[entry_bar + hold_bars].open,
 *         entry_bar == raw_offset + i + seq_len.
 */
#include "test_common.h"

int test_tb5_run(void) {
    int num_raw = 30, seq = 5, h = 1, raw_offset = 3;
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

    net = make_constant_net(ds->num_features, 20.0);  /* p ≈ 1.0 → long */

    vol_preds = (double *)malloc(ds->num_samples * sizeof(double));
    for (i = 0; i < ds->num_samples; i++) vol_preds[i] = 0.01;

    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.0; cfg.cost_commission = 0.0;
    cfg.pip_value = 0.0001; cfg.threshold_k = 0.0;
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.02;

    br = backtest_run(net, ds, raw, num_raw, raw_offset, vol_preds, &cfg);

    printf("  trades=%d\n", br.num_trades);

    for (i = 0; i < br.num_trades; i++) {
        int exp_entry_bar = raw_offset + i + seq;
        int exp_exit_bar  = exp_entry_bar + cfg.hold_bars;

        ASSERT_INT(br.trades[i].entry_bar, exp_entry_bar);
        ASSERT_INT(br.trades[i].exit_bar, exp_exit_bar);
        ASSERT_APPROX(br.trades[i].entry_price, raw[exp_entry_bar].open, 1e-15);
        ASSERT_APPROX(br.trades[i].exit_price,  raw[exp_exit_bar].open,  1e-15);
    }

    backtest_free_result(&br);
    free(vol_preds);
    lstm_free(net);
    data_free_dataset(ds);
    printf("T-B5 PASS: entry=raw[offset+i+seq].open, exit=raw[entry+hold].open, bars correct\n");
    return 0;
}
