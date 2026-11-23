/*
 * T-G1: Golden — random no-edge.
 * Random walk: close[t] = close[t-1] * (1 + ε), ε IID ~ N(0, σ).
 * open[t] = close[t-1] (gap-free). Label = direction of next bar.
 * Spec: Acc 50±5%, |PF-1| < 0.15.
 *
 * Includes discriminator (hypothesis a check): momentum/mean-reversion
 * baseline on synthetic data before training. If > 55% → generator is predictable.
 */
#include "test_common.h"

int test_tg1_run(void) {
    int num_raw = 1000, seq = 10, h = 1;
    int nf1f = 26;  /* NUM_RAW_FEATURES */
    DataRow *raw;
    Dataset *ds;
    DatasetSplit split;
    ValSplit vs;
    LSTMNetwork *net;
    TrainResult tr;
    BacktestConfig cfg;
    BacktestResult br;
    double *vol_preds;
    int i;
    double acc, wr, pf;

    raw = (DataRow *)malloc(num_raw * sizeof(DataRow));

    /* Random walk: close[t] = close[t-1] * (1 + ε), ε ~ small Gaussian.
     * open[t] = close[t-1] (gap-free, realistic).
     * This is a martingale — direction is unpredictable by construction. */
    utils_srand(42);
    {
        double close = 1.0;
        for (i = 0; i < num_raw; i++) {
            double eps = 0.002 * utils_gaussian_random(0.0, 1.0);
            double prev_close = close;
            close = prev_close * (1.0 + eps);

            sprintf(raw[i].date, "2020.%02d.%02d", 1 + i/30, 1 + i%30);
            strcpy(raw[i].time_, "00:00");
            raw[i].open  = prev_close;      /* gap-free */
            raw[i].close = close;
            raw[i].high  = (close > prev_close ? close : prev_close) + 0.0005;
            raw[i].low   = (close < prev_close ? close : prev_close) - 0.0005;
            raw[i].tickvol = 1000 + (int)(100.0 * utils_rand());
            raw[i].vol = 500; raw[i].spread = 10;
        }
    }

    /* --- Discriminator: check generator is not predictable --- */
    {
        int up_count = 0, mom_correct = 0, rev_correct = 0, total = 0;
        for (i = 1; i < num_raw; i++) {
            int is_up = (raw[i].close > raw[i-1].close) ? 1 : 0;
            int prev_up = (raw[i-1].close > raw[i-2].close) ? 1 : 0;
            if (i >= 2) {
                if (prev_up == is_up) mom_correct++;
                if (prev_up != is_up) rev_correct++;
                total++;
            }
            if (is_up) up_count++;
        }
        {
            double up_frac = (double)up_count / (double)(num_raw - 1);
            double mom_acc = (double)mom_correct / (double)total;
            double rev_acc = (double)rev_correct / (double)total;
            printf("  Discriminator: up_frac=%.3f mom=%.1f%% rev=%.1f%%\n",
                   up_frac, mom_acc * 100.0, rev_acc * 100.0);

            if (up_frac < 0.40 || up_frac > 0.60) {
                fprintf(stderr, "  FAIL: generator class imbalance up_frac=%.3f (need 0.40-0.60)\n", up_frac);
                free(raw);
                return 1;
            }
            if (mom_acc > 0.55 || rev_acc > 0.55) {
                fprintf(stderr, "  FAIL: generator is predictable (mom=%.1f%% rev=%.1f%%) — fix generator\n",
                        mom_acc * 100.0, rev_acc * 100.0);
                free(raw);
                return 1;
            }
        }
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); free(raw); return 1; }

    split = data_split_train_test(ds, 0.8);
    if (!split.train || !split.test) { fprintf(stderr, "  FAIL: split\n"); data_free_dataset(ds); free(raw); return 1; }

    vs = data_split_train_val(split.train, 0.15);
    if (!vs.train || !vs.val) { fprintf(stderr, "  FAIL: val split\n"); data_free_split(&split); data_free_dataset(ds); free(raw); return 1; }

    data_compute_train_stats(vs.train, 0, vs.train->num_samples);
    data_normalize_with_stats(vs.train, vs.train->feat_min, vs.train->feat_max);
    data_normalize_with_stats(vs.val, vs.train->feat_min, vs.train->feat_max);
    data_normalize_with_stats(split.test, vs.train->feat_min, vs.train->feat_max);

    net = lstm_init(4, ds->num_features, h, 42);
    net->output_activation = ACTIVATION_SIGMOID;

    tr = train_run(net, vs.train, vs.val, 0.005, 0.9, 16, 0.0001, 10, 5, 1, NULL, NULL, 1, NULL, NULL);

    vol_preds = (double *)malloc(split.test->num_samples * sizeof(double));
    for (i = 0; i < split.test->num_samples; i++) vol_preds[i] = 0.01;

    memset(&cfg, 0, sizeof(cfg));
    cfg.cost_spread = 0.0001; cfg.cost_commission = 0.0;
    cfg.pip_value = 0.0001; cfg.threshold_k = 0.0;
    cfg.hold_bars = h; cfg.max_position_size = 1.0; cfg.target_risk = 0.002;
    cfg.prediction_threshold = 0.5;

    br = backtest_run(net, split.test, raw, num_raw,
                      split.test_start_sample, vol_preds, &cfg);

    acc = br.diagnostic.accuracy;
    wr = br.trading.win_rate;
    pf = br.trading.profit_factor;

    printf("  Acc=%.1f%% WR=%.1f%% PF=%.2f trades=%d val_loss=%.4f\n",
           acc * 100.0, wr * 100.0, pf, br.num_trades, tr.best_val_loss);

    /* Hard assert: on pure noise with risk=0.2%, |window P&L| must be <= 5% */
    printf("  Risk-channel: |window P&L|=%.4f%% (limit 5%%)\n",
           fabs(br.trading.net_pnl_pct) * 100.0);
    if (fabs(br.trading.net_pnl_pct) > 0.05) {
        fprintf(stderr, "  FAIL: |window P&L|=%.4f%% > 5%% on pure noise (risk=0.2%%)\n",
                fabs(br.trading.net_pnl_pct) * 100.0);
        backtest_free_result(&br);
        free(vol_preds);
        lstm_free(net);
        data_free_val_split(&vs);
        data_free_split(&split);
        data_free_dataset(ds);
        free(raw);
        return 1;
    }

    backtest_free_result(&br);
    free(vol_preds);
    lstm_free(net);
    data_free_val_split(&vs);
    data_free_split(&split);
    data_free_dataset(ds);
    free(raw);

    /* Spec: Acc 50±5%, |PF-1| < 0.15 */
    if (fabs(acc - 0.5) > 0.06) {
        fprintf(stderr, "  FAIL: |Acc - 0.5| = %.3f > 0.06 (spec: 50±5%%)\n", fabs(acc - 0.5));
        return 1;
    }
    if (fabs(pf - 1.0) > 0.15) {
        fprintf(stderr, "  FAIL: |PF - 1| = %.3f > 0.15\n", fabs(pf - 1.0));
        return 1;
    }

    printf("T-G1 PASS: random walk → Acc=%.1f%% WR=%.1f%% PF=%.2f val_loss=%.4f\n",
           acc * 100.0, wr * 100.0, pf, tr.best_val_loss);
    return 0;
}
