/*
 * T-G3: Golden — final honest run.
 * Run walkfwd on real data with minimal config (2 windows, 5 epochs).
 * After fixing column offset bug: expect Acc 51-55%, |Acc - WR| <= 2pp.
 * Currently will likely show anomalous results due to column offset (see T-D4).
 */
#include "test_common.h"

int test_tg3_run(void) {
    const char *path = "../resources/EURUSD_Daily_200301010000_202611160000.csv";
    DataRow *raw = NULL;
    int num_raw = 0, rc;
    ParamsFile pf;
    WalkFwdResult wfr;
    int w;
    double mean_acc = 0.0, mean_wr = 0.0;
    int n_valid = 0;

    rc = data_load_raw(path, &raw, &num_raw);
    if (rc != 0) {
        fprintf(stderr, "  SKIP: cannot load %s\n", path);
        return 0;
    }

    /* Minimal walk-forward config */
    memset(&pf, 0, sizeof(pf));
    pf.has_wf_train_months = 1; pf.wf_train_months = 60;  /* 5 years train */
    pf.has_wf_test_months = 1; pf.wf_test_months = 12;    /* 1 year test */
    pf.has_wf_shift = 1; pf.wf_shift = 12;
    pf.has_seq = 1; pf.seq = 10;
    pf.has_output = 1; pf.output = 1;
    pf.has_cells = 1; pf.cells = 16;
    pf.has_seed = 1; pf.seed = 42;
    pf.has_epochs = 1; pf.epochs = 5;
    pf.has_patience = 1; pf.patience = 3;
    pf.has_lr = 1; pf.lr = 0.005;
    pf.has_batch_size = 1; pf.batch_size = 16;
    pf.has_momentum = 1; pf.momentum = 0.9;
    pf.has_wd = 1; pf.wd = 0.0001;
    pf.has_val_ratio = 1; pf.val_ratio = 0.15;
    pf.has_cost_spread = 1; pf.cost_spread = 1.0;
    pf.has_cost_commission = 1; pf.cost_commission = 0.2;
    pf.has_threshold_k = 1; pf.threshold_k = 2.0;
    strcpy(pf.vol_mode, "ewma"); pf.has_vol_mode = 1;
    pf.has_ewma_lambda = 1; pf.ewma_lambda = 0.94;

    wfr = walkfwd_run(raw, num_raw, NULL, 0, &pf);

    for (w = 0; w < wfr.num_windows; w++) {
        double acc = wfr.windows[w].backtest_result.diagnostic.accuracy;
        double wr = wfr.windows[w].backtest_result.trading.win_rate;
        if (wfr.windows[w].backtest_result.num_trades > 0) {
            mean_acc += acc;
            mean_wr += wr;
            n_valid++;
            printf("  Window %d: Acc=%.1f%% WR=%.1f%% trades=%d\n",
                   w + 1, acc * 100.0, wr * 100.0,
                   wfr.windows[w].backtest_result.num_trades);
        }
    }

    walkfwd_free(&wfr);
    free(raw);

    if (n_valid == 0) {
        fprintf(stderr, "  SKIP: no valid windows with trades\n");
        return 0;
    }

    mean_acc /= n_valid;
    mean_wr /= n_valid;
    printf("  Mean: Acc=%.1f%% WR=%.1f%% |Acc-WR|=%.1fpp\n",
           mean_acc * 100.0, mean_wr * 100.0, fabs(mean_acc - mean_wr) * 100.0);

    /* Check: |Acc - WR| <= 2pp */
    if (fabs(mean_acc - mean_wr) > 0.02) {
        fprintf(stderr, "  FAIL: |Acc-WR| = %.1fpp > 2pp — mismatch between prediction and trading\n",
                fabs(mean_acc - mean_wr) * 100.0);
        fprintf(stderr, "  Note: if closes look like tickvols → column offset bug (see T-D4)\n");
        return 1;
    }

    printf("T-G3 PASS: %d windows, Acc=%.1f%% WR=%.1f%% |Acc-WR|=%.1fpp\n",
           n_valid, mean_acc * 100.0, mean_wr * 100.0, fabs(mean_acc - mean_wr) * 100.0);
    return 0;
}
