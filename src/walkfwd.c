#include "walkfwd.h"
#include "direction/data_direction.h"
#include "direction/predict_direction.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Compute log-returns from raw close prices */
static double *compute_log_returns(DataRow *raw, int num_raw) {
    double *returns;
    int i;
    if (num_raw <= 1) return NULL;
    returns = (double *)utils_safe_malloc((size_t)(num_raw - 1) * sizeof(double));
    for (i = 0; i < num_raw - 1; i++) {
        if (raw[i].close > 0.0 && raw[i + 1].close > 0.0)
            returns[i] = log(raw[i + 1].close / raw[i].close);
        else
            returns[i] = 0.0;
    }
    return returns;
}

WalkFwdResult walkfwd_run(DataRow *raw, int num_raw,
                          DataRow *sec_raw, int sec_num_raw,
                          const ParamsFile *params) {
    WalkFwdResult wfr;
    int wf_train, wf_test, wf_shift;
    int num_windows, w;
    int seq_len, pred_horizon, num_outputs;
    int num_cells;
    unsigned int base_seed;
    double val_ratio;
    int vol_mode;
    double ewma_lambda;
    char start_date[32], end_date[32];
    int start_idx, end_idx;
    double *returns;

    memset(&wfr, 0, sizeof(WalkFwdResult));

    /* Config */
    wf_train = params->has_wf_train_months ? params->wf_train_months : 24;
    wf_test = params->has_wf_test_months ? params->wf_test_months : 3;
    wf_shift = params->has_wf_shift ? params->wf_shift : 3;
    seq_len = params->has_seq ? params->seq : 10;
    pred_horizon = params->has_output ? params->output : 1;
    num_outputs = pred_horizon;
    num_cells = params->has_cells ? params->cells : 63;
    base_seed = params->has_seed ? params->seed : 42;
    val_ratio = params->has_val_ratio ? params->val_ratio : 0.12;

    if (strcmp(params->vol_mode, "ewma") == 0) vol_mode = VOL_MODE_EWMA;
    else vol_mode = VOL_MODE_GARCH;
    ewma_lambda = params->has_ewma_lambda ? params->ewma_lambda : 0.94;

    /* Determine number of windows */
    /* Start from the earliest data, slide forward */
    {
        char test_end[32];
        int total_months = wf_train + wf_test;
        int max_windows = 20;
        int actual_windows = 0;
        int *train_starts;
        int *train_ends;
        int *test_starts;
        int *test_ends;
        char cur_train_start[32];
        char cur_test_start[32];
        char cur_test_end[32];
        int i;

        train_starts = (int *)utils_safe_malloc((size_t)max_windows * sizeof(int));
        train_ends = (int *)utils_safe_malloc((size_t)max_windows * sizeof(int));
        test_starts = (int *)utils_safe_malloc((size_t)max_windows * sizeof(int));
        test_ends = (int *)utils_safe_malloc((size_t)max_windows * sizeof(int));

        /* Start from the first available date */
        strcpy(cur_train_start, raw[0].date);
        /* Add train months to get test start */
        data_add_months(cur_train_start, wf_train, cur_test_start, sizeof(cur_test_start));
        /* Add test months to get test end */
        data_add_months(cur_test_start, wf_test, cur_test_end, sizeof(cur_test_end));

        for (i = 0; i < max_windows; i++) {
            int ts_idx, te_idx;

            ts_idx = data_find_date_boundary(raw, num_raw, cur_test_start, "00:00");
            te_idx = data_find_date_boundary(raw, num_raw, cur_test_end, "00:00");

            if (te_idx >= num_raw || ts_idx >= num_raw) break;
            if (te_idx - ts_idx < seq_len + pred_horizon + 1) break;

            {
                int tr_s_idx = data_find_date_boundary(raw, num_raw, cur_train_start, "00:00");
                if (tr_s_idx < 0) tr_s_idx = 0;
                train_starts[actual_windows] = tr_s_idx;
                train_ends[actual_windows] = ts_idx;
                test_starts[actual_windows] = ts_idx;
                test_ends[actual_windows] = te_idx;
            }
            actual_windows++;

            /* Shift forward */
            data_add_months(cur_train_start, wf_shift, cur_train_start, sizeof(cur_train_start));
            data_add_months(cur_test_start, wf_shift, cur_test_start, sizeof(cur_test_start));
            data_add_months(cur_test_end, wf_shift, cur_test_end, sizeof(cur_test_end));
        }

        if (actual_windows < 1) {
            fprintf(stderr, "Error: not enough data for any walk-forward window.\n");
            free(train_starts); free(train_ends); free(test_starts); free(test_ends);
            return wfr;
        }

        num_windows = actual_windows;
        wfr.num_windows = num_windows;
        wfr.windows = (WindowResult *)utils_safe_calloc((size_t)num_windows, sizeof(WindowResult));

        printf("\n=== Walk-Forward: %d windows (train=%dmo, test=%dmo, shift=%dmo) ===\n",
               num_windows, wf_train, wf_test, wf_shift);
        printf("Data range: %s %s — %s %s (%d bars)\n",
               raw[0].date, raw[0].time_, raw[num_raw-1].date, raw[num_raw-1].time_, num_raw);
        printf("Vol mode: %s | Val ratio: %.2f | Cells: %d | Seq: %d | Horizon: %d\n\n",
               vol_mode == VOL_MODE_GARCH ? "garch" : "ewma", val_ratio,
               num_cells, seq_len, pred_horizon);

        /* Compute log-returns for volatility */
        returns = compute_log_returns(raw, num_raw);

        /* Backtest config */
        {
            BacktestConfig bt_cfg;
            bt_cfg.cost_spread = params->has_cost_spread ? params->cost_spread : 0.0001;
            bt_cfg.cost_commission = params->has_cost_commission ? params->cost_commission : 0.0;
            bt_cfg.pip_value = 0.0001;
            bt_cfg.threshold_k = params->has_threshold_k ? params->threshold_k : 0.0;
            bt_cfg.hold_bars = pred_horizon;   /* h=1 for baseline measurement */
            bt_cfg.max_position_size = 1.0;
            bt_cfg.target_risk = 0.002;

            for (w = 0; w < num_windows; w++) {
                int tr_s, tr_e, te_s, te_e;
                int window_num_rows;
                DataRow *window_raw;
                Dataset *full_ds;
                DatasetSplit split;
                ValSplit vs;
                double *train_min, *train_max;
                int train_count, val_count;
                int j;
                LSTMNetwork *net;
                TrainResult tr;
                double *vol_preds;
                double *vol_sigma;  /* predictive sigma per window bar */
                GarchParams garch;
                BacktestResult bt_result;
                int vol_fit_end;
                char period_str[64];

                tr_s = train_starts[w];
                tr_e = train_ends[w];
                te_s = test_starts[w];
                te_e = test_ends[w];

                wfr.windows[w].window_num = w + 1;
                strcpy(wfr.windows[w].train_start, raw[tr_s].date);
                strcpy(wfr.windows[w].train_end, raw[tr_e].date);
                strcpy(wfr.windows[w].test_start, raw[te_s].date);
                strcpy(wfr.windows[w].test_end, raw[te_e].date);

                snprintf(period_str, sizeof(period_str), "%s-%s", raw[te_s].date, raw[te_e].date);

                printf("--- Window %d/%d: train %s..%s (%d bars) | test %s..%s (%d bars) ---\n",
                       w + 1, num_windows,
                       raw[tr_s].date, raw[tr_e > 0 ? tr_e - 1 : 0].date, tr_e - tr_s,
                       raw[te_s].date, raw[te_e > 0 ? te_e - 1 : 0].date, te_e - te_s);

                /* Extract window raw data (train + test) */
                window_num_rows = te_e - tr_s;
                window_raw = (DataRow *)utils_safe_malloc((size_t)window_num_rows * sizeof(DataRow));
                memcpy(window_raw, raw + tr_s, (size_t)window_num_rows * sizeof(DataRow));

                /* Build dataset (direction mode, 26 features) */
                full_ds = data_build_dataset_direction(window_raw, window_num_rows,
                                                        seq_len, pred_horizon);
                if (!full_ds) {
                    fprintf(stderr, "Window %d: failed to build dataset, skipping.\n", w + 1);
                    free(window_raw);
                    continue;
                }

                /* --- Fit volatility on TRAIN returns only (no leak) --- */
                {
                    int window_ret_len = window_num_rows - 1;
                    int train_ret_len = (tr_e - tr_s) - 1;
                    double *sigma2;
                    int bi;

                    if (vol_mode == VOL_MODE_GARCH) {
                        garch = vol_garch_fit(returns + tr_s, train_ret_len, 300);
                        sigma2 = vol_garch_series(returns + tr_s, window_ret_len, &garch);
                    } else {
                        garch.omega = 0.0;
                        garch.alpha = 1.0 - ewma_lambda;
                        garch.beta = ewma_lambda;
                        garch.sigma2_init = 0.0;
                        sigma2 = vol_ewma(returns + tr_s, window_ret_len, ewma_lambda);
                    }

                    /* vol_sigma[t] = predictive sigma for bar t
                     * (sigma2[t] is conditional on returns up to t-1, predicts bar t) */
                    /* vol_sigma[t] = predictive sigma for bar t.
                     * sigma2[t] is conditional on returns up to t-1, so it is causal
                     * at the open of bar t. Use sigma2[bi] directly (not bi-1). */
                    vol_sigma = (double *)utils_safe_malloc((size_t)window_num_rows * sizeof(double));
                    for (bi = 0; bi < window_num_rows; bi++) {
                        if (bi < window_ret_len)
                            vol_sigma[bi] = sqrt(fabs(sigma2[bi]));
                        else
                            vol_sigma[bi] = sqrt(fabs(sigma2[window_ret_len - 1]));
                    }

                    /* Append vol as feature 27 to the dataset (all samples/timesteps) */
                    data_append_vol_feature(full_ds, vol_sigma, window_num_rows, "pred_vol");

                    if (vol_mode == VOL_MODE_GARCH) {
                        printf("  GARCH(1,1): omega=%.6e  alpha=%.4f  beta=%.4f  (a+b=%.4f)\n",
                               garch.omega, garch.alpha, garch.beta, garch.alpha + garch.beta);
                    } else {
                        printf("  EWMA: lambda=%.4f\n", ewma_lambda);
                    }
                    printf("  Vol feature appended: %d features -> %d\n",
                           full_ds->num_features - 1, full_ds->num_features);

                    free(sigma2);
                    /* vol_sigma kept for vol_preds construction after split */
                }

                /* Split into train+test — use explicit ratio, not global */
                split = data_split_train_test(full_ds, (double)(tr_e - tr_s) / (double)(te_e - tr_s));

                train_count = split.train->num_samples;

                /* Construct vol_preds aligned to test samples.
                 * Test sample i → full_ds sample (test_start_sample + i).
                 * Entry bar = test_start_sample + i + seq_len.
                 * Vol known at last feature bar = test_start_sample + i + seq_len - 1.
                 * vol_preds[i] = vol_sigma[test_start_sample + i + seq_len - 1] */
                {
                    int test_ns = split.test->num_samples;
                    int bi;
                    vol_preds = (double *)utils_safe_malloc((size_t)test_ns * sizeof(double));
                    for (bi = 0; bi < test_ns; bi++) {
                        int bar_idx = split.test_start_sample + bi + seq_len - 1;
                        if (bar_idx >= 0 && bar_idx < window_num_rows)
                            vol_preds[bi] = vol_sigma[bar_idx];
                        else
                            vol_preds[bi] = vol_sigma[window_num_rows - 1];
                    }
                }
                free(vol_sigma);

                /* Split train into train/val */
                vs = data_split_train_val(split.train, val_ratio);
                val_count = vs.val->num_samples;

                printf("  Samples: train=%d  val=%d  test=%d  features=%d (incl. pred_vol)\n",
                       vs.train->num_samples, val_count, split.test->num_samples,
                       full_ds->num_features);

                /* Compute normalization stats on train ONLY (excluding val) */
                train_min = (double *)utils_safe_malloc((size_t)full_ds->num_features * sizeof(double));
                train_max = (double *)utils_safe_malloc((size_t)full_ds->num_features * sizeof(double));
                {
                    int j;
                    for (j = 0; j < full_ds->num_features; j++) {
                        train_min[j] = vs.train->feat_min[j];
                        train_max[j] = vs.train->feat_max[j];
                    }
                }
                data_compute_train_stats(vs.train, 0, vs.train->num_samples);
                for (j = 0; j < full_ds->num_features; j++) {
                    train_min[j] = vs.train->feat_min[j];
                    train_max[j] = vs.train->feat_max[j];
                }

                /* Normalize all pieces with train stats */
                data_normalize_with_stats(vs.train, train_min, train_max);
                data_normalize_with_stats(vs.val, train_min, train_max);
                data_normalize_with_stats(split.test, train_min, train_max);
                printf("  Normalization: train-only stats applied to train/val/test\n");

                /* Initialize LSTM with per-window seed */
                net = lstm_init(num_cells, full_ds->num_features, num_outputs,
                                base_seed + (unsigned int)w);
                net->output_activation = ACTIVATION_SIGMOID;
                if (params->has_clip_min) net->clip_min = params->clip_min;
                if (params->has_clip_max) net->clip_max = params->clip_max;
                if (params->has_forget_gate_bias_init) net->forget_gate_bias_init = params->forget_gate_bias_init;
                printf("  Training (seed=%u, patience=%d)...\n",
                       base_seed + (unsigned int)w,
                       params->has_patience ? params->patience : 8);

                /* Train — pass model path so best weights are saved on each val improvement */
                {
                    double lr = params->has_lr ? params->lr : 0.005;
                    double momentum = params->has_momentum ? params->momentum : 0.9;
                    int batch_size = params->has_batch_size ? params->batch_size : 32;
                    double wd = params->has_wd ? params->wd : 0.0001;
                    int epochs = params->has_epochs ? params->epochs : 60;
                    int patience = params->has_patience ? params->patience : 8;
                    char model_path[256];

                    snprintf(model_path, sizeof(model_path), "model_wf_window%d.txt", w + 1);

                    tr = train_run(net, vs.train, vs.val,
                                   lr, momentum, batch_size, wd,
                                   epochs, patience, w + 1, model_path);
                }

                wfr.windows[w].train_result = tr;
                printf("  Training done: best_epoch=%d  best_val_loss=%.6f  epochs_actual=%d\n",
                       tr.best_epoch, tr.best_val_loss, tr.epochs_actual);

                /* Backtest on test — pass test_start_sample (train_count + purge) as offset.
                 * Using train_count alone misses the purge gap and causes look-ahead bias. */
                bt_result = backtest_run(net, split.test, window_raw, window_num_rows,
                                         split.test_start_sample, vol_preds, &bt_cfg);
                wfr.windows[w].backtest_result = bt_result;
                printf("  Backtest: %d trades | PF=%.2f | Sharpe=%.2f | MaxDD=%.1f%% | P&L=%.2f%%\n",
                       bt_result.num_trades,
                       bt_result.trading.profit_factor,
                       bt_result.trading.sharpe_annual,
                       bt_result.trading.max_drawdown_pct * 100.0,
                       bt_result.trading.net_pnl_pct * 100.0);

                /* Print window results */
                metrics_print_window(w + 1, period_str,
                                     &bt_result.trading, &bt_result.diagnostic);
                printf("\n");

                /* Cleanup per window */
                if (vol_preds) vol_free(vol_preds);
                lstm_free(net);
                free(train_min);
                free(train_max);
                data_free_val_split(&vs);
                data_free_split(&split);
                data_free_dataset(full_ds);
                free(window_raw);

                /* Check for Ctrl+C interrupt — stop after completing current window */
                if (g_interrupted) {
                    printf("\n=== Interrupted after window %d/%d ===\n", w + 1, num_windows);
                    num_windows = w + 1;
                    wfr.num_windows = num_windows;
                    break;
                }
            }

            /* Aggregate metrics */
            {
                TradingMetrics *tm_arr;
                int j;
                double sum_pf = 0, sum_sharpe = 0, sum_dd = 0, sum_pnl = 0, sum_wr = 0;
                int total_trades = 0;
                int worst_idx = 0;
                double worst_pf = 1e30;

                tm_arr = (TradingMetrics *)utils_safe_malloc((size_t)num_windows * sizeof(TradingMetrics));
                for (j = 0; j < num_windows; j++) {
                    tm_arr[j] = wfr.windows[j].backtest_result.trading;
                    sum_pf += tm_arr[j].profit_factor;
                    sum_sharpe += tm_arr[j].sharpe_annual;
                    sum_dd += tm_arr[j].max_drawdown_pct;
                    sum_pnl += tm_arr[j].net_pnl_pct;
                    sum_wr += tm_arr[j].win_rate;
                    total_trades += tm_arr[j].num_trades;
                    if (tm_arr[j].profit_factor < worst_pf) {
                        worst_pf = tm_arr[j].profit_factor;
                        worst_idx = j;
                    }
                }

                wfr.aggregate.profit_factor = sum_pf / num_windows;
                wfr.aggregate.sharpe_annual = sum_sharpe / num_windows;
                wfr.aggregate.max_drawdown_pct = sum_dd / num_windows;
                wfr.aggregate.net_pnl_pct = sum_pnl / num_windows;
                wfr.aggregate.win_rate = sum_wr / num_windows;
                wfr.aggregate.num_trades = total_trades;

                wfr.worst = tm_arr[worst_idx];
                wfr.criteria_met = metrics_check_criteria(&wfr.worst);

                free(tm_arr);
            }

            free(returns);
        }

        free(train_starts); free(train_ends); free(test_starts); free(test_ends);
    }

    return wfr;
}

void walkfwd_free(WalkFwdResult *result) {
    int i;
    if (!result) return;
    if (result->windows) {
        for (i = 0; i < result->num_windows; i++) {
            backtest_free_result(&result->windows[i].backtest_result);
        }
        free(result->windows);
    }
    memset(result, 0, sizeof(WalkFwdResult));
}

void walkfwd_print_report(const WalkFwdResult *result) {
    int i;

    printf("\n=== WALK-FORWARD REPORT ===\n\n");
    printf("Window results:\n");
    for (i = 0; i < result->num_windows; i++) {
        char period_str[64];
        snprintf(period_str, sizeof(period_str), "%s-%s",
                 result->windows[i].test_start, result->windows[i].test_end);
        metrics_print_window(i + 1, period_str,
                             &result->windows[i].backtest_result.trading,
                             &result->windows[i].backtest_result.diagnostic);
    }

    {
        TradingMetrics *tm_arr;
        tm_arr = (TradingMetrics *)utils_safe_malloc((size_t)result->num_windows * sizeof(TradingMetrics));
        for (i = 0; i < result->num_windows; i++) {
            tm_arr[i] = result->windows[i].backtest_result.trading;
        }
        metrics_print_aggregate(tm_arr, result->num_windows);
        metrics_print_worst(tm_arr, result->num_windows);
        free(tm_arr);
    }

    metrics_print_verdict(&result->worst);
}
