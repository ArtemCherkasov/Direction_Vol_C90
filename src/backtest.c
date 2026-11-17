#include "backtest.h"
#include "train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static LSTMCellState *alloc_cell_states_bt(int seq_len, int num_cells, int num_features) {
    LSTMCellState *cs;
    int t;
    int combined = num_cells + num_features;

    cs = (LSTMCellState *)utils_safe_calloc((size_t)seq_len, sizeof(LSTMCellState));
    for (t = 0; t < seq_len; t++) {
        cs[t].hidden_state = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].cell_state = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].forget_gate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].input_gate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].output_gate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].candidate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].concatenated = (double *)utils_safe_calloc((size_t)combined, sizeof(double));
    }
    return cs;
}

BacktestResult backtest_run(LSTMNetwork *net,
                             Dataset *test_ds,
                             DataRow *raw, int num_raw,
                             int raw_offset,
                             const double *vol_predictions,
                             const BacktestConfig *config) {
    BacktestResult result;
    int i, seq_len, last_base;
    double *preds;
    LSTMCellState *cell_states;
    int max_trades;
    int n_trades;
    double *all_preds;
    double *all_targets;
    int n = net->num_cells;
    int combined = n + net->num_features;

    memset(&result, 0, sizeof(BacktestResult));

    seq_len = test_ds->sequence_length;
    last_base = (seq_len - 1) * net->num_outputs;

    cell_states = alloc_cell_states_bt(seq_len, n, net->num_features);
    preds = (double *)utils_safe_calloc((size_t)(net->num_outputs * seq_len), sizeof(double));

    max_trades = test_ds->num_samples;
    result.trades = (Trade *)utils_safe_calloc((size_t)max_trades, sizeof(Trade));
    n_trades = 0;

    all_preds = (double *)utils_safe_calloc((size_t)test_ds->num_samples, sizeof(double));
    all_targets = (double *)utils_safe_calloc((size_t)test_ds->num_samples, sizeof(double));

    for (i = 0; i < test_ds->num_samples; i++) {
        double p;
        double entry_price, exit_price;
        int entry_bar, exit_bar;
        double cost_fraction, vol_i;
        int direction = 0;
        double position_size;

        /* Forward pass — one signal per bar (last timestep only) */
        lstm_forward(net, test_ds->features[i], cell_states, seq_len, preds);
        p = preds[last_base];

        all_preds[i] = p;
        all_targets[i] = test_ds->labels[i][last_base];

        /* Map test sample i to raw bar index.
         * Test sample i → full_ds sample (raw_offset + i).
         * Full_ds sample j has last feature bar at j + seq_len - 1.
         * Target predicts: close[j+seq_len] vs close[j+seq_len-1].
         * Entry at open of bar j+seq_len (the predicted bar).
         * Exit at open of bar j+seq_len+hold_bars. */
        entry_bar = raw_offset + i + seq_len;
        if (entry_bar >= num_raw) continue;

        vol_i = vol_predictions[i];
        if (vol_i < 1e-9) continue;  /* skip bars with zero/near-zero vol (holiday, flat) */

        /* Scale vol for hold horizon: σ_hold = σ_daily × √hold_bars */
        if (config->hold_bars > 1) {
            vol_i *= sqrt((double)config->hold_bars);
        }

        /* Cost calculation — cost_spread is a direct fraction (e.g. 0.0001 = 1bp).
         * cost_fraction = spread + 2*commission (per round-trip). */
        entry_price = raw[entry_bar].open;
        cost_fraction = config->cost_spread + 2.0 * config->cost_commission;

        /* Vol cost filter: skip trade if expected movement (k·σ̂) doesn't cover costs.
         * threshold_k == 0 disables the filter entirely.
         * Fixed p-threshold at 0.5 — no dynamic offset. */
        if (config->threshold_k > 0.0 && config->threshold_k * vol_i < cost_fraction)
            continue;  /* vol too low to cover costs */

        /* Signal — fixed threshold at 0.5 */
        if (p >= 0.5) {
            direction = 1;  /* long */
        } else {
            direction = -1; /* short */
        }

        /* Exit */
        exit_bar = entry_bar + config->hold_bars;
        if (exit_bar >= num_raw) continue;
        exit_price = raw[exit_bar].open;

        /* Position sizing (vol targeting) */
        position_size = config->target_risk / vol_i;
        if (position_size > config->max_position_size) position_size = config->max_position_size;
        if (position_size < 0.0) position_size = 0.0;

        /* Record trade */
        result.trades[n_trades].bar_index = i;
        result.trades[n_trades].entry_bar = entry_bar;
        result.trades[n_trades].exit_bar = exit_bar;
        result.trades[n_trades].direction = direction;
        result.trades[n_trades].entry_price = entry_price;
        result.trades[n_trades].exit_price = exit_price;
        result.trades[n_trades].position_size = position_size;
        result.trades[n_trades].pnl_fraction =
            direction * (exit_price - entry_price) / entry_price * position_size
            - cost_fraction * position_size;

        n_trades++;

        /* Dump trade for diagnostics */
        printf("%s,%d,%.4f,%d,%d,%.6f,%.6f\n",
               raw[entry_bar].date,
               (int)all_targets[i], p, direction,
               (result.trades[n_trades - 1].pnl_fraction > 0.0),
               raw[entry_bar].close - raw[entry_bar - 1].close,
               raw[exit_bar].open - raw[entry_bar].open);
    }

    result.num_trades = n_trades;

    /* Diagnostic: print p distribution for this window */
    {
        double p_min = 1e30, p_max = -1e30, p_sum = 0.0;
        int n_valid = test_ds->num_samples;
        int k;
        for (k = 0; k < n_valid; k++) {
            double pv = all_preds[k];
            if (pv < p_min) p_min = pv;
            if (pv > p_max) p_max = pv;
            p_sum += pv;
        }
        if (n_valid > 0) {
            /* Simple quantiles: sort copy and pick percentiles */
            double *sorted = (double *)utils_safe_malloc((size_t)n_valid * sizeof(double));
            memcpy(sorted, all_preds, (size_t)n_valid * sizeof(double));
            /* Insertion sort (n_valid is small, ~60) */
            {
                int a, b; double tmp;
                for (a = 1; a < n_valid; a++) {
                    tmp = sorted[a]; b = a - 1;
                    while (b >= 0 && sorted[b] > tmp) { sorted[b+1] = sorted[b]; b--; }
                    sorted[b+1] = tmp;
                }
            }
            printf("  p distribution: min=%.4f  q25=%.4f  med=%.4f  q75=%.4f  max=%.4f  mean=%.4f  n=%d\n",
                   p_min,
                   sorted[n_valid / 4],
                   sorted[n_valid / 2],
                   sorted[3 * n_valid / 4],
                   p_max,
                   p_sum / n_valid,
                   n_valid);
            free(sorted);
        }
    }

    /* Compute metrics */
    {
        double ann_factor = 252.0; /* daily bars per year */
        result.trading = metrics_compute(result.trades, n_trades, ann_factor);
    }

    /* CHECK-print: verify window P&L equals sum of pnl_fraction */
    {
        double sum_pnl = 0.0;
        int ti;
        for (ti = 0; ti < n_trades; ti++) sum_pnl += result.trades[ti].pnl_fraction;
        printf("  CHECK: window P&L=%.6f  sum(pnl_fraction)=%.6f  %s\n",
               result.trading.net_pnl_pct, sum_pnl,
               fabs(result.trading.net_pnl_pct - sum_pnl) < 1e-9 ? "OK" : "MISMATCH");
    }
    result.diagnostic = metrics_diagnostic(all_preds, all_targets, test_ds->num_samples);

    free(preds);
    free(all_preds);
    free(all_targets);
    lstm_free_cell_states(cell_states, seq_len);

    return result;
}

void backtest_free_result(BacktestResult *result) {
    if (!result) return;
    if (result->trades) free(result->trades);
    result->trades = NULL;
    result->num_trades = 0;
}
