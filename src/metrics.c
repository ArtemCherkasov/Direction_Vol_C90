#include "metrics.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

TradingMetrics metrics_compute(const Trade *trades, int num_trades,
                                double annualization_factor) {
    TradingMetrics tm;
    double sum_wins = 0.0, sum_losses = 0.0;
    double cumulative = 0.0, peak = 0.0, max_dd = 0.0;
    double mean_pnl, std_pnl, sum_pnl = 0.0, sum_pnl2 = 0.0;
    int num_wins = 0, i;

    memset(&tm, 0, sizeof(TradingMetrics));
    tm.num_trades = num_trades;

    if (num_trades <= 0) {
        tm.profit_factor = 0.0;
        tm.sharpe_annual = 0.0;
        return tm;
    }

    for (i = 0; i < num_trades; i++) {
        double pnl = trades[i].pnl_fraction;
        tm.total_pnl += pnl;
        sum_pnl += pnl;
        sum_pnl2 += pnl * pnl;

        if (pnl > 0.0) {
            sum_wins += pnl;
            num_wins++;
        } else {
            sum_losses += pnl;
        }

        cumulative += pnl;
        if (cumulative > peak) peak = cumulative;
        {
            /* Relative drawdown: (peak_equity - current_equity) / peak_equity.
             * Equity = 1.0 + cumulative (initial capital = 1.0).
             * Using (1.0 + peak) as denominator avoids division-by-zero when peak ≈ 0. */
            double peak_equity = 1.0 + peak;
            double dd = (peak - cumulative) / peak_equity;
            if (dd > max_dd) max_dd = dd;
        }
    }

    /* Profit Factor */
    if (fabs(sum_losses) < 1e-15)
        tm.profit_factor = (sum_wins > 0.0) ? 1e6 : 0.0;
    else
        tm.profit_factor = sum_wins / fabs(sum_losses);

    /* Sharpe (annualized) */
    mean_pnl = sum_pnl / (double)num_trades;
    {
        double var_pnl = sum_pnl2 / (double)num_trades - mean_pnl * mean_pnl;
        std_pnl = (var_pnl > 0.0) ? sqrt(var_pnl) : 0.0;
    }
    if (std_pnl > 1e-15)
        tm.sharpe_annual = mean_pnl / std_pnl * sqrt(annualization_factor);
    else
        tm.sharpe_annual = 0.0;

    /* Other metrics */
    tm.max_drawdown_pct = max_dd;  /* absolute fraction, e.g. 0.15 = 15% of initial capital */
    tm.win_rate = (double)num_wins / (double)num_trades;
    tm.avg_pnl_per_trade = tm.total_pnl / (double)num_trades;
    tm.net_pnl_pct = tm.total_pnl;  /* raw cumulative P&L for the window (not annualized) */

    /* Invariant: if total P&L is negative, MaxDD must be at least |total_pnl|.
     * If dd < |final_pnl| - eps, the dd computation is wrong. */
    if (tm.total_pnl < 0.0 && max_dd < fabs(tm.total_pnl) - 1e-9) {
        fprintf(stderr, "  WARNING: MaxDD %.6f < |total_pnl| %.6f — dd computation error\n",
                max_dd, fabs(tm.total_pnl));
    }

    return tm;
}

DiagnosticMetrics metrics_diagnostic(const double *predictions,
                                      const double *targets, int n) {
    DiagnosticMetrics dm;
    int i, correct = 0;
    double brier_sum = 0.0, logloss_sum = 0.0;

    memset(&dm, 0, sizeof(DiagnosticMetrics));
    dm.num_predictions = n;

    if (n <= 0) return dm;

    for (i = 0; i < n; i++) {
        double p = predictions[i];
        double t = targets[i];
        int pred_up = (p > 0.5) ? 1 : 0;
        int act_up = (t > 0.5) ? 1 : 0;
        if (pred_up == act_up) correct++;

        brier_sum += (p - t) * (p - t);
        logloss_sum += utils_binary_cross_entropy(p, t);
    }

    dm.accuracy = (double)correct / (double)n;
    dm.brier_score = brier_sum / (double)n;
    dm.log_loss = logloss_sum / (double)n;

    return dm;
}

void metrics_print_window(int window_num, const char *period_str,
                           const TradingMetrics *tm, const DiagnosticMetrics *dm) {
    printf("| %2d | %s | PF=%6.2f | Sharpe=%6.2f | MaxDD=%5.1f%% | Trades=%4d | WR=%4.1f%% | P&L=%7.2f%% | Acc=%5.1f%% | Brier=%.4f |\n",
           window_num, period_str,
           tm->profit_factor, tm->sharpe_annual, tm->max_drawdown_pct * 100.0,
           tm->num_trades, tm->win_rate * 100.0, tm->net_pnl_pct * 100.0,
           dm ? dm->accuracy * 100.0 : 0.0, dm ? dm->brier_score : 0.0);
}

void metrics_print_aggregate(const TradingMetrics *windows, int num_windows) {
    double sum_pf = 0.0, sum_sharpe = 0.0, sum_dd = 0.0;
    double sum_pnl = 0.0, sum_wr = 0.0;
    int total_trades = 0, i;

    for (i = 0; i < num_windows; i++) {
        sum_pf += windows[i].profit_factor;
        sum_sharpe += windows[i].sharpe_annual;
        sum_dd += windows[i].max_drawdown_pct;
        sum_pnl += windows[i].net_pnl_pct;
        sum_wr += windows[i].win_rate;
        total_trades += windows[i].num_trades;
    }

    printf("\n=== AGGREGATE (mean across %d windows) ===\n", num_windows);
    printf("  PF=%.2f  Sharpe=%.2f  MaxDD=%.1f%%  Trades=%d  WR=%.1f%%  P&L=%.2f%%\n",
           sum_pf / num_windows, sum_sharpe / num_windows,
           sum_dd / num_windows * 100.0, total_trades,
           sum_wr / num_windows * 100.0, sum_pnl / num_windows * 100.0);
}

void metrics_print_worst(const TradingMetrics *windows, int num_windows) {
    int worst_idx = 0, i;
    double worst_pf = windows[0].profit_factor;

    for (i = 1; i < num_windows; i++) {
        if (windows[i].profit_factor < worst_pf) {
            worst_pf = windows[i].profit_factor;
            worst_idx = i;
        }
    }

    printf("\n=== WORST WINDOW (window %d) ===\n", worst_idx + 1);
    printf("  PF=%.2f  Sharpe=%.2f  MaxDD=%.1f%%  Trades=%d  WR=%.1f%%  P&L=%.2f%%\n",
           windows[worst_idx].profit_factor, windows[worst_idx].sharpe_annual,
           windows[worst_idx].max_drawdown_pct * 100.0, windows[worst_idx].num_trades,
           windows[worst_idx].win_rate * 100.0, windows[worst_idx].net_pnl_pct * 100.0);
}

int metrics_check_criteria(const TradingMetrics *tm) {
    return (tm->profit_factor >= 1.2 && tm->sharpe_annual >= 0.5 && tm->max_drawdown_pct <= 0.25);
}

void metrics_print_verdict(const TradingMetrics *tm) {
    printf("\n=== VERDICT ===\n");
    printf("  Profit Factor >= 1.2:   %.2f %s\n", tm->profit_factor,
           tm->profit_factor >= 1.2 ? "PASS" : "FAIL");
    printf("  Sharpe >= 0.5:          %.2f %s\n", tm->sharpe_annual,
           tm->sharpe_annual >= 0.5 ? "PASS" : "FAIL");
    printf("  Max Drawdown <= 25%%:   %.1f%% %s\n", tm->max_drawdown_pct * 100.0,
           tm->max_drawdown_pct <= 0.25 ? "PASS" : "FAIL");
    printf("  Overall: %s\n", metrics_check_criteria(tm) ? "PASS" : "FAIL");
}
