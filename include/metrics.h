#ifndef METRICS_H
#define METRICS_H

/* ==================== Trade Record ==================== */
typedef struct {
    int bar_index;
    int entry_bar;
    int exit_bar;
    int direction;       /* +1 = long, -1 = short, 0 = no trade */
    double entry_price;
    double exit_price;
    double pnl_fraction;
    double position_size;
} Trade;

/* ==================== Trading Metrics (verdict) ==================== */
typedef struct {
    double net_pnl_pct;       /* net P&L as % annualized */
    double profit_factor;     /* sum(wins) / abs(sum(losses)) */
    double sharpe_annual;     /* Sharpe * sqrt(252) */
    double max_drawdown_pct;  /* peak-to-trough drawdown as % */
    int num_trades;
    double avg_pnl_per_trade;
    double win_rate;
    double total_pnl;
    int invariant_error;  /* 1 if MaxDD < |total_pnl| invariant violated (final_pnl < 0) */
} TradingMetrics;

TradingMetrics metrics_compute(const Trade *trades, int num_trades,
                                double annualization_factor);

/* ==================== Diagnostic Metrics ==================== */
typedef struct {
    double accuracy;
    double brier_score;
    double log_loss;
    int num_predictions;
} DiagnosticMetrics;

DiagnosticMetrics metrics_diagnostic(const double *predictions,
                                      const double *targets, int n);

/* ==================== Report ==================== */
void metrics_print_window(int window_num, const char *period_str,
                           const TradingMetrics *tm, const DiagnosticMetrics *dm);
void metrics_print_worst(const TradingMetrics *windows, int num_windows);
int metrics_check_criteria(const TradingMetrics *tm);
void metrics_print_verdict(const TradingMetrics *tm);

#endif /* METRICS_H */
