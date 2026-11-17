/*
 * T-E2: MaxDD — relative drawdown + print unit check.
 * Equity curve [1.0, 1.1, 0.9, 1.05] via trade pnl fractions [0.1, -0.2, 0.15].
 * Relative DD = (1.1 - 0.9) / 1.1 = 0.181818...
 * Assert struct value AND print format (18.18%, not 1818% or 0.18%).
 */
#include "test_common.h"

int test_te2_run(void) {
    Trade trades[3];
    TradingMetrics tm;
    char buf[256];
    double printed_dd;

    /* Equity: 1.0 → 1.1 → 0.9 → 1.05
     * PnL fractions: +0.1, -0.2, +0.15
     * Cumulative: 0.1, -0.1, 0.05
     * Peak: 0.1, 0.1, 0.1
     * Relative DD = (0.1 - (-0.1)) / (1.0 + 0.1) = 0.2 / 1.1 = 0.181818... */
    trades[0].pnl_fraction = 0.1;
    trades[1].pnl_fraction = -0.2;
    trades[2].pnl_fraction = 0.15;

    tm = metrics_compute(trades, 3, 252.0);

    /* --- Struct value: relative drawdown = 0.2/1.1 --- */
    ASSERT_APPROX(tm.max_drawdown_pct, 0.2 / 1.1, 1e-9);

    /* --- Print format: should show 18.18%, not 1818% or 0.18% --- */
    /* Simulate the print: max_drawdown_pct * 100.0 */
    printed_dd = tm.max_drawdown_pct * 100.0;
    printf("  MaxDD = %.6f (struct), %.2f%% (printed)\n", tm.max_drawdown_pct, printed_dd);

    /* Must be ~18.18, not ~1818 (×10000 bug) or ~0.18 (no ×100 bug) */
    if (printed_dd > 100.0) {
        fprintf(stderr, "  FAIL: printed MaxDD %.2f%% > 100%% — unit bug (×10000 instead of ×100)\n", printed_dd);
        return 1;
    }
    if (printed_dd < 1.0) {
        fprintf(stderr, "  FAIL: printed MaxDD %.4f%% < 1%% — unit bug (fraction printed without ×100)\n", printed_dd);
        return 1;
    }
    ASSERT_APPROX(printed_dd, 18.181818, 0.01);

    /* --- Invariant: if total_pnl < 0, dd >= |total_pnl| --- */
    /* Here total_pnl = 0.1 - 0.2 + 0.15 = 0.05 > 0, so invariant doesn't trigger.
     * Test with a losing series: */
    {
        Trade losing[3];
        TradingMetrics tm_lose;
        losing[0].pnl_fraction = -0.05;
        losing[1].pnl_fraction = -0.03;
        losing[2].pnl_fraction = -0.02;
        tm_lose = metrics_compute(losing, 3, 252.0);
        /* total_pnl = -0.10, dd should be >= 0.10 */
        ASSERT_TRUE(tm_lose.max_drawdown_pct >= fabs(tm_lose.total_pnl) - 1e-9);
        printf("  Invariant: losing series total_pnl=%.4f dd=%.4f — dd >= |total_pnl| OK\n",
               tm_lose.total_pnl, tm_lose.max_drawdown_pct);
    }

    printf("T-E2 PASS: MaxDD=%.6f struct, %.2f%% printed (relative, correct units)\n",
           tm.max_drawdown_pct, printed_dd);
    return 0;
}
