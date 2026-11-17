/*
 * T-E3: P&L for window — no ×252 annualization.
 * 3 trades with known pnl fractions. net_pnl_pct = raw sum (not annualized).
 */
#include "test_common.h"

int test_te3_run(void) {
    Trade trades[3];
    TradingMetrics tm;

    trades[0].pnl_fraction = 0.05;
    trades[1].pnl_fraction = -0.02;
    trades[2].pnl_fraction = 0.03;

    /* With annualization_factor=252, but net_pnl_pct should NOT be annualized */
    tm = metrics_compute(trades, 3, 252.0);

    /* net_pnl_pct = 0.05 + (-0.02) + 0.03 = 0.06 (raw, no ×252) */
    ASSERT_APPROX(tm.net_pnl_pct, 0.06, 1e-12);
    ASSERT_APPROX(tm.total_pnl, 0.06, 1e-12);

    /* avg_pnl_per_trade = 0.06 / 3 = 0.02 */
    ASSERT_APPROX(tm.avg_pnl_per_trade, 0.02, 1e-12);

    /* If it were annualized: 0.06 * 252 = 15.12 — verify it's NOT this */
    ASSERT_TRUE(tm.net_pnl_pct != 0.06 * 252.0);

    printf("T-E3 PASS: net_pnl_pct=%.4f (raw sum, not ×252=%.2f)\n",
           tm.net_pnl_pct, 0.06 * 252.0);
    return 0;
}
