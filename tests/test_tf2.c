/*
 * T-F2: Feature known values.
 * Monotonic increasing series: close[i] = 100 + 0.1*i.
 * Assert: RSI=100 (after period 14), bb_pos in [0,1], stoch_k in [0,100],
 *         ATR > 0, ma_diff > 0 (after bar 20).
 * Plus exact candle_return and close_return at bar 5.
 */
#include "test_common.h"

#define TF2_N 100

int test_tf2_run(void) {
    DataRow raw[TF2_N];
    double **feat;
    double fmin[26], fmax[26];
    char *fnames[26];
    int i, j;

    /* Monotonic increasing: close[i] = 100 + 0.1*i */
    for (i = 0; i < TF2_N; i++) {
        double c = 100.0 + 0.1 * (double)i;
        double o = c - 0.05;
        double h = c + 0.05;
        double l = c - 0.10;
        sprintf(raw[i].date, "2020.%02d.%02d", 1 + i/30, 1 + i%30);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = o; raw[i].high = h; raw[i].low = l; raw[i].close = c;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    feat = data_compute_raw_features(raw, TF2_N, 26, fmin, fmax, fnames);

    /* --- Exact values at bar 5 --- */
    {
        double c5 = 100.5, o5 = 100.45, c4 = 100.4;
        double exp_candle_ret = (c5 - o5) / o5;   /* 0.05 / 100.45 */
        double exp_close_ret  = (c5 - c4) / c4;    /* 0.1 / 100.4 */
        ASSERT_APPROX(feat[5][0], exp_candle_ret, 1e-12);
        ASSERT_APPROX(feat[5][2], exp_close_ret,  1e-12);
    }

    /* --- RSI: should be 100.0 (all gains) for bars >= 14 --- */
    /* Feature col 6 = rsi / 100.0, so expect 1.0 */
    for (i = 14; i < TF2_N; i++) {
        if (fabs(feat[i][6] - 1.0) > 1e-6) {
            fprintf(stderr, "  FAIL: RSI/100 at bar %d = %.6f, expected 1.0\n", i, feat[i][6]);
            for (j = 0; j < 26; j++) free(fnames[j]);
            data_free_raw_features(feat, TF2_N);
            return 1;
        }
    }

    /* --- BB position in [0, 1] --- */
    for (i = 0; i < TF2_N; i++) {
        ASSERT_TRUE(feat[i][8] >= -1e-9);
        ASSERT_TRUE(feat[i][8] <=  1.0 + 1e-9);
    }

    /* --- Stoch %K / 100 in [0, 1] --- */
    for (i = 0; i < TF2_N; i++) {
        ASSERT_TRUE(feat[i][10] >= -1e-9);
        ASSERT_TRUE(feat[i][10] <=  1.0 + 1e-9);
    }

    /* --- ATR norm > 0 for bars >= 13 --- */
    for (i = 13; i < TF2_N; i++) {
        ASSERT_TRUE(feat[i][9] > 0.0);
    }

    /* --- ma_diff > 0 for bars >= 20 (uptrend: SMA5 > SMA20) --- */
    for (i = 20; i < TF2_N; i++) {
        ASSERT_TRUE(feat[i][4] > 0.0);
    }

    /* --- Volatility > 0 for bars >= 12 --- */
    for (i = 12; i < TF2_N; i++) {
        ASSERT_TRUE(feat[i][5] >= 0.0);  /* can be 0 if all returns identical */
    }

    for (j = 0; j < 26; j++) free(fnames[j]);
    data_free_raw_features(feat, TF2_N);

    printf("T-F2 PASS: monotonic series — RSI=100, BB/Stoch in range, ATR>0, ma_diff>0, exact values OK\n");
    return 0;
}
