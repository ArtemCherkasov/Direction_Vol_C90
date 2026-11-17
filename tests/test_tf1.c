/*
 * T-F1: Feature causality — THE critical test.
 * Build 200 bars with seeded-noise closes, compute 26 features.
 * Then modify raw[150].close by +1% and recompute.
 * Assert: features[i][j] unchanged for ALL i < 150, ALL j in [0,25].
 * One test catches any forward leakage in indicators or MTF.
 */
#include "test_common.h"

#define TF1_N 200
#define TF1_MOD_BAR 150

static void build_synthetic(DataRow *raw, int n) {
    int i;
    for (i = 0; i < n; i++) {
        double close = 1.0 + 0.001 * (double)i + 0.01 * lcg_noise(i);
        double open  = (i > 0) ? raw[i-1].close : close - 0.001;
        double hi    = (open > close ? open : close) + 0.002;
        double lo    = (open < close ? open : close) - 0.002;
        int yr = 2020 + (i / 365);
        int doy = i % 365;
        int mo = 1 + doy / 30;
        int dy = 1 + doy % 30;
        if (mo > 12) { mo = 12; dy = 31; }
        sprintf(raw[i].date, "%04d.%02d.%02d", yr, mo, dy);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = open;
        raw[i].high = hi;
        raw[i].low  = lo;
        raw[i].close = close;
        raw[i].tickvol = 1000 + i;
        raw[i].vol = 500;
        raw[i].spread = 10;
    }
}

int test_tf1_run(void) {
    DataRow raw1[TF1_N], raw2[TF1_N];
    double **feat1, **feat2;
    double min1[26], max1[26], min2[26], max2[26];
    char *names1[26], *names2[26];
    int i, j;
    int violations = 0;

    /* Build identical synthetic data */
    build_synthetic(raw1, TF1_N);
    memcpy(raw2, raw1, sizeof(raw1));

    /* Compute features on original */
    feat1 = data_compute_raw_features(raw1, TF1_N, 26, min1, max1, names1);

    /* Modify raw[150].close by +1% */
    raw2[TF1_MOD_BAR].close *= 1.01;

    /* Recompute features */
    feat2 = data_compute_raw_features(raw2, TF1_N, 26, min2, max2, names2);

    /* Assert: all features at bars < 150 are unchanged */
    for (i = 0; i < TF1_MOD_BAR; i++) {
        for (j = 0; j < 26; j++) {
            double diff = fabs(feat1[i][j] - feat2[i][j]);
            if (diff > 1e-15) {
                if (violations < 5) {
                    fprintf(stderr, "  LEAK at bar %d col %d: orig=%.15g new=%.15g diff=%.3e\n",
                            i, j, feat1[i][j], feat2[i][j], diff);
                }
                violations++;
            }
        }
    }

    /* Also verify bar 150 IS affected (at least base feature 0: candle_return) */
    {
        double d150 = fabs(feat1[TF1_MOD_BAR][0] - feat2[TF1_MOD_BAR][0]);
        if (d150 < 1e-15) {
            fprintf(stderr, "  WARNING: bar %d col 0 unexpectedly unchanged (diff=%.3e)\n",
                    TF1_MOD_BAR, d150);
        }
    }

    /* Cleanup feature names */
    for (j = 0; j < 26; j++) { free(names1[j]); free(names2[j]); }
    data_free_raw_features(feat1, TF1_N);
    data_free_raw_features(feat2, TF1_N);

    if (violations > 0) {
        fprintf(stderr, "  FAIL: %d feature leaks detected (bars < %d changed by modification at bar %d)\n",
                violations, TF1_MOD_BAR, TF1_MOD_BAR);
        return 1;
    }

    printf("T-F1 PASS: 0 leaks — modifying bar %d does not affect any feature at bars < %d (26 cols)\n",
           TF1_MOD_BAR, TF1_MOD_BAR);
    return 0;
}
