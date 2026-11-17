/*
 * T-F3: MTF feature mapping.
 * 10 bars, compute 26 features. Separately aggregate by 2 and compute 13 base features.
 * Assert: raw_features[i][13+j] == mtf_features[mi][j] where mi = max(0, (i-1)/2).
 * Check clamp at i=0 (mi=0) and tail.
 */
#include "test_common.h"

#define TF3_N 10

int test_tf3_run(void) {
    DataRow raw[TF3_N];
    double **feat;
    double fmin[26], fmax[26];
    char *fnames[26];
    double **mtf_feat;
    DataRow *mtf_raw;
    int mtf_count;
    double mtf_min[13], mtf_max[13];
    char *mtf_names[13];
    int i, j;

    /* Build 10 bars with varied data */
    for (i = 0; i < TF3_N; i++) {
        double c = 1.0 + 0.01 * lcg_noise(i) + 0.001 * i;
        double o = c - 0.002 * lcg_noise(i + 100);
        double h = (o > c ? o : c) + 0.003;
        double l = (o < c ? o : c) - 0.003;
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = o; raw[i].high = h; raw[i].low = l; raw[i].close = c;
        raw[i].tickvol = 1000 + i * 10; raw[i].vol = 500; raw[i].spread = 10;
    }

    /* Compute 26 features */
    feat = data_compute_raw_features(raw, TF3_N, 26, fmin, fmax, fnames);

    /* Separately aggregate by 2 and compute 13 base features */
    data_aggregate_raw_consecutive(raw, TF3_N, 2, &mtf_raw, &mtf_count);
    mtf_feat = data_compute_raw_features(mtf_raw, mtf_count, 13, mtf_min, mtf_max, mtf_names);

    /* mtf_count = 10/2 = 5 */
    ASSERT_INT(mtf_count, 5);

    /* Assert mapping: raw_features[i][13+j] == mtf_features[mi][j] */
    for (i = 0; i < TF3_N; i++) {
        int mi = (i - 1) / 2;
        if (mi < 0) mi = 0;
        if (mi >= mtf_count) mi = mtf_count - 1;

        for (j = 0; j < 13; j++) {
            double diff = fabs(feat[i][13 + j] - mtf_feat[mi][j]);
            if (diff > 1e-12) {
                fprintf(stderr, "  FAIL: bar %d col %d (mi=%d): raw=%.12g mtf=%.12g diff=%.3e\n",
                        i, 13 + j, mi, feat[i][13 + j], mtf_feat[mi][j], diff);
                for (j = 0; j < 26; j++) free(fnames[j]);
                for (j = 0; j < 13; j++) free(mtf_names[j]);
                data_free_raw_features(feat, TF3_N);
                data_free_raw_features(mtf_feat, mtf_count);
                free(mtf_raw);
                return 1;
            }
        }
    }

    /* Explicitly check clamp at i=0: mi = (0-1)/2 = -1, clamped to 0 */
    {
        int mi0 = (0 - 1) / 2;
        if (mi0 < 0) mi0 = 0;
        ASSERT_INT(mi0, 0);
        for (j = 0; j < 13; j++) {
            ASSERT_APPROX(feat[0][13 + j], mtf_feat[0][j], 1e-12);
        }
    }

    for (j = 0; j < 26; j++) free(fnames[j]);
    for (j = 0; j < 13; j++) free(mtf_names[j]);
    data_free_raw_features(feat, TF3_N);
    data_free_raw_features(mtf_feat, mtf_count);
    free(mtf_raw);

    printf("T-F3 PASS: MTF mapping raw[i][13+j] == mtf[(i-1)/2][j] for all 10 bars, 13 cols\n");
    return 0;
}
