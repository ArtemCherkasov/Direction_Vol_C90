/*
 * T-F4: Feature min/max sanity.
 * 50 bars with varied data, compute 26 features.
 * Assert: all 26 pairs feat_min[j] <= feat_max[j], both finite.
 * Build direction dataset: label_min=0, label_max=1.
 */
#include "test_common.h"

#define TF4_N 50

int test_tf4_run(void) {
    DataRow raw[TF4_N];
    double **feat;
    double fmin[26], fmax[26];
    char *fnames[26];
    int i, j;

    for (i = 0; i < TF4_N; i++) {
        double c = 1.0 + 0.01 * lcg_noise(i) + 0.002 * i;
        double o = c - 0.003 * lcg_noise(i + 50);
        double h = (o > c ? o : c) + 0.004;
        double l = (o < c ? o : c) - 0.004;
        sprintf(raw[i].date, "2020.%02d.%02d", 1 + i/30, 1 + i%30);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = o; raw[i].high = h; raw[i].low = l; raw[i].close = c;
        raw[i].tickvol = 1000 + i * 10; raw[i].vol = 500; raw[i].spread = 10;
    }

    feat = data_compute_raw_features(raw, TF4_N, 26, fmin, fmax, fnames);

    for (j = 0; j < 26; j++) {
        /* Both finite (not NaN, not Inf) */
        if (!isfinite(fmin[j]) || !isfinite(fmax[j])) {
            fprintf(stderr, "  FAIL: col %d min=%.6g max=%.6g not finite\n", j, fmin[j], fmax[j]);
            for (i = 0; i < 26; i++) free(fnames[i]);
            data_free_raw_features(feat, TF4_N);
            return 1;
        }
        /* min <= max */
        if (fmin[j] > fmax[j]) {
            fprintf(stderr, "  FAIL: col %d min=%.6g > max=%.6g\n", j, fmin[j], fmax[j]);
            for (i = 0; i < 26; i++) free(fnames[i]);
            data_free_raw_features(feat, TF4_N);
            return 1;
        }
    }

    for (i = 0; i < 26; i++) free(fnames[i]);
    data_free_raw_features(feat, TF4_N);

    /* Build direction dataset and check label bounds */
    {
        Dataset *ds = data_build_dataset_direction(raw, TF4_N, 5, 1);
        if (!ds) {
            fprintf(stderr, "  FAIL: cannot build dataset\n");
            return 1;
        }
        ASSERT_APPROX(ds->label_min[0], 0.0, 1e-15);
        ASSERT_APPROX(ds->label_max[0], 1.0, 1e-15);
        data_free_dataset(ds);
    }

    printf("T-F4 PASS: all 26 min<=max finite, label_min=0 label_max=1\n");
    return 0;
}
