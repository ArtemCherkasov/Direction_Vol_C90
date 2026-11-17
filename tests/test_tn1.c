/*
 * T-N1: Train-only normalization stats.
 * 10 samples, 2 features, seq=1. Brute-force min/max over [0,6).
 * Compare with data_compute_train_stats. Then normalize test [6,10) — no NaN.
 */
#include "test_common.h"

int test_tn1_run(void) {
    Dataset ds;
    double features[10][2];
    double *feat_ptrs[10];
    double *label_ptrs[10];
    double labels[10];
    double train_min[2], train_max[2];
    int i, j;

    for (i = 0; i < 10; i++) {
        features[i][0] = (double)i;         /* 0..9  */
        features[i][1] = (double)(i * 2);   /* 0..18 */
        feat_ptrs[i] = features[i];
        labels[i] = (double)(i % 2);
        label_ptrs[i] = &labels[i];
    }

    ds.features = feat_ptrs;
    ds.labels = label_ptrs;
    ds.num_samples = 10;
    ds.num_features = 2;
    ds.sequence_length = 1;
    ds.num_outputs = 1;
    ds.feat_min = (double *)malloc(2 * sizeof(double));
    ds.feat_max = (double *)malloc(2 * sizeof(double));

    /* Brute-force min/max over [0,6) */
    for (j = 0; j < 2; j++) {
        train_min[j] = features[0][j];
        train_max[j] = features[0][j];
        for (i = 1; i < 6; i++) {
            if (features[i][j] < train_min[j]) train_min[j] = features[i][j];
            if (features[i][j] > train_max[j]) train_max[j] = features[i][j];
        }
    }

    /* data_compute_train_stats */
    data_compute_train_stats(&ds, 0, 6);

    /* Compare brute-force vs function */
    ASSERT_APPROX(ds.feat_min[0], train_min[0], 1e-12);
    ASSERT_APPROX(ds.feat_max[0], train_max[0], 1e-12);
    ASSERT_APPROX(ds.feat_min[1], train_min[1], 1e-12);
    ASSERT_APPROX(ds.feat_max[1], train_max[1], 1e-12);

    /* Expected: min=[0,0], max=[5,10] */
    ASSERT_APPROX(ds.feat_min[0], 0.0, 1e-12);
    ASSERT_APPROX(ds.feat_max[0], 5.0, 1e-12);
    ASSERT_APPROX(ds.feat_min[1], 0.0, 1e-12);
    ASSERT_APPROX(ds.feat_max[1], 10.0, 1e-12);

    /* Normalize full dataset with train stats */
    data_normalize_with_stats(&ds, ds.feat_min, ds.feat_max);

    /* Train samples [0,6): should be in [0, 1] */
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 2; j++) {
            double v = ds.features[i][j];
            ASSERT_TRUE(v >= -1e-9);
            ASSERT_TRUE(v <=  1.0 + 1e-9);
        }
    }

    /* Test samples [6,10): may exceed [0,1] (this is normal), but no NaN */
    for (i = 6; i < 10; i++) {
        for (j = 0; j < 2; j++) {
            double v = ds.features[i][j];
            ASSERT_TRUE(isfinite(v));
        }
    }

    /* Verify test sample 9 feature 0 = (9 - 0) / (5 - 0) = 1.8 (exceeds 1.0) */
    ASSERT_APPROX(features[9][0], 1.8, 1e-12);

    free(ds.feat_min);
    free(ds.feat_max);
    printf("T-N1 PASS: train stats match brute-force, test normalized no NaN, test exceeds [0,1]\n");
    return 0;
}
