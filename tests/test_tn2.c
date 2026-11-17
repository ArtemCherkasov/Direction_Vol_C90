/*
 * T-N2: Normalization order — stats must be computed before normalize.
 * Part 1: Normalize with uninitialized stats (min=1e30, max=-1e30) → garbage/NaN.
 * Part 2: Compute train stats first, then normalize → correct [0,1] for train.
 */
#include "test_common.h"

int test_tn2_run(void) {
    Dataset ds;
    double features[10][2];
    double *feat_ptrs[10];
    double *label_ptrs[10];
    double labels[10];
    double bad_min[2], bad_max[2];
    int i, j;
    int has_nan_bad, has_nan_good;

    for (i = 0; i < 10; i++) {
        features[i][0] = (double)i;
        features[i][1] = (double)(i * 3);
        feat_ptrs[i] = features[i];
        labels[i] = 0.0;
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

    /* --- Part 1: uninitialized stats (the wrong order) --- */
    bad_min[0] = 1e30; bad_min[1] = 1e30;
    bad_max[0] = -1e30; bad_max[1] = -1e30;
    /* range = max - min = -1e30 - 1e30 = -2e30 < epsilon → range set to 1.0 */
    /* normalized = (val - 1e30) / 1.0 = huge negative → not NaN but garbage */
    data_normalize_with_stats(&ds, bad_min, bad_max);
    has_nan_bad = 0;
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 2; j++) {
            if (!isfinite(features[i][j])) has_nan_bad = 1;
        }
    }
    /* With range < epsilon, range is set to 1.0, so values = (val - 1e30) */
    /* These are huge negative numbers — garbage, demonstrating wrong order */
    ASSERT_TRUE(features[0][0] < -1e29);  /* (0 - 1e30) / 1.0 ≈ -1e30 */

    /* Reset features */
    for (i = 0; i < 10; i++) {
        features[i][0] = (double)i;
        features[i][1] = (double)(i * 3);
    }

    /* --- Part 2: correct order — compute stats first --- */
    data_compute_train_stats(&ds, 0, 10);
    data_normalize_with_stats(&ds, ds.feat_min, ds.feat_max);

    /* All values should be in [0, 1] and finite */
    has_nan_good = 0;
    for (i = 0; i < 10; i++) {
        for (j = 0; j < 2; j++) {
            double v = features[i][j];
            if (!isfinite(v)) has_nan_good = 1;
            ASSERT_TRUE(v >= -1e-9);
            ASSERT_TRUE(v <=  1.0 + 1e-9);
        }
    }
    ASSERT_INT(has_nan_good, 0);

    /* First sample should be 0.0 (min), last should be 1.0 (max) */
    ASSERT_APPROX(features[0][0], 0.0, 1e-12);
    ASSERT_APPROX(features[9][0], 1.0, 1e-12);

    free(ds.feat_min);
    free(ds.feat_max);
    printf("T-N2 PASS: wrong order → garbage; correct order (stats then normalize) → [0,1] no NaN\n");
    return 0;
}
