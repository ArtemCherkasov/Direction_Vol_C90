/*
 * T-S3: Feature timing.
 * Build dataset, seq=3, h=1. Assert that the feature at the last timestep
 * belongs to bar i+seq-1, not i+seq.
 * Specifically: features[i][(seq-1)*nf + 2] == (c[i+seq-1] - c[i+seq-2]) / c[i+seq-2]
 * (close_return at last timestep = return of bar i+seq-1).
 */
#include "test_common.h"

int test_ts3_run(void) {
    int num_raw = 30, seq = 3, h = 1;
    int nf = 26;  /* NUM_RAW_FEATURES */
    DataRow raw[30];
    Dataset *ds;
    int i;

    for (i = 0; i < num_raw; i++) {
        /* Distinct closes so close_return is nonzero */
        double c = 1.0 + 0.01 * (double)i + 0.003 * lcg_noise(i);
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open  = c - 0.001;
        raw[i].high  = c + 0.002;
        raw[i].low   = c - 0.003;
        raw[i].close = c;
        raw[i].tickvol = 1000 + i; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    /* For each sample, check close_return at last timestep */
    for (i = 0; i < ds->num_samples; i++) {
        int ti = i + seq - 1;  /* last feature bar */
        double exp_cret = (raw[ti].close - raw[ti - 1].close) / raw[ti - 1].close;
        double got = ds->features[i][(seq - 1) * nf + 2];
        ASSERT_APPROX(got, exp_cret, 1e-12);
    }

    /* Also check first timestep: features[i][0*nf + 2] == close_return at bar i */
    for (i = 0; i < ds->num_samples; i++) {
        int ti = i;  /* first feature bar */
        if (ti == 0) continue;  /* close_return[0] = 0.0 by definition */
        double exp_cret = (raw[ti].close - raw[ti - 1].close) / raw[ti - 1].close;
        double got = ds->features[i][0 * nf + 2];
        ASSERT_APPROX(got, exp_cret, 1e-12);
    }

    data_free_dataset(ds);
    printf("T-S3 PASS: feature at last timestep belongs to bar i+seq-1 (not i+seq)\n");
    return 0;
}
