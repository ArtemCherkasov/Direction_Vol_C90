/*
 * T-S4: Baseline on real data (discriminator D1).
 * Load real EURUSD CSV, build dataset, compute close_return feature.
 * Accuracy of momentum (sign(X[i-1])) and mean-reversion (-sign(X[i-1])) vs labels.
 * Honest data expectation: 45-55% for both.
 * If ~70% → data itself is predictable. If both <=55% but model gives 70% → leak.
 */
#include "test_common.h"

int test_ts4_run(void) {
    const char *path = "../resources/EURUSD_Daily_200301010000_202611160000.csv";
    DataRow *raw = NULL;
    int num_raw = 0, rc;
    Dataset *ds;
    int seq = 10, h = 1;
    int nf = 26;
    int chunk = 1000;  /* use first 1000 bars */
    int i;
    int mom_correct = 0, rev_correct = 0, total = 0;
    double mom_acc, rev_acc;

    rc = data_load_raw(path, &raw, &num_raw);
    if (rc != 0) {
        fprintf(stderr, "  SKIP: cannot load %s\n", path);
        return 0;
    }

    if (num_raw > chunk) num_raw = chunk;

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) {
        fprintf(stderr, "  SKIP: cannot build dataset (not enough data)\n");
        free(raw);
        return 0;
    }

    /* For each sample, use close_return at last timestep as the signal.
     * Label at last timestep: (close[i+seq] > close[i+seq-1]) ? 1 : 0
     * Momentum: predict label = 1 if close_return > 0 else 0
     * Mean-reversion: predict label = 0 if close_return > 0 else 1
     */
    for (i = 1; i < ds->num_samples; i++) {
        double cret = ds->features[i][(seq - 1) * nf + 2];  /* close_return at last step */
        double label = ds->labels[i][(seq - 1) * h];          /* label at last step */
        int mom_pred = (cret > 0.0) ? 1 : 0;
        int rev_pred = (cret > 0.0) ? 0 : 1;
        int actual   = (label > 0.5) ? 1 : 0;

        if (mom_pred == actual) mom_correct++;
        if (rev_pred == actual) rev_correct++;
        total++;
    }

    mom_acc = (total > 0) ? (double)mom_correct / (double)total : 0.0;
    rev_acc = (total > 0) ? (double)rev_correct / (double)total : 0.0;

    printf("  Baseline on %d samples: momentum=%.1f%%  mean-reversion=%.1f%%\n",
           total, mom_acc * 100.0, rev_acc * 100.0);

    /* Print first few close values to diagnose column mapping */
    printf("  First 3 closes: %.5f %.5f %.5f\n", raw[0].close, raw[1].close, raw[2].close);

    data_free_dataset(ds);
    free(raw);

    /* Check: both should be in [0.40, 0.60] for honest data */
    if (mom_acc < 0.40 || mom_acc > 0.60 || rev_acc < 0.40 || rev_acc > 0.60) {
        fprintf(stderr, "  FAIL: baseline outside 40-60%%: mom=%.1f%% rev=%.1f%%\n",
                mom_acc * 100.0, rev_acc * 100.0);
        fprintf(stderr, "  If closes look like tickvols (601, 8880, ...) → column offset bug (see T-D4)\n");
        fprintf(stderr, "  If closes are real prices and one rule ~70%% → data is predictable\n");
        return 1;
    }

    printf("T-S4 PASS: both baselines in 40-60%% — data is not trivially predictable\n");
    return 0;
}
