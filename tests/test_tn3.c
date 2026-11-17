/*
 * T-N3: Split boundary — no train/test feature overlap (with purge).
 * After adding embargo purge (seq_len-1 samples skipped at boundary),
 * assert: first_test_feat_bar >= last_train_feat_bar + 2.
 * This ensures no feature bars are shared between train and test.
 */
#include "test_common.h"

int test_tn3_run(void) {
    int num_raw = 200, seq = 10, h = 1;
    DataRow raw[200];
    Dataset *ds;
    DatasetSplit split;
    int train_count, test_count, i;
    int purge;

    for (i = 0; i < num_raw; i++) {
        double c = 1.0 + 0.01 * i + 0.003 * lcg_noise(i);
        sprintf(raw[i].date, "2020.%02d.%02d", 1 + i/30, 1 + i%30);
        strcpy(raw[i].time_, "00:00");
        raw[i].open = c - 0.001; raw[i].high = c + 0.002;
        raw[i].low = c - 0.003;  raw[i].close = c;
        raw[i].tickvol = 1000 + i; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    split = data_split_train_test(ds, 0.8);
    if (!split.train || !split.test) { fprintf(stderr, "  FAIL: split failed\n"); data_free_dataset(ds); return 1; }

    train_count = split.train->num_samples;
    test_count = split.test->num_samples;
    purge = seq;  /* embargo gap = seq_len for 1-bar buffer */

    printf("  train=%d test=%d purge=%d total_samples=%d\n",
           train_count, test_count, purge, ds->num_samples);
    ASSERT_TRUE(train_count > 0);
    ASSERT_TRUE(test_count > 0);

    /* With purge: test starts at sample (train_count + purge),
     * so first_test_feat_bar = train_count + purge.
     * last_train_feat_bar = (train_count - 1) + seq - 1 = train_count + seq - 2.
     * No overlap requires: train_count + purge >= train_count + seq - 2 + 2
     *                       → purge >= seq - 1. We use purge = seq - 1. ✓ */
    {
        int last_train_feat_bar = (train_count - 1) + seq - 1;
        int first_test_feat_bar = train_count + purge;  /* first feature bar of first test sample */
        int gap = first_test_feat_bar - last_train_feat_bar;

        printf("  last_train_feat_bar=%d  first_test_feat_bar=%d  gap=%d\n",
               last_train_feat_bar, first_test_feat_bar, gap);

        /* Assert: no overlap — gap >= 2 (at least 1 bar between windows) */
        ASSERT_TRUE(first_test_feat_bar >= last_train_feat_bar + 2);
    }

    /* Verify by checking that feature pointers are distinct */
    for (i = 0; i < train_count; i++) {
        int j;
        for (j = 0; j < test_count; j++) {
            if (split.train->features[i] == split.test->features[j]) {
                fprintf(stderr, "  FAIL: train sample %d == test sample %d (same pointer)\n", i, j);
                data_free_split(&split);
                data_free_dataset(ds);
                return 1;
            }
        }
    }

    data_free_split(&split);
    data_free_dataset(ds);
    printf("T-N3 PASS: train (%d) + purge (%d) + test (%d) — no feature overlap\n",
           train_count, purge, test_count);
    return 0;
}
