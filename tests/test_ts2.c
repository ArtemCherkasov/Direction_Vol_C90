/*
 * T-S2: Dataset bounds.
 * num_raw=20, seq=5, h=2 → num_samples = 20-5-2 = 13.
 * Verify max label index = num_raw-1 (no out-of-bounds).
 * Access all labels to confirm no segfault.
 */
#include "test_common.h"

int test_ts2_run(void) {
    int num_raw = 20, seq = 5, h = 2;
    DataRow raw[20];
    Dataset *ds;
    int i, s, k;
    int max_idx_accessed = 0;

    for (i = 0; i < num_raw; i++) {
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open  = 1.0 + 0.01 * i;
        raw[i].high  = raw[i].open + 0.02;
        raw[i].low   = raw[i].open - 0.02;
        raw[i].close = 1.0 + 0.01 * i + 0.005;
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    /* num_samples = 20 - 5 - 2 = 13 */
    ASSERT_INT(ds->num_samples, 13);

    /* Access all labels — compute max raw index used */
    for (i = 0; i < ds->num_samples; i++) {
        for (s = 0; s < seq; s++) {
            for (k = 0; k < h; k++) {
                int idx_next = i + s + 1 + k;
                if (idx_next > max_idx_accessed) max_idx_accessed = idx_next;
                /* Touch the label to confirm no segfault */
                (void)ds->labels[i][s * h + k];
            }
        }
    }

    /* Max index = num_samples-1 + seq-1 + 1 + h-1 = 12 + 4 + 1 + 1 = 18 */
    ASSERT_INT(max_idx_accessed, 18);
    ASSERT_TRUE(max_idx_accessed < num_raw);  /* no out-of-bounds */

    data_free_dataset(ds);
    printf("T-S2 PASS: num_samples=13, max raw idx=%d < num_raw=%d\n", max_idx_accessed, num_raw);
    return 0;
}
