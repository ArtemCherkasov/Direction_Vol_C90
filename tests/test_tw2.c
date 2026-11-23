/*
 * T-W2: Walk-forward offset.
 * In walkfwd.c: raw_offset = train_count = split.train->num_samples (passed to backtest).
 * Build dataset, split, verify the offset equals train sample count.
 */
#include "test_common.h"

int test_tw2_run(void) {
    int num_raw = 200, seq = 10, h = 1;
    DataRow raw[200];
    Dataset *ds;
    DatasetSplit split;
    int i, raw_offset;

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

    /* Simulate walkfwd split: ratio = train_bars / total_bars */
    {
        int train_bars = 160;  /* example: 80% of 200 */
        double ratio = (double)train_bars / (double)num_raw;
        split = data_split_train_test(ds, ratio);
    }

    if (!split.train || !split.test) {
        fprintf(stderr, "  FAIL: split failed\n");
        data_free_dataset(ds); return 1;
    }

    /* In walkfwd.c: raw_offset = split.test_start_sample (train_count + purge) passed to backtest_run */
    raw_offset = split.test_start_sample;

    printf("  train_samples=%d  test_samples=%d  raw_offset=%d\n",
           split.train->num_samples, split.test->num_samples, raw_offset);

    /* Assert: raw_offset == train_count + purge (seq_len) — accounts for embargo gap */
    ASSERT_INT(raw_offset, split.train->num_samples + seq);
    ASSERT_INT(split.test_start_sample, split.train->num_samples + seq);

    /* Assert: test sample 0 maps to full_ds sample raw_offset */
    /* Full_ds sample raw_offset has first feature bar = raw_offset */
    /* Entry bar = raw_offset + 0 + seq_len = raw_offset + seq_len */
    {
        int entry_bar_0 = raw_offset + 0 + seq;
        ASSERT_TRUE(entry_bar_0 < num_raw);
        printf("  First test: entry_bar=%d (raw_offset + seq_len = %d + %d = %d)\n",
               entry_bar_0, raw_offset, seq, entry_bar_0);
    }

    data_free_split(&split);
    data_free_dataset(ds);
    printf("T-W2 PASS: raw_offset = test_start_sample = %d (train_count + purge), entry_bar = offset + seq_len\n", raw_offset);
    return 0;
}
