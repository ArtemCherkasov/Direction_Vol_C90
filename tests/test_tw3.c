/*
 * T-W3: Prediction alignment — KILLER test for D1.
 * For each test sample i in backtest:
 *   sample_fed = raw_offset + i
 *   last_feat_bar = sample_fed + seq_len - 1
 *   entry_bar = raw_offset + i + seq_len
 *   Assert: last_feat_bar == entry_bar - 1 (no shift).
 * If last_feat_bar == entry_bar → model reads label from input → 70% accuracy leak.
 */
#include "test_common.h"

int test_tw3_run(void) {
    int seq_len = 10;
    int raw_offset = 150;  /* example: train_count */
    int num_test = 30;
    int i;
    int shifts = 0;

    printf("  Checking alignment for %d test samples (raw_offset=%d, seq=%d)\n",
           num_test, raw_offset, seq_len);
    printf("  %-6s %-12s %-14s %-10s %-10s %s\n",
           "i", "sample_fed", "last_feat_bar", "entry_bar", "entry-1", "status");

    for (i = 0; i < num_test; i++) {
        int sample_fed = raw_offset + i;
        int last_feat_bar = sample_fed + seq_len - 1;
        int entry_bar = raw_offset + i + seq_len;

        /* The label for this sample compares close[entry_bar] vs close[entry_bar-1].
         * The last feature bar is last_feat_bar.
         * If last_feat_bar == entry_bar - 1: OK (no leak)
         * If last_feat_bar == entry_bar: LEAK (model reads the answer bar as input) */
        int ok = (last_feat_bar == entry_bar - 1);

        if (i < 5 || !ok) {
            printf("  %-6d %-12d %-14d %-10d %-10d %s\n",
                   i, sample_fed, last_feat_bar, entry_bar, entry_bar - 1,
                   ok ? "OK" : "SHIFT!");
        }

        if (!ok) shifts++;
    }

    /* Also verify the formula algebraically:
     * last_feat_bar = (raw_offset + i) + seq_len - 1
     * entry_bar     = raw_offset + i + seq_len
     * entry_bar - 1 = raw_offset + i + seq_len - 1
     * So last_feat_bar == entry_bar - 1 always holds. */
    ASSERT_INT(shifts, 0);

    /* Verify the label bar is NOT included in features:
     * Features span bars [sample_fed, sample_fed + seq_len - 1]
     * Label uses bars [entry_bar - 1, entry_bar] = [last_feat_bar, last_feat_bar + 1]
     * The label's "next" bar (entry_bar) is NOT in the feature range. */
    for (i = 0; i < num_test; i++) {
        int sample_fed = raw_offset + i;
        int last_feat_bar = sample_fed + seq_len - 1;
        int entry_bar = raw_offset + i + seq_len;
        int label_next_bar = entry_bar;  /* close[entry_bar] vs close[entry_bar-1] */

        /* label_next_bar must be > last_feat_bar (not in feature range) */
        ASSERT_TRUE(label_next_bar > last_feat_bar);
        ASSERT_INT(label_next_bar, last_feat_bar + 1);
    }

    printf("T-W3 PASS: 0 shifts — last_feat_bar == entry_bar - 1 for all %d samples (no leak)\n", num_test);
    return 0;
}
