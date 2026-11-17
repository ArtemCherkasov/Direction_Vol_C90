/*
 * T-M2: MTF consecutive aggregation — tail behavior.
 * 7 bars, h_period=2 → num_agg = 7/2 = 3 (integer division drops last incomplete block).
 * Document and fix this behavior with the test.
 */
#include "test_common.h"

int test_tm2_run(void) {
    DataRow raw[7];
    DataRow *agg = NULL;
    int agg_count = 0;
    int i;

    for (i = 0; i < 7; i++) {
        char d[16];
        sprintf(d, "2020.01.%02d", i + 1);
        make_row(&raw[i], d, "00:00",
                 1.10 + 0.01 * i,
                 1.12 + 0.01 * i,
                 1.09 + 0.01 * i,
                 1.11 + 0.01 * i,
                 100 * (i + 1), 50, 10);
    }

    data_aggregate_raw_consecutive(raw, 7, 2, &agg, &agg_count);

    /* num_agg = 7 / 2 = 3 — last incomplete block (bar 6) is dropped */
    ASSERT_INT(agg_count, 3);

    /* Verify agg[2] uses bars 4,5 (not bar 6) */
    ASSERT_APPROX(agg[2].open,  raw[4].open,  1e-12);
    ASSERT_APPROX(agg[2].close, raw[5].close, 1e-12);
    ASSERT_INT(agg[2].tickvol, raw[4].tickvol + raw[5].tickvol);

    /* Bar 6 is NOT included in any aggregate */
    ASSERT_TRUE(agg[2].close != raw[6].close);

    free(agg);
    printf("T-M2 PASS: 7 bars / h=2 → 3 agg, last incomplete block dropped (documented)\n");
    return 0;
}
