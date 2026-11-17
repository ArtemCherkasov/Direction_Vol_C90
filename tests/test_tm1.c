/*
 * T-M1: MTF consecutive aggregation — merge.
 * 6 synthetic bars, h_period=2 → 3 aggregate bars.
 * Assert: o=first, h=max, l=min, c=last, tickvol=sum. Tolerance 1e-12.
 */
#include "test_common.h"

int test_tm1_run(void) {
    DataRow raw[6];
    DataRow *agg = NULL;
    int agg_count = 0;
    int i;

    /* 6 bars with known OHLC */
    make_row(&raw[0], "2020.01.01","00:00", 1.10, 1.12, 1.09, 1.11, 100, 50, 10);
    make_row(&raw[1], "2020.01.02","00:00", 1.11, 1.15, 1.08, 1.14, 200, 60, 10);
    make_row(&raw[2], "2020.01.03","00:00", 1.14, 1.16, 1.13, 1.15, 300, 70, 10);
    make_row(&raw[3], "2020.01.06","00:00", 1.15, 1.18, 1.10, 1.12, 400, 80, 10);
    make_row(&raw[4], "2020.01.07","00:00", 1.12, 1.14, 1.11, 1.13, 500, 90, 10);
    make_row(&raw[5], "2020.01.08","00:00", 1.13, 1.17, 1.10, 1.16, 600, 100, 10);

    data_aggregate_raw_consecutive(raw, 6, 2, &agg, &agg_count);

    ASSERT_INT(agg_count, 3);

    /* Agg[0] = bars 0,1 */
    ASSERT_APPROX(agg[0].open,  1.10, 1e-12);  /* first */
    ASSERT_APPROX(agg[0].high,  1.15, 1e-12);  /* max(1.12, 1.15) */
    ASSERT_APPROX(agg[0].low,   1.08, 1e-12);  /* min(1.09, 1.08) */
    ASSERT_APPROX(agg[0].close, 1.14, 1e-12);  /* last */
    ASSERT_INT(agg[0].tickvol, 300);            /* 100 + 200 */

    /* Agg[1] = bars 2,3 */
    ASSERT_APPROX(agg[1].open,  1.14, 1e-12);
    ASSERT_APPROX(agg[1].high,  1.18, 1e-12);
    ASSERT_APPROX(agg[1].low,   1.10, 1e-12);
    ASSERT_APPROX(agg[1].close, 1.12, 1e-12);
    ASSERT_INT(agg[1].tickvol, 700);            /* 300 + 400 */

    /* Agg[2] = bars 4,5 */
    ASSERT_APPROX(agg[2].open,  1.12, 1e-12);
    ASSERT_APPROX(agg[2].high,  1.17, 1e-12);
    ASSERT_APPROX(agg[2].low,   1.10, 1e-12);
    ASSERT_APPROX(agg[2].close, 1.16, 1e-12);
    ASSERT_INT(agg[2].tickvol, 1100);           /* 500 + 600 */

    free(agg);
    printf("T-M1 PASS: 6 bars → 3 agg, OHLC and tickvol correct to 1e-12\n");
    return 0;
}
