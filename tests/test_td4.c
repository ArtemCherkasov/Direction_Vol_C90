/*
 * T-D4: Gap statistics on real data (D2 discriminator).
 * Load real EURUSD CSV, compute |open[t] - close[t-1]| in pips.
 * Criteria: mean <= 10 pips, fraction > 20 pips <= 25%.
 * If fails → open column is wrong series (column offset bug in parser).
 *
 * Also prints first 5 rows as loaded by the parser to diagnose column mapping.
 */
#include "test_common.h"

int test_td4_run(void) {
    const char *path = "../resources/EURUSD_Daily_200301010000_202611160000.csv";
    DataRow *rows = NULL;
    int count = 0, rc, i;
    double pip_value = 0.0001;
    double sum_gap = 0.0;
    int num_gaps = 0;
    int big_gaps = 0;
    double mean_gap, big_frac;

    rc = data_load_raw(path, &rows, &count);
    if (rc != 0) {
        fprintf(stderr, "  SKIP: cannot load %s (run from build dir)\n", path);
        return 0;  /* skip if file not found */
    }

    printf("  Loaded %d rows\n", count);

    /* Print first 5 rows as the parser sees them */
    printf("  First 5 rows (as parsed):\n");
    for (i = 0; i < 5 && i < count; i++) {
        printf("    [%d] date=%s time=%s o=%.5f h=%.5f l=%.5f c=%.5f tv=%d v=%d sp=%d\n",
               i, rows[i].date, rows[i].time_,
               rows[i].open, rows[i].high, rows[i].low, rows[i].close,
               rows[i].tickvol, rows[i].vol, rows[i].spread);
    }

    /* Compute gap statistics */
    for (i = 1; i < count; i++) {
        double gap = fabs(rows[i].open - rows[i-1].close);
        double gap_pips = gap / pip_value;
        sum_gap += gap_pips;
        num_gaps++;
        if (gap_pips > 20.0) big_gaps++;
    }

    mean_gap = sum_gap / (double)num_gaps;
    big_frac = (double)big_gaps / (double)num_gaps;

    printf("  Gap stats: mean=%.2f pips, big_frac=%.4f (>20pip=%d/%d)\n",
           mean_gap, big_frac, big_gaps, num_gaps);

    free(rows);

    /* Criteria: mean <= 10 pips, big_frac <= 0.25 */
    if (mean_gap > 10.0) {
        fprintf(stderr, "  FAIL: mean gap %.2f pips > 10 pips — open column may be wrong series!\n", mean_gap);
        fprintf(stderr, "  This indicates a column offset bug: parser expects 9 columns (date,time,open,high,low,close,...)\n");
        fprintf(stderr, "  but CSV may have 8 columns (no time). Check T-D1 format vs real CSV header.\n");
        return 1;
    }

    if (big_frac > 0.25) {
        fprintf(stderr, "  FAIL: big gap fraction %.4f > 0.25 — too many large gaps\n", big_frac);
        return 1;
    }

    printf("T-D4 PASS: mean gap %.2f pips, big_frac %.4f — data looks consistent\n", mean_gap, big_frac);
    return 0;
}
