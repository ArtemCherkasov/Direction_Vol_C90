/*
 * T-D1: Data loading roundtrip.
 * Write 5 known MT5-format rows (9-column: date,time,open,high,low,close,tickvol,vol,spread),
 * load via data_load_raw, assert OHLC match to 1e-9, date/time byte-exact.
 */
#include "test_common.h"

int test_td1_run(void) {
    const char *path = "test_td1_tmp.csv";
    FILE *f;
    DataRow *rows = NULL;
    int count = 0, i;
    int rc;

    /* Known data — values chosen to be exactly representable */
    double exp_o[5] = {1.10000, 1.10500, 1.09800, 1.11200, 1.12000};
    double exp_h[5] = {1.10500, 1.10800, 1.10200, 1.11500, 1.12500};
    double exp_l[5] = {1.09500, 1.09900, 1.09500, 1.10700, 1.11800};
    double exp_c[5] = {1.10200, 1.10000, 1.10100, 1.11400, 1.12300};
    int    exp_tv[5] = {1000, 1500, 800, 2000, 1200};
    int    exp_v[5]  = {500, 600, 400, 700, 550};
    int    exp_sp[5] = {10, 10, 12, 8, 10};
    const char *exp_date[5] = {"2020.01.01","2020.01.02","2020.01.03","2020.01.06","2020.01.07"};
    const char *exp_time[5] = {"00:00","00:00","00:00","00:00","00:00"};

    /* Write CSV with full precision */
    f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return 1; }
    fprintf(f, "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n");
    for (i = 0; i < 5; i++) {
        fprintf(f, "%s\t%s\t%.10f\t%.10f\t%.10f\t%.10f\t%d\t%d\t%d\n",
                exp_date[i], exp_time[i],
                exp_o[i], exp_h[i], exp_l[i], exp_c[i],
                exp_tv[i], exp_v[i], exp_sp[i]);
    }
    fclose(f);

    /* Load */
    rc = data_load_raw(path, &rows, &count);
    if (rc != 0) { fprintf(stderr, "  FAIL: data_load_raw returned %d\n", rc); remove(path); return 1; }

    /* Assert count */
    ASSERT_INT(count, 5);

    /* Assert each row */
    for (i = 0; i < 5; i++) {
        ASSERT_STR(rows[i].date, exp_date[i]);
        ASSERT_STR(rows[i].time_, exp_time[i]);
        ASSERT_APPROX(rows[i].open,  exp_o[i], 1e-9);
        ASSERT_APPROX(rows[i].high,  exp_h[i], 1e-9);
        ASSERT_APPROX(rows[i].low,   exp_l[i], 1e-9);
        ASSERT_APPROX(rows[i].close, exp_c[i], 1e-9);
        ASSERT_INT(rows[i].tickvol, exp_tv[i]);
        ASSERT_INT(rows[i].vol,     exp_v[i]);
        ASSERT_INT(rows[i].spread,  exp_sp[i]);
    }

    free(rows);
    remove(path);
    printf("T-D1 PASS: 5 rows roundtrip OK (OHLC to 1e-9, date/time exact)\n");
    return 0;
}
