/*
 * T-D2: Data field validation.
 * Part 1: Valid file — assert high >= max(o,c), low <= min(o,c), o,c > 0 for all rows.
 * Part 2: Broken file (missing column → atof returns 0) — loader now validates OHLC
 *         and rejects the broken file (returns non-zero).
 */
#include "test_common.h"

int test_td2_run(void) {
    const char *good_path = "test_td2_good.csv";
    const char *bad_path  = "test_td2_bad.csv";
    FILE *f;
    DataRow *rows = NULL;
    int count = 0, i, rc;

    /* --- Part 1: valid file --- */
    f = fopen(good_path, "w");
    if (!f) { fprintf(stderr, "Cannot create %s\n", good_path); return 1; }
    fprintf(f, "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n");
    fprintf(f, "2020.01.01\t00:00\t1.10000\t1.10500\t1.09500\t1.10200\t1000\t500\t10\n");
    fprintf(f, "2020.01.02\t00:00\t1.10200\t1.10800\t1.09900\t1.10000\t1500\t600\t10\n");
    fprintf(f, "2020.01.03\t00:00\t1.10000\t1.10300\t1.09700\t1.10100\t800\t400\t12\n");
    fclose(f);

    rc = data_load_raw(good_path, &rows, &count);
    if (rc != 0) { fprintf(stderr, "  FAIL: cannot load good file\n"); remove(good_path); return 1; }
    ASSERT_INT(count, 3);

    for (i = 0; i < count; i++) {
        double mx = (rows[i].open > rows[i].close) ? rows[i].open : rows[i].close;
        double mn = (rows[i].open < rows[i].close) ? rows[i].open : rows[i].close;
        ASSERT_TRUE(rows[i].high >= mx);
        ASSERT_TRUE(rows[i].low  <= mn);
        ASSERT_TRUE(rows[i].open  > 0.0);
        ASSERT_TRUE(rows[i].close > 0.0);
    }
    free(rows);
    rows = NULL;
    remove(good_path);

    /* --- Part 2: broken file (missing HIGH column — tab gap) --- */
    f = fopen(bad_path, "w");
    if (!f) { fprintf(stderr, "Cannot create %s\n", bad_path); return 1; }
    fprintf(f, "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n");
    /* Missing HIGH: two tabs where high should be → atof("\t...") = 0 */
    fprintf(f, "2020.01.01\t00:00\t1.10000\t\t1.09500\t1.10200\t1000\t500\t10\n");
    fclose(f);

    rc = data_load_raw(bad_path, &rows, &count);
    if (rc == 0) {
        fprintf(stderr, "  FAIL: broken file loaded without error (validation missing)\n");
        free(rows);
        remove(bad_path);
        return 1;
    }
    /* rc != 0 → loader rejected invalid OHLC. This is the expected behavior. */
    remove(bad_path);

    printf("T-D2 PASS: valid file loads OK, broken file rejected by loader validation\n");
    return 0;
}
