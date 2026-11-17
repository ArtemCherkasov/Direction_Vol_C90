/*
 * T-D3: Chronology and duplicates.
 * The loader does NOT sort or deduplicate — it reads rows in file order.
 * Part 1: Shuffled dates → assert loader preserves file order (no sort).
 * Part 2: Duplicate dates → assert both rows kept (no dedup).
 * This documents that caller must ensure chronology; otherwise labels read past data.
 */
#include "test_common.h"

int test_td3_run(void) {
    const char *path = "test_td3_tmp.csv";
    FILE *f;
    DataRow *rows = NULL;
    int count = 0, rc;

    /* --- Part 1: shuffled dates --- */
    f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return 1; }
    fprintf(f, "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n");
    fprintf(f, "2020.01.03\t00:00\t1.10000\t1.10500\t1.09500\t1.10200\t1000\t500\t10\n");
    fprintf(f, "2020.01.01\t00:00\t1.12000\t1.12500\t1.11800\t1.12300\t2000\t600\t10\n");
    fprintf(f, "2020.01.02\t00:00\t1.11000\t1.11300\t1.10700\t1.11100\t1500\t550\t12\n");
    fclose(f);

    rc = data_load_raw(path, &rows, &count);
    if (rc != 0) { fprintf(stderr, "  FAIL: cannot load\n"); remove(path); return 1; }
    ASSERT_INT(count, 3);

    /* Loader preserves file order — does NOT sort */
    ASSERT_STR(rows[0].date, "2020.01.03");
    ASSERT_STR(rows[1].date, "2020.01.01");
    ASSERT_STR(rows[2].date, "2020.01.02");

    /* Document: dates are out of chronological order */
    ASSERT_TRUE(strcmp(rows[1].date, rows[0].date) < 0);  /* 01.01 < 01.03 */

    free(rows);
    rows = NULL;
    remove(path);

    /* --- Part 2: duplicate dates --- */
    f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot create %s\n", path); return 1; }
    fprintf(f, "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n");
    fprintf(f, "2020.01.01\t00:00\t1.10000\t1.10500\t1.09500\t1.10200\t1000\t500\t10\n");
    fprintf(f, "2020.01.01\t00:00\t1.10200\t1.10800\t1.09900\t1.10000\t1500\t600\t10\n");
    fprintf(f, "2020.01.02\t00:00\t1.11000\t1.11300\t1.10700\t1.11100\t800\t400\t12\n");
    fclose(f);

    rc = data_load_raw(path, &rows, &count);
    if (rc != 0) { fprintf(stderr, "  FAIL: cannot load\n"); remove(path); return 1; }
    ASSERT_INT(count, 3);

    /* Both rows with date 2020.01.01 are kept — no dedup */
    ASSERT_STR(rows[0].date, "2020.01.01");
    ASSERT_STR(rows[1].date, "2020.01.01");
    ASSERT_STR(rows[2].date, "2020.01.02");
    ASSERT_TRUE(rows[0].open != rows[1].open);  /* different rows, same date */

    free(rows);
    remove(path);

    printf("T-D3 PASS: loader preserves file order (no sort, no dedup) — caller must ensure chronology\n");
    return 0;
}
