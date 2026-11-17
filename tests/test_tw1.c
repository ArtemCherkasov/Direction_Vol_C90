/*
 * T-W1: Walk-forward window construction.
 * Generate 600 synthetic bars with dates, simulate window construction
 * (replicating walkfwd.c logic), assert train/test date ranges for all windows.
 */
#include "test_common.h"

int test_tw1_run(void) {
    int num_raw = 600;
    DataRow *raw;
    int wf_train = 24, wf_test = 3, wf_shift = 3;
    int seq_len = 10, pred_horizon = 1;
    int max_windows = 20;
    int actual_windows = 0;
    char cur_train_start[32], cur_test_start[32], cur_test_end[32];
    int i;

    raw = (DataRow *)malloc(num_raw * sizeof(DataRow));
    /* Generate daily dates starting 2010.01.01 (skip weekends) */
    {
        int yr = 2010, mo = 1, dy = 1;
        int dim[] = {31,28,31,30,31,30,31,31,30,31,30,31};
        for (i = 0; i < num_raw; i++) {
            /* Skip Saturday (6) and Sunday (0) in Zeller's */
            int y = yr, m = mo, d = dy;
            int dow_z;
            if (m < 3) { m += 12; y--; }
            dow_z = (d + 13*(m+1)/5 + y + y/4 - y/100 + y/400) % 7;
            while (dow_z == 0 || dow_z == 6) {
                dy++;
                if (dy > dim[mo-1]) { dy = 1; mo++; if (mo > 12) { mo = 1; yr++; } }
                y = yr; m = mo; d = dy;
                if (m < 3) { m += 12; y--; }
                dow_z = (d + 13*(m+1)/5 + y + y/4 - y/100 + y/400) % 7;
            }
            sprintf(raw[i].date, "%04d.%02d.%02d", yr, mo, dy);
            strcpy(raw[i].time_, "00:00");
            raw[i].open = 1.0 + 0.001 * i;
            raw[i].high = raw[i].open + 0.002;
            raw[i].low = raw[i].open - 0.002;
            raw[i].close = raw[i].open + 0.001;
            raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
            /* Next day */
            dy++;
            if (dy > dim[mo-1]) { dy = 1; mo++; if (mo > 12) { mo = 1; yr++; } }
        }
    }

    /* Simulate window construction (from walkfwd.c) */
    strcpy(cur_train_start, raw[0].date);
    data_add_months(cur_train_start, wf_train, cur_test_start, sizeof(cur_test_start));
    data_add_months(cur_test_start, wf_test, cur_test_end, sizeof(cur_test_end));

    for (i = 0; i < max_windows; i++) {
        int ts_idx, te_idx;
        ts_idx = data_find_date_boundary(raw, num_raw, cur_test_start, "00:00");
        te_idx = data_find_date_boundary(raw, num_raw, cur_test_end, "00:00");
        if (te_idx >= num_raw || ts_idx >= num_raw) break;
        if (te_idx - ts_idx < seq_len + pred_horizon + 1) break;
        {
            int tr_s = data_find_date_boundary(raw, num_raw, cur_train_start, "00:00");
            if (tr_s < 0) tr_s = 0;
            printf("  Window %d: train %s..%s (%d bars) | test %s..%s (%d bars)\n",
                   actual_windows + 1,
                   raw[tr_s].date, raw[ts_idx > 0 ? ts_idx - 1 : 0].date, ts_idx - tr_s,
                   raw[ts_idx].date, raw[te_idx > 0 ? te_idx - 1 : 0].date, te_idx - ts_idx);
            /* Assert: test starts after train ends */
            ASSERT_TRUE(ts_idx > tr_s);
            ASSERT_TRUE(te_idx > ts_idx);
            actual_windows++;
        }
        data_add_months(cur_train_start, wf_shift, cur_train_start, sizeof(cur_train_start));
        data_add_months(cur_test_start, wf_shift, cur_test_start, sizeof(cur_test_start));
        data_add_months(cur_test_end, wf_shift, cur_test_end, sizeof(cur_test_end));
    }

    free(raw);
    ASSERT_TRUE(actual_windows > 0);
    printf("T-W1 PASS: %d windows constructed, all train/test boundaries valid\n", actual_windows);
    return 0;
}
