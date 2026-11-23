/*
 * Unified test runner — single executable for all debug tests.
 * Each test file has its main renamed to test_<name>_run(void).
 * Returns: 0 = pass, 1 = fail (same convention as individual mains).
 */
#include "test_common.h"

typedef int (*test_fn)(void);

typedef struct {
    const char *name;
    test_fn fn;
    int slow;  /* 1 = slow (training/golden), 0 = quick */
} TestCase;

/* Declarations for all test functions */
extern int test_td1_run(void);
extern int test_td2_run(void);
extern int test_td3_run(void);
extern int test_td4_run(void);
extern int test_tm1_run(void);
extern int test_tm2_run(void);
extern int test_tf1_run(void);
extern int test_tf2_run(void);
extern int test_tf3_run(void);
extern int test_tf4_run(void);
extern int test_ts1_run(void);
extern int test_ts2_run(void);
extern int test_ts3_run(void);
extern int test_ts4_run(void);
extern int test_tn1_run(void);
extern int test_tn2_run(void);
extern int test_tn3_run(void);
extern int test_tv1_run(void);
extern int test_tv2_run(void);
extern int test_tv3_run(void);
extern int test_tl1_run(void);
extern int test_tl2_run(void);
extern int test_tl3_run(void);
extern int test_tl4_run(void);
extern int test_tl5_run(void);
extern int test_tb1_run(void);
extern int test_tb2_run(void);
extern int test_tb3_run(void);
extern int test_tb4_run(void);
extern int test_tb4b_run(void);
extern int test_tb5_run(void);
extern int test_tb6_run(void);
extern int test_te1_run(void);
extern int test_te2_run(void);
extern int test_te3_run(void);
extern int test_tw1_run(void);
extern int test_tw2_run(void);
extern int test_tw3_run(void);
extern int test_tg1_run(void);
extern int test_tg2_run(void);
extern int test_tg3_run(void);

static TestCase tests[] = {
    {"T-D1", test_td1_run, 0},
    {"T-D2", test_td2_run, 0},
    {"T-D3", test_td3_run, 0},
    {"T-D4", test_td4_run, 0},
    {"T-M1", test_tm1_run, 0},
    {"T-M2", test_tm2_run, 0},
    {"T-F1", test_tf1_run, 0},
    {"T-F2", test_tf2_run, 0},
    {"T-F3", test_tf3_run, 0},
    {"T-F4", test_tf4_run, 0},
    {"T-S1", test_ts1_run, 0},
    {"T-S2", test_ts2_run, 0},
    {"T-S3", test_ts3_run, 0},
    {"T-S4", test_ts4_run, 0},
    {"T-N1", test_tn1_run, 0},
    {"T-N2", test_tn2_run, 0},
    {"T-N3", test_tn3_run, 0},
    {"T-V1", test_tv1_run, 0},
    {"T-V2", test_tv2_run, 0},
    {"T-V3", test_tv3_run, 0},
    {"T-L1", test_tl1_run, 0},
    {"T-L2", test_tl2_run, 0},
    {"T-L3", test_tl3_run, 1},
    {"T-L4", test_tl4_run, 0},
    {"T-L5", test_tl5_run, 1},
    {"T-B1", test_tb1_run, 0},
    {"T-B2", test_tb2_run, 0},
    {"T-B3", test_tb3_run, 0},
    {"T-B4",  test_tb4_run,  0},
    {"T-B4b", test_tb4b_run, 0},
    {"T-B5",  test_tb5_run,  0},
    {"T-B6",  test_tb6_run,  0},
    {"T-E1", test_te1_run, 0},
    {"T-E2", test_te2_run, 0},
    {"T-E3", test_te3_run, 0},
    {"T-W1", test_tw1_run, 0},
    {"T-W2", test_tw2_run, 0},
    {"T-W3", test_tw3_run, 0},
    {"T-G1", test_tg1_run, 1},
    {"T-G2", test_tg2_run, 0},
    {"T-G3", test_tg3_run, 1},
};

int main(int argc, char **argv) {
    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0, failed = 0, skipped = 0;
    int i;
    int run_slow = 0;  /* default: skip slow tests */

    if (argc > 1 && strcmp(argv[1], "--all") == 0)
        run_slow = 1;

    printf("=== Direction_Vol_C90 Test Suite ===\n");
    printf("Run %d tests%s\n\n", num_tests, run_slow ? " (including slow)" : " (quick only, use --all for slow)");

    for (i = 0; i < num_tests; i++) {
        int rc;

        if (tests[i].slow && !run_slow) {
            printf("[%2d/%2d] %-6s  SKIP (slow)\n", i + 1, num_tests, tests[i].name);
            skipped++;
            continue;
        }

        printf("[%2d/%2d] %-6s  ", i + 1, num_tests, tests[i].name);
        fflush(stdout);

        rc = tests[i].fn();

        if (rc == 0) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
            failed++;
        }
    }

    printf("\n=== Summary: %d passed, %d failed, %d skipped (total %d) ===\n",
           passed, failed, skipped, num_tests);

    return (failed > 0) ? 1 : 0;
}
