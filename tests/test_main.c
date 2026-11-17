#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

#include "test_utils.h"

/* ==================== Test Runner ==================== */
typedef struct {
    const char *name;
    int (*func)(void);
} TestCase;

static TestCase tests[] = {
    /* Data leak tests */
    {"mtf_no_lookahead",       test_mtf_no_lookahead},
    {"target_consecutive",     test_target_consecutive_future},
    {"norm_train_only",        test_normalization_train_only},
    {"val_split_chrono",       test_val_split_chronological},
    {"date_boundary",          test_date_boundary},
    {"add_months",             test_add_months},

    /* Volatility tests */
    {"vol_ewma_series",        test_vol_ewma_series},
    {"vol_ewma_next",          test_vol_ewma_next},
    {"vol_garch_fit",          test_vol_garch_fit},
    {"vol_garch_constraints",  test_vol_garch_constraints},

    /* Metrics tests */
    {"metrics_pf",             test_metrics_profit_factor},
    {"metrics_sharpe",         test_metrics_sharpe},
    {"metrics_drawdown",       test_metrics_drawdown},
    {"metrics_brier",          test_metrics_brier},
    {"metrics_verdict",        test_metrics_verdict},

    /* Params tests */
    {"params_new_keys",        test_params_new_keys},
    {"params_patience_default",test_params_patience_default},

    /* Backtest tests */
    {"backtest_one_signal",    test_backtest_one_signal_per_bar},
    {"backtest_costs",         test_backtest_costs},

    /* Model tests */
    {"model_garch_roundtrip",  test_model_garch_roundtrip},
};

int main(void) {
    int num_tests = (int)(sizeof(tests) / sizeof(tests[0]));
    int passed = 0, i;

    printf("Running %d tests...\n\n", num_tests);

    for (i = 0; i < num_tests; i++) {
        int result;
        printf("[%2d/%2d] %-30s ", i + 1, num_tests, tests[i].name);
        fflush(stdout);
        result = tests[i].func();
        if (result) {
            printf("PASS\n");
            passed++;
        } else {
            printf("FAIL\n");
        }
    }

    printf("\n%d/%d tests passed.\n", passed, num_tests);
    return (passed == num_tests) ? 0 : 1;
}
