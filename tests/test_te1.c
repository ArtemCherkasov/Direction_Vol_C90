/*
 * T-E1: Diagnostic metrics exact values.
 * 4 (prediction, target) pairs → exact accuracy, Brier score, log loss.
 */
#include "test_common.h"

int test_te1_run(void) {
    double preds[4]  = {0.8, 0.3, 0.6, 0.1};
    double targets[4] = {1.0, 0.0, 1.0, 0.0};
    DiagnosticMetrics dm;

    dm = metrics_diagnostic(preds, targets, 4);

    /* Accuracy: all 4 correct (0.8>0.5→1=t, 0.3<0.5→0=t, 0.6>0.5→1=t, 0.1<0.5→0=t) */
    ASSERT_APPROX(dm.accuracy, 1.0, 1e-9);

    /* Brier = mean((0.8-1)^2 + (0.3-0)^2 + (0.6-1)^2 + (0.1-0)^2)
     *       = mean(0.04 + 0.09 + 0.16 + 0.01) = 0.30/4 = 0.075 */
    ASSERT_APPROX(dm.brier_score, 0.075, 1e-9);

    /* Log loss = mean(BCE(0.8,1) + BCE(0.3,0) + BCE(0.6,1) + BCE(0.1,0))
     * BCE(0.8,1) = -log(0.8) = 0.2231435513...
     * BCE(0.3,0) = -log(0.7) = 0.3566749439...
     * BCE(0.6,1) = -log(0.6) = 0.5108256238...
     * BCE(0.1,0) = -log(0.9) = 0.1053605157...
     * Mean = 1.1959946347.../4 = 0.2989986587... */
    {
        double exp_logloss = (-log(0.8) + -log(0.7) + -log(0.6) + -log(0.9)) / 4.0;
        ASSERT_APPROX(dm.log_loss, exp_logloss, 1e-9);
    }

    ASSERT_INT(dm.num_predictions, 4);

    printf("T-E1 PASS: acc=1.0, brier=0.075, logloss=%.6f (exact)\n", dm.log_loss);
    return 0;
}
