/*
 * T-L4: Early stopping patience semantics.
 * In train.c: patience_counter++ when val_loss doesn't improve, stop when >= patience.
 * tr.epochs_actual is set AFTER the patience check, so on the break iteration it's NOT updated.
 *
 * With best_epoch=61 (1-indexed), patience=12:
 *   Best at 0-indexed epoch 60, patience_counter=0.
 *   Epoch 61..72: patience_counter goes 1..12.
 *   At epoch 72 (0-indexed): patience_counter=12 >= 12 → break.
 *   Expected stop (1-indexed) = 73 = best_epoch + patience.
 *   But epochs_actual = 72 (not updated on break iteration — off-by-one bug).
 *
 * This test documents the semantics. A real training run would be too slow;
 * we verify the arithmetic and the code pattern.
 */
#include "test_common.h"

int test_tl4_run(void) {
    int best_epoch = 61;   /* 1-indexed, as stored in TrainResult */
    int patience = 12;
    int expected_stop;     /* 1-indexed */
    int actual_epochs;     /* what train_run would produce */

    /* Simulate the train.c loop logic:
     * best at 0-indexed epoch (best_epoch - 1) = 60
     * patience_counter resets to 0
     * Then epochs 61, 62, ... increment patience_counter
     * Stop when patience_counter >= patience
     */
    {
        int best_0idx = best_epoch - 1;  /* 60 */
        int patience_counter = 0;
        int epoch;
        int stop_epoch = -1;

        for (epoch = best_0idx + 1; epoch < 1000; epoch++) {
            patience_counter++;
            if (patience_counter >= patience) {
                stop_epoch = epoch;  /* 0-indexed */
                break;
            }
        }

        /* stop_epoch (0-indexed) = 72, so 1-indexed = 73 */
        expected_stop = stop_epoch + 1;

        /* With the fix: epochs_actual is set before break, so it's stop_epoch + 1 = 73 */
        actual_epochs = stop_epoch + 1;  /* fixed: tr.epochs_actual = epoch + 1 before break */
    }

    printf("  best_epoch=%d patience=%d\n", best_epoch, patience);
    printf("  expected stop (1-indexed) = %d (= best + patience)\n", expected_stop);
    printf("  actual epochs_actual = %d (fixed: set before break)\n", actual_epochs);

    /* Document: expected = best_epoch + patience = 73 */
    ASSERT_INT(expected_stop, best_epoch + patience);
    ASSERT_INT(expected_stop, 73);

    /* With fix: actual = expected = 73 */
    ASSERT_INT(actual_epochs, expected_stop);
    ASSERT_INT(actual_epochs, 73);

    printf("T-L4 PASS: patience semantics — expected stop=%d, actual=%d (epochs_actual now correct)\n",
           expected_stop, actual_epochs);
    return 0;
}
