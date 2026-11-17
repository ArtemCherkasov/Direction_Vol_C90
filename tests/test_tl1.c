/*
 * T-L1: LSTM determinism.
 * Init net with seed=42, run lstm_predict twice on identical input.
 * Assert: outputs are bit-identical (diff == 0.0).
 */
#include "test_common.h"

int test_tl1_run(void) {
    LSTMNetwork *net;
    int cells = 4, features = 2, outputs = 1, seq_len = 5;
    double seq[10];  /* seq_len * features = 5 * 2 */
    double out1[5], out2[5];  /* seq_len * outputs = 5 * 1 */
    int i;
    int mismatches = 0;

    for (i = 0; i < 10; i++)
        seq[i] = 0.1 * (double)i - 0.3;

    net = lstm_init(cells, features, outputs, 42);
    net->output_activation = ACTIVATION_SIGMOID;

    lstm_predict(net, seq, seq_len, out1);
    lstm_predict(net, seq, seq_len, out2);

    for (i = 0; i < seq_len * outputs; i++) {
        double diff = fabs(out1[i] - out2[i]);
        if (diff > 0.0) {
            fprintf(stderr, "  MISMATCH at %d: %.17g vs %.17g diff=%.3e\n",
                    i, out1[i], out2[i], diff);
            mismatches++;
        }
    }

    lstm_free(net);

    if (mismatches > 0) {
        fprintf(stderr, "  FAIL: %d output mismatches in deterministic forward pass\n", mismatches);
        return 1;
    }

    printf("T-L1 PASS: two forward passes produce identical outputs (bit-exact)\n");
    return 0;
}
