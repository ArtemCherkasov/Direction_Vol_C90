/*
 * T-L2: Sigmoid output range.
 * Feed extreme inputs (+100, -100) to a sigmoid-activated LSTM.
 * Assert all outputs in (0.001, 0.999) — no saturation to 0 or 1.
 */
#include "test_common.h"

int test_tl2_run(void) {
    LSTMNetwork *net;
    int cells = 8, features = 3, outputs = 1, seq_len = 5;
    double seq_pos[15], seq_neg[15];  /* seq_len * features */
    double out[5];
    int i;

    for (i = 0; i < 15; i++) {
        seq_pos[i] = 100.0;
        seq_neg[i] = -100.0;
    }

    net = lstm_init(cells, features, outputs, 42);
    net->output_activation = ACTIVATION_SIGMOID;

    /* Positive extreme inputs */
    lstm_predict(net, seq_pos, seq_len, out);
    for (i = 0; i < seq_len; i++) {
        if (out[i] <= 0.001 || out[i] >= 0.999) {
            fprintf(stderr, "  FAIL: pos input output[%d] = %.6g outside (0.001, 0.999)\n", i, out[i]);
            lstm_free(net);
            return 1;
        }
    }

    /* Negative extreme inputs */
    lstm_predict(net, seq_neg, seq_len, out);
    for (i = 0; i < seq_len; i++) {
        if (out[i] <= 0.001 || out[i] >= 0.999) {
            fprintf(stderr, "  FAIL: neg input output[%d] = %.6g outside (0.001, 0.999)\n", i, out[i]);
            lstm_free(net);
            return 1;
        }
    }

    /* Zero inputs */
    {
        double seq_zero[15];
        for (i = 0; i < 15; i++) seq_zero[i] = 0.0;
        lstm_predict(net, seq_zero, seq_len, out);
        for (i = 0; i < seq_len; i++) {
            if (out[i] <= 0.001 || out[i] >= 0.999) {
                fprintf(stderr, "  FAIL: zero input output[%d] = %.6g outside (0.001, 0.999)\n", i, out[i]);
                lstm_free(net);
                return 1;
            }
        }
    }

    lstm_free(net);
    printf("T-L2 PASS: all outputs in (0.001, 0.999) for extreme inputs\n");
    return 0;
}
