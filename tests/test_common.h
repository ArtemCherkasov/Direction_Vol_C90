#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include "utils.h"
#include "data.h"
#include "types.h"
#include "vol.h"
#include "metrics.h"
#include "train.h"
#include "backtest.h"
#include "lstm.h"
#include "model.h"
#include "params.h"
#include "walkfwd.h"
#include "direction/data_direction.h"
#include "direction/predict_direction.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- Assertions ---- */

#define ASSERT_APPROX(val, exp, tol) do { \
    double _v = (double)(val), _e = (double)(exp), _t = (double)(tol); \
    if (fabs(_v - _e) > _t) { \
        fprintf(stderr, "  FAIL at %s:%d: expected %.12g, got %.12g (diff=%.3e, tol=%.3e)\n", \
                __FILE__, __LINE__, _e, _v, fabs(_v - _e), _t); \
        return 1; \
    } \
} while (0)

#define ASSERT_INT(val, exp) do { \
    int _v = (int)(val), _e = (int)(exp); \
    if (_v != _e) { \
        fprintf(stderr, "  FAIL at %s:%d: expected %d, got %d\n", \
                __FILE__, __LINE__, _e, _v); \
        return 1; \
    } \
} while (0)

#define ASSERT_STR(val, exp) do { \
    if (strcmp((val), (exp)) != 0) { \
        fprintf(stderr, "  FAIL at %s:%d: expected \"%s\", got \"%s\"\n", \
                __FILE__, __LINE__, (exp), (val)); \
        return 1; \
    } \
} while (0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL at %s:%d: condition false: %s\n", \
                __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

/* ---- Synthetic data helpers ---- */

/* Fill a DataRow with known values */
static void make_row(DataRow *r, const char *date, const char *time_,
                     double o, double h, double lo, double c,
                     int tv, int v, int sp) {
    strncpy(r->date, date, 15); r->date[15] = 0;
    strncpy(r->time_, time_, 15); r->time_[15] = 0;
    r->open = o; r->high = h; r->low = lo; r->close = c;
    r->tickvol = tv; r->vol = v; r->spread = sp;
}

/* Write DataRow array to temp CSV in 9-column MT5 format (with TIME column):
 *   date \t time \t open \t high \t low \t close \t tickvol \t vol \t spread
 */
static void write_test_csv(const char *path, DataRow *rows, int n) {
    FILE *f = fopen(path, "w");
    int i;
    if (!f) { fprintf(stderr, "Cannot write %s\n", path); return; }
    fprintf(f, "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n");
    for (i = 0; i < n; i++) {
        fprintf(f, "%s\t%s\t%.5f\t%.5f\t%.5f\t%.5f\t%d\t%d\t%d\n",
                rows[i].date, rows[i].time_,
                rows[i].open, rows[i].high, rows[i].low, rows[i].close,
                rows[i].tickvol, rows[i].vol, rows[i].spread);
    }
    fclose(f);
}

/* Simple deterministic LCG noise (not using rand()) */
static double lcg_noise(int i) {
    unsigned long state = (unsigned long)(i + 1) * 2654435761UL;
    state = state % 2147483647UL;
    return (double)state / 2147483647.0 - 0.5;  /* [-0.5, 0.5) */
}

/* Create a network that always outputs sigmoid(output_logit) regardless of input.
 * All gate weights zeroed → hidden state always 0 → output = sigmoid(bias_output). */
static LSTMNetwork *make_constant_net_ext(int num_features, double output_logit, int num_outputs);

static LSTMNetwork *make_constant_net(int num_features, double output_logit) {
    return make_constant_net_ext(num_features, output_logit, 1);
}

/* Extended variant: accepts num_outputs parameter for multi-horizon tests.
 * Sets bias_output_layer[k] = output_logit for k == num_outputs-1 (the horizon
 * output that backtest uses), and 0 for others. */
static LSTMNetwork *make_constant_net_ext(int num_features, double output_logit, int num_outputs) {
    LSTMNetwork *net = lstm_init(4, num_features, num_outputs, 42);
    int i;
    net->output_activation = ACTIVATION_SIGMOID;
    for (i = 0; i < net->weight_forget_recurrent->rows * net->weight_forget_recurrent->cols; i++) {
        net->weight_forget_recurrent->data[i] = 0.0;
        net->weight_input_recurrent->data[i] = 0.0;
        net->weight_output_recurrent->data[i] = 0.0;
        net->weight_candidate_recurrent->data[i] = 0.0;
    }
    for (i = 0; i < net->weight_forget_input->rows * net->weight_forget_input->cols; i++) {
        net->weight_forget_input->data[i] = 0.0;
        net->weight_input_input->data[i] = 0.0;
        net->weight_output_input->data[i] = 0.0;
        net->weight_candidate_input->data[i] = 0.0;
    }
    for (i = 0; i < net->weight_output_layer->rows * net->weight_output_layer->cols; i++) {
        net->weight_output_layer->data[i] = 0.0;
    }
    for (i = 0; i < net->num_cells; i++) {
        net->bias_input[i] = 0.0;
        net->bias_output[i] = 0.0;
        net->bias_candidate[i] = 0.0;
    }
    /* Set the bias on the last output (k = num_outputs-1) — the one backtest uses */
    for (i = 0; i < num_outputs; i++)
        net->bias_output_layer[i] = (i == num_outputs - 1) ? output_logit : 0.0;
    return net;
}

#endif /* TEST_COMMON_H */
