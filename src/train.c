#include "train.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Allocate cell states for a sequence */
static LSTMCellState *alloc_cell_states(int seq_len, int num_cells, int num_features) {
    LSTMCellState *cs;
    int t;
    int combined = num_cells + num_features;

    cs = (LSTMCellState *)utils_safe_calloc((size_t)seq_len, sizeof(LSTMCellState));
    for (t = 0; t < seq_len; t++) {
        cs[t].hidden_state = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].cell_state = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].forget_gate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].input_gate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].output_gate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].candidate = (double *)utils_safe_calloc((size_t)num_cells, sizeof(double));
        cs[t].concatenated = (double *)utils_safe_calloc((size_t)combined, sizeof(double));
    }
    return cs;
}

TrainResult train_run(LSTMNetwork *net,
                       Dataset *train_ds, Dataset *val_ds,
                       double lr, double momentum,
                       int batch_size, double weight_decay,
                       int max_epochs, int patience,
                       int window_num,
                       const char *model_path,
                       const GarchParams *garch,
                       int h_period,
                       const char *train_start_date,
                       const char *train_end_date) {
    TrainResult tr;
    int epoch, i, batch_start;
    int *indices;
    int num_train, num_val, seq_len;
    double *preds;
    LSTMCellState *cell_states;
    double best_val_loss;
    int patience_counter;
    int n, f, combined;

    tr.best_epoch = 0;
    tr.best_val_loss = 1e30;
    tr.train_loss_at_best = 1e30;
    tr.val_loss_at_best = 1e30;
    tr.epochs_actual = 0;

    num_train = train_ds->num_samples;
    num_val = val_ds ? val_ds->num_samples : 0;
    seq_len = train_ds->sequence_length;
    n = net->num_cells;
    f = net->num_features;
    combined = n + f;

    net->weight_decay = weight_decay;

    indices = (int *)utils_safe_malloc((size_t)num_train * sizeof(int));
    for (i = 0; i < num_train; i++) indices[i] = i;

    preds = (double *)utils_safe_calloc((size_t)(net->num_outputs * seq_len), sizeof(double));
    cell_states = alloc_cell_states(seq_len, n, f);

    best_val_loss = 1e30;
    patience_counter = 0;

    for (epoch = 0; epoch < max_epochs; epoch++) {
        double train_loss = 0.0;
        double val_loss = 0.0;
        int train_count = 0;

        /* Shuffle indices */
        {
            int j, tmp;
            for (j = num_train - 1; j > 0; j--) {
                int k = (int)(utils_rand() * (double)(j + 1));
                if (k > j) k = j;
                tmp = indices[j]; indices[j] = indices[k]; indices[k] = tmp;
            }
        }

        /* Batch training — also accumulate train_loss for logging */
        for (batch_start = 0; batch_start < num_train; batch_start += batch_size) {
            int batch_end = batch_start + batch_size;
            int actual_batch = 0;
            if (batch_end > num_train) batch_end = num_train;
            actual_batch = batch_end - batch_start;

            lstm_zero_gradients(net);

            for (i = batch_start; i < batch_end; i++) {
                int idx = indices[i];
                int t, o;
                lstm_forward(net, train_ds->features[idx], cell_states, seq_len, preds);
                lstm_backward(net, train_ds->features[idx], cell_states,
                              preds, train_ds->labels[idx], seq_len);
                /* Accumulate train loss from this forward pass */
                for (t = 0; t < seq_len; t++) {
                    for (o = 0; o < net->num_outputs; o++) {
                        double p = preds[t * net->num_outputs + o];
                        double tgt = train_ds->labels[idx][t * net->num_outputs + o];
                        train_loss += utils_binary_cross_entropy(p, tgt);
                        train_count++;
                    }
                }
            }

            lstm_update(net, lr, momentum, actual_batch);
        }

        train_loss /= (double)train_count;

        /* Compute val loss (for model selection) */
        if (num_val > 0) {
            int val_count = 0;
            for (i = 0; i < num_val; i++) {
                int t, o;
                lstm_forward(net, val_ds->features[i], cell_states, seq_len, preds);
                for (t = 0; t < seq_len; t++) {
                    for (o = 0; o < net->num_outputs; o++) {
                        double p = preds[t * net->num_outputs + o];
                        double tgt = val_ds->labels[i][t * net->num_outputs + o];
                        val_loss += utils_binary_cross_entropy(p, tgt);
                        val_count++;
                    }
                }
            }
            val_loss /= (double)val_count;
        }

        printf("Window %d | Epoch %d/%d | train_loss=%.6f | val_loss=%.6f | best_val=%.6f\n",
               window_num, epoch + 1, max_epochs, train_loss, val_loss, best_val_loss);

        /* Model selection by val loss — guard against NaN/Inf */
        if (isnan(val_loss) || isinf(val_loss)) {
            fprintf(stderr, "Window %d | Epoch %d | val_loss is NaN/Inf, skipping model save\n",
                   window_num, epoch + 1);
            patience_counter++;
        } else if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            tr.best_epoch = epoch + 1;
            tr.train_loss_at_best = train_loss;
            tr.val_loss_at_best = val_loss;
            patience_counter = 0;

            if (model_path) {
                model_save(model_path, net, (double)(epoch + 1), train_loss,
                           val_loss, 0.0, train_ds, h_period, garch,
                           train_start_date, train_end_date);
            }
        } else {
            patience_counter++;
        }

        /* Early stopping */
        if (patience > 0 && patience_counter >= patience) {
            tr.epochs_actual = epoch + 1;
            printf("Window %d | Early stopping at epoch %d (patience=%d)\n",
                   window_num, epoch + 1, patience);
            break;
        }

        /* Check for Ctrl+C interrupt */
        if (g_interrupted) {
            printf("Window %d | Interrupted at epoch %d. Saving best model (epoch %d, val_loss=%.6f)...\n",
                   window_num, epoch + 1, tr.best_epoch, best_val_loss);
            if (model_path) {
                model_save(model_path, net, (double)tr.best_epoch,
                           tr.train_loss_at_best, best_val_loss, 0.0,
                           train_ds, h_period, garch,
                           train_start_date, train_end_date);
                printf("Window %d | Best model saved to %s\n", window_num, model_path);
            }
            tr.epochs_actual = epoch + 1;
            break;
        }

        tr.epochs_actual = epoch + 1;
    }

    tr.best_val_loss = best_val_loss;
    free(indices);
    free(preds);
    lstm_free_cell_states(cell_states, seq_len);

    /* Reload best weights so callers get the best model, not the last epoch.
     * Without this, early stopping leaves net with degraded weights. */
    if (model_path) {
        if (model_load(model_path, net) != 0) {
            fprintf(stderr, "Window %d | Warning: failed to reload best model from %s, "
                   "using last epoch weights\n", window_num, model_path);
        }
    }

    return tr;
}

/* ==================== Last-timestep metrics ==================== */

double train_loss_last_timestep(LSTMNetwork *net, Dataset *ds) {
    int i, o;
    double total_loss = 0.0;
    double *preds;
    LSTMCellState *cell_states;
    int seq_len = ds->sequence_length;
    int n = net->num_cells;
    int combined = n + net->num_features;
    int last_base;

    cell_states = alloc_cell_states(seq_len, n, net->num_features);
    preds = (double *)utils_safe_calloc((size_t)(net->num_outputs * seq_len), sizeof(double));
    last_base = (seq_len - 1) * net->num_outputs;

    for (i = 0; i < ds->num_samples; i++) {
        lstm_forward(net, ds->features[i], cell_states, seq_len, preds);
        for (o = 0; o < net->num_outputs; o++) {
            double p = preds[last_base + o];
            double tgt = ds->labels[i][last_base + o];
            total_loss += utils_binary_cross_entropy(p, tgt);
        }
    }

    free(preds);
    lstm_free_cell_states(cell_states, seq_len);
    return total_loss / (double)ds->num_samples;
}

double train_accuracy_last_timestep(LSTMNetwork *net, Dataset *ds) {
    int i, o, correct = 0;
    double *preds;
    LSTMCellState *cell_states;
    int seq_len = ds->sequence_length;
    int n = net->num_cells;
    int last_base;

    cell_states = alloc_cell_states(seq_len, n, net->num_features);
    preds = (double *)utils_safe_calloc((size_t)(net->num_outputs * seq_len), sizeof(double));
    last_base = (seq_len - 1) * net->num_outputs;

    for (i = 0; i < ds->num_samples; i++) {
        lstm_forward(net, ds->features[i], cell_states, seq_len, preds);
        for (o = 0; o < net->num_outputs; o++) {
            int pred_up = (preds[last_base + o] > 0.5) ? 1 : 0;
            int act_up = (ds->labels[i][last_base + o] > 0.0) ? 1 : 0;
            if (pred_up == act_up) correct++;
        }
    }

    free(preds);
    lstm_free_cell_states(cell_states, seq_len);
    return (double)correct / ((double)ds->num_samples * (double)net->num_outputs);
}

double train_brier_score(LSTMNetwork *net, Dataset *ds) {
    int i, o;
    double sum = 0.0;
    double *preds;
    LSTMCellState *cell_states;
    int seq_len = ds->sequence_length;
    int n = net->num_cells;
    int last_base;

    cell_states = alloc_cell_states(seq_len, n, net->num_features);
    preds = (double *)utils_safe_calloc((size_t)(net->num_outputs * seq_len), sizeof(double));
    last_base = (seq_len - 1) * net->num_outputs;

    for (i = 0; i < ds->num_samples; i++) {
        lstm_forward(net, ds->features[i], cell_states, seq_len, preds);
        for (o = 0; o < net->num_outputs; o++) {
            double p = preds[last_base + o];
            double t = ds->labels[i][last_base + o];
            sum += (p - t) * (p - t);
        }
    }

    free(preds);
    lstm_free_cell_states(cell_states, seq_len);
    return sum / ((double)ds->num_samples * (double)net->num_outputs);
}

double train_log_loss(LSTMNetwork *net, Dataset *ds) {
    return train_loss_last_timestep(net, ds);
}
