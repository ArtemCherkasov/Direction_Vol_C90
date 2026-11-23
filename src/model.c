#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ==================== Load section identifiers ==================== */
#define SECTION_NONE 0
#define SECTION_NORMALIZATION 1
#define SECTION_LSTM_WEIGHTS 2
#define SECTION_OUTPUT_WEIGHTS 3
#define SECTION_GARCH_PARAMS 4
#define SECTION_TRAIN_WINDOW 5

/* ==================== Load gate identifiers ==================== */
#define GATE_FORGET 0
#define GATE_INPUT 1
#define GATE_OUTPUT 2
#define GATE_CANDIDATE 3

/* ==================== Load matrix type identifiers ==================== */
#define MATRIX_TYPE_RECURRENT 0
#define MATRIX_TYPE_INPUT 1
#define MATRIX_TYPE_BIAS 2

/* ==================== Save ==================== */

int model_save(const char *filepath, LSTMNetwork *net,
               double epochs_trained, double best_train_loss,
               double best_val_loss, double val_accuracy,
               Dataset *norm_params, int h_period,
               const GarchParams *garch,
               const char *train_start_date, const char *train_end_date) {
    FILE *f;
    int n, f_, combined, num_out, i, j;

    f = fopen(filepath, "w");
    if (!f) {
        fprintf(stderr, "[MODEL SAVE ERROR] Cannot open %s\n", filepath);
        return -1;
    }

    n = net->num_cells;
    f_ = net->num_features;
    num_out = net->num_outputs;
    combined = n + f_;

    fprintf(f, "# LSTM Model - Text Format\n");
    fprintf(f, "# num_cells: %d\n", n);
    fprintf(f, "# num_features: %d\n", f_);
    fprintf(f, "# num_outputs: %d\n", num_out);
    if (norm_params && norm_params->mode == MODE_DIRECTION)
        fprintf(f, "# predict_count: %d\n", num_out);
    else
        fprintf(f, "# predict_count: %d\n", num_out / OUTPUTS_PER_CANDLE);
    fprintf(f, "# mode: %s\n", (norm_params && norm_params->mode == MODE_DIRECTION) ? "direction" : "ohlc");
    fprintf(f, "# output_activation: %d\n", net->output_activation);
    fprintf(f, "# seq_len: %d\n", norm_params ? norm_params->sequence_length : 0);
    fprintf(f, "# h_period: %d\n", h_period);
    fprintf(f, "# epochs_trained: %.0f\n", epochs_trained);
    if (norm_params && norm_params->mode == MODE_DIRECTION) {
        fprintf(f, "# best_train_bce: %.6f\n", best_train_loss);
        fprintf(f, "# best_val_bce: %.6f\n", best_val_loss);
        fprintf(f, "# val_accuracy: %.4f\n\n", val_accuracy);
    } else {
        fprintf(f, "# best_train_mse: %.6f\n", best_train_loss);
        fprintf(f, "# best_val_mse: %.6f\n", best_val_loss);
        fprintf(f, "# val_accuracy: %.4f\n\n", val_accuracy);
    }

    if (norm_params) {
        fprintf(f, "[normalization]\n");
        for (i = 0; i < norm_params->num_features; i++) {
            fprintf(f, "feature_name: %s | min: %.8f | max: %.8f\n",
                    norm_params->feat_names[i],
                    norm_params->feat_min[i],
                    norm_params->feat_max[i]);
        }
        fprintf(f, "label_min:");
        for (i = 0; i < norm_params->num_outputs; i++) {
            fprintf(f, " %.8f", norm_params->label_min[i]);
        }
        fprintf(f, "\n");
        fprintf(f, "label_max:");
        for (i = 0; i < norm_params->num_outputs; i++) {
            fprintf(f, " %.8f", norm_params->label_max[i]);
        }
        fprintf(f, "\n\n");
    }

    /* GARCH parameters */
    if (garch) {
        fprintf(f, "[garch_params]\n");
        fprintf(f, "omega: %.17g\n", garch->omega);
        fprintf(f, "alpha: %.17g\n", garch->alpha);
        fprintf(f, "beta: %.17g\n", garch->beta);
        fprintf(f, "sigma2_init: %.17g\n\n", garch->sigma2_init);
    }

    /* Train window dates */
    if (train_start_date && train_end_date) {
        fprintf(f, "[train_window]\n");
        fprintf(f, "start_date: %s\n", train_start_date);
        fprintf(f, "end_date: %s\n\n", train_end_date);
    }

    fprintf(f, "[lstm_weights]\n");

    #define SAVE_GATE(name) do { \
        fprintf(f, "# Gate: " #name "\n"); \
        fprintf(f, #name "_recurrent:\n"); \
        for (i = 0; i < n; i++) { \
            fprintf(f, "  row[%d]:", i); \
            for (j = 0; j < combined; j++) { \
                fprintf(f, " %.17g", net->weight_##name##_recurrent->data[i * combined + j]); \
            } \
            fprintf(f, "\n"); \
        } \
        fprintf(f, #name "_input:\n"); \
        for (i = 0; i < n; i++) { \
            fprintf(f, "  row[%d]:", i); \
            for (j = 0; j < f_; j++) { \
                fprintf(f, " %.17g", net->weight_##name##_input->data[i * f_ + j]); \
            } \
            fprintf(f, "\n"); \
        } \
        fprintf(f, #name "_bias:"); \
        for (i = 0; i < n; i++) { \
            fprintf(f, " %.17g", net->bias_##name[i]); \
        } \
        fprintf(f, "\n\n"); \
    } while (0)

    SAVE_GATE(forget);
    SAVE_GATE(input);
    SAVE_GATE(output);
    SAVE_GATE(candidate);
    #undef SAVE_GATE

    fprintf(f, "[output_weights]\n");
    fprintf(f, "weight_output_layer:\n");
    for (i = 0; i < num_out; i++) {
        fprintf(f, "  row[%d]:", i);
        for (j = 0; j < n; j++) {
            fprintf(f, " %.17g", net->weight_output_layer->data[i * n + j]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "bias_output_layer:");
    for (i = 0; i < num_out; i++) {
        fprintf(f, " %.17g", net->bias_output_layer[i]);
    }
    fprintf(f, "\n");

    fclose(f);
    return 0;
}

/* ==================== Load ==================== */

int model_load(const char *filepath, LSTMNetwork *net) {
    FILE *f;
    char line[8192];
    int n, f_, combined, num_out;
    int section;
    int gate;
    int matrix_type;
    double val;
    int parsed_row;
    int idx;
    int i;
    int dim;
    int rows_loaded = 0;
    int lines_processed = 0;
    int gate_set_count = 0;

    n = net->num_cells;
    f_ = net->num_features;
    num_out = net->num_outputs;
    combined = n + f_;

    f = fopen(filepath, "r");
    if (!f) return -1;

    /* Validate model dimensions against network */
    {
        int model_cells = -1, model_features = -1, model_outputs = -1;
        char vline[512];
        while (fgets(vline, sizeof(vline), f)) {
            utils_trim(vline);
            if (strncmp(vline, "# num_cells:", 12) == 0)
                model_cells = atoi(vline + 12);
            else if (strncmp(vline, "# num_features:", 15) == 0)
                model_features = atoi(vline + 15);
            else if (strncmp(vline, "# num_outputs:", 14) == 0)
                model_outputs = atoi(vline + 14);
            if (model_cells >= 0 && model_features >= 0 && model_outputs >= 0)
                break;
        }
        rewind(f);
        if (model_cells >= 0 && model_cells != n) {
            fprintf(stderr, "Error: model num_cells %d != network num_cells %d\n", model_cells, n);
            fclose(f);
            return -1;
        }
        if (model_features >= 0 && model_features != f_) {
            fprintf(stderr, "Error: model num_features %d != network num_features %d\n", model_features, f_);
            fclose(f);
            return -1;
        }
        if (model_outputs >= 0 && model_outputs != num_out) {
            fprintf(stderr, "Error: model num_outputs %d != network num_outputs %d\n", model_outputs, num_out);
            fclose(f);
            return -1;
        }
    }

    section = SECTION_NONE;
    gate = -1;
    matrix_type = MATRIX_TYPE_RECURRENT;

    while (fgets(line, sizeof(line), f)) {
        utils_trim(line);
        lines_processed++;

        if (line[0] == 0) continue;

        /* Detect sections */
        if (strcmp(line, "[normalization]") == 0) { section = SECTION_NORMALIZATION; continue; }
        if (strcmp(line, "[lstm_weights]") == 0) { section = SECTION_LSTM_WEIGHTS; continue; }
        if (strcmp(line, "[output_weights]") == 0) { section = SECTION_OUTPUT_WEIGHTS; continue; }
        if (strcmp(line, "[garch_params]") == 0) { section = SECTION_GARCH_PARAMS; continue; }
        if (strcmp(line, "[train_window]") == 0) { section = SECTION_TRAIN_WINDOW; continue; }

        /* Parse output_activation from metadata */
        if (section == SECTION_NONE && strncmp(line, "# output_activation:", 20) == 0) {
            net->output_activation = atoi(line + 20);
            continue;
        }

        /* Skip metadata comments but parse gate comments */
        if (section == SECTION_NONE) {
            if (line[0] == '#') continue;
            continue;
        }
        
        /* Parse gate comments within lstm_weights section */
        if (section == SECTION_LSTM_WEIGHTS && line[0] == '#') {
            continue;
        }

        /* Parse LSTM weights */
        if (section == SECTION_LSTM_WEIGHTS) {
            if (strncmp(line, "forget_recurrent:", strlen("forget_recurrent:")) == 0) { 
                gate = GATE_FORGET; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "forget_input:", strlen("forget_input:")) == 0) { 
                gate = GATE_FORGET; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "forget_bias:", strlen("forget_bias:")) == 0) { 
                gate = GATE_FORGET; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                /* Fall through to bias parsing — values are on this same line */
            }
            if (strncmp(line, "input_recurrent:", strlen("input_recurrent:")) == 0) { 
                gate = GATE_INPUT; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "input_input:", strlen("input_input:")) == 0) { 
                gate = GATE_INPUT; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "input_bias:", strlen("input_bias:")) == 0) { 
                gate = GATE_INPUT; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                /* Fall through to bias parsing — values are on this same line */
            }
            if (strncmp(line, "output_recurrent:", strlen("output_recurrent:")) == 0) { 
                gate = GATE_OUTPUT; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "output_input:", strlen("output_input:")) == 0) { 
                gate = GATE_OUTPUT; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "output_bias:", strlen("output_bias:")) == 0) { 
                gate = GATE_OUTPUT; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                /* Fall through to bias parsing — values are on this same line */
            }
            if (strncmp(line, "candidate_recurrent:", strlen("candidate_recurrent:")) == 0) { 
                gate = GATE_CANDIDATE; 
                matrix_type = MATRIX_TYPE_RECURRENT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "candidate_input:", strlen("candidate_input:")) == 0) { 
                gate = GATE_CANDIDATE; 
                matrix_type = MATRIX_TYPE_INPUT;
                gate_set_count++;
                continue; 
            }
            if (strncmp(line, "candidate_bias:", strlen("candidate_bias:")) == 0) { 
                gate = GATE_CANDIDATE; 
                matrix_type = MATRIX_TYPE_BIAS;
                gate_set_count++;
                /* Fall through to bias parsing — values are on this same line */
            }

            /* Row lines: "row[0]: v1 v2 v3 ..." */
            if (gate >= 0 && strncmp(line, "row[", strlen("row[")) == 0) {
                if (sscanf(line, "row[%d]:", &parsed_row) == 1) {
                    char *p = strchr(line, ':');
                    if (p && parsed_row >= 0 && parsed_row < n) {
                        p++;
                        while (*p == ' ') p++;
                        dim = (matrix_type == MATRIX_TYPE_RECURRENT) ? combined : f_;
                        for (i = 0; i < dim && *p; i++) {
                            val = strtod(p, &p);
                            if (gate == GATE_FORGET) {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_forget_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_forget_input->data[parsed_row * f_ + i] = val;
                            } else if (gate == GATE_INPUT) {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_input_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_input_input->data[parsed_row * f_ + i] = val;
                            } else if (gate == GATE_OUTPUT) {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_output_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_output_input->data[parsed_row * f_ + i] = val;
                            } else {
                                if (matrix_type == MATRIX_TYPE_RECURRENT)
                                    net->weight_candidate_recurrent->data[parsed_row * combined + i] = val;
                                else
                                    net->weight_candidate_input->data[parsed_row * f_ + i] = val;
                            }
                        }
                        rows_loaded++;
                    }
                }
                continue;
            }

            /* Bias lines: "forget_bias: v1 v2 v3 ..." */
            if (gate >= 0 && matrix_type == MATRIX_TYPE_BIAS) {
                char *p = line;
                while (*p && *p != ':') p++;
                if (*p == ':') p++;
                while (*p == ' ') p++;
                for (i = 0; i < n && *p; i++) {
                    val = strtod(p, &p);
                    if (gate == GATE_FORGET) net->bias_forget[i] = val;
                    else if (gate == GATE_INPUT) net->bias_input[i] = val;
                    else if (gate == GATE_OUTPUT) net->bias_output[i] = val;
                    else net->bias_candidate[i] = val;
                }
                continue;
            }
        }

        /* Parse output weights */
        if (section == SECTION_OUTPUT_WEIGHTS) {
            if (strncmp(line, "weight_output_layer:", strlen("weight_output_layer:")) == 0) continue;
            /* Row lines for output layer: "row[o]: w0 w1 ... wn" */
            if (strncmp(line, "row[", strlen("row[")) == 0) {
                if (sscanf(line, "row[%d]:", &parsed_row) == 1) {
                    char *p = strchr(line, ':');
                    if (p && parsed_row >= 0 && parsed_row < num_out) {
                        p++;
                        while (*p == ' ') p++;
                        for (i = 0; i < n && *p; i++) {
                            val = strtod(p, &p);
                            net->weight_output_layer->data[parsed_row * n + i] = val;
                        }
                    }
                }
                continue;
            }
            if (strncmp(line, "bias_output_layer:", strlen("bias_output_layer:")) == 0) {
                char *p = line + strlen("bias_output_layer:");
                while (*p == ' ') p++;
                for (i = 0; i < num_out && *p; i++) {
                    net->bias_output_layer[i] = strtod(p, &p);
                }
                continue;
            }
        }
    }

    fclose(f);

    /* Completeness check: verify all gate sections and rows were loaded */
    {
        int expected_gate_sets = 12;  /* 4 gates × 3 types (recurrent, input, bias) */
        int expected_rows = 8 * n;    /* 4 gates × 2 matrix types × n rows per matrix */
        if (gate_set_count != expected_gate_sets || rows_loaded != expected_rows) {
            fprintf(stderr, "Error: model file truncated — gate_sets=%d/%d rows=%d/%d\n",
                    gate_set_count, expected_gate_sets, rows_loaded, expected_rows);
            return -1;
        }
    }

    return 0;
}

/* ==================== Load GARCH params ==================== */

int model_load_garch(const char *filepath, GarchParams *garch) {
    FILE *f;
    char line[512];
    int found = 0;

    garch->omega = 0.0;
    garch->alpha = 0.1;
    garch->beta = 0.85;
    garch->sigma2_init = 1e-6;

    f = fopen(filepath, "r");
    if (!f) return -1;

    while (fgets(line, sizeof(line), f)) {
        utils_trim(line);
        if (strcmp(line, "[garch_params]") == 0) { found = 1; continue; }
        if (found) {
            if (line[0] == '[') break;
            if (strncmp(line, "omega:", 6) == 0) garch->omega = atof(line + 6);
            else if (strncmp(line, "alpha:", 6) == 0) garch->alpha = atof(line + 6);
            else if (strncmp(line, "beta:", 5) == 0) garch->beta = atof(line + 5);
            else if (strncmp(line, "sigma2_init:", 12) == 0) garch->sigma2_init = atof(line + 12);
        }
    }
    fclose(f);
    return found ? 0 : -1;
}

/* ==================== Load normalization stats ==================== */

int model_load_normalization(const char *filepath,
                              double *feat_min, double *feat_max, int num_features,
                              double *label_min, double *label_max, int num_outputs) {
    FILE *f;
    char line[8192];
    int count = 0;
    char name[64];
    double mn, mx;

    f = fopen(filepath, "r");
    if (!f) return -1;

    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (strncmp(line, "label_min:", 10) == 0) {
            char *p = line + 10; int j;
            for (j = 0; j < num_outputs && *p; j++) label_min[j] = strtod(p, &p);
            continue;
        }
        if (strncmp(line, "label_max:", 10) == 0) {
            char *p = line + 10; int j;
            for (j = 0; j < num_outputs && *p; j++) label_max[j] = strtod(p, &p);
            continue;
        }
        if (sscanf(line, "feature_name: %63s | min: %lf | max: %lf", name, &mn, &mx) == 3) {
            if (count < num_features) {
                feat_min[count] = mn;
                feat_max[count] = mx;
                count++;
            }
        }
    }
    fclose(f);
    return (count == num_features) ? 0 : -1;
}
