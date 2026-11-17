#include "utils.h"
#include "data.h"
#include "lstm.h"
#include "params.h"
#include "model.h"
#include "vol.h"
#include "train.h"
#include "backtest.h"
#include "metrics.h"
#include "walkfwd.h"
#include "direction/data_direction.h"
#include "direction/predict_direction.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <math.h>

/* ==================== Globals ==================== */

static void signal_handler(int sig) {
    (void)sig;
    if (g_interrupted) {
        /* Second Ctrl+C — force exit */
        fprintf(stderr, "\nForce exit.\n");
        exit(1);
    }
    g_interrupted = 1;
    fprintf(stderr, "\n[Ctrl+C] Finishing current epoch and saving best model... (press again to force exit)\n");
}

/* ==================== Usage ==================== */
static void print_usage(const char *prog) {
    printf("Usage: %s --mode <mode> [options]\n\n", prog);
    printf("Modes:\n");
    printf("  train      Run full walk-forward training + backtest + report\n");
    printf("  predict    Load model, predict direction for last N bars\n\n");
    printf("Options:\n");
    printf("  --parameters <file>   Config file (key=value)\n");
    printf("  --mode <mode>         train | predict\n");
    printf("  --data <file>         Primary CSV\n");
    printf("  --sdata <file>        Secondary CSV (GBPUSD)\n");
    printf("  --cells N             LSTM cells\n");
    printf("  --epochs N            Max epochs\n");
    printf("  --lr F                Learning rate\n");
    printf("  --seq N               Sequence length (days)\n");
    printf("  --seed N              Random seed\n");
    printf("  --vol_mode garch|ewma Volatility mode\n");
}

/* ==================== CLI Parsing ==================== */
static void parse_cli(int argc, char **argv, ParamsFile *pf, const char **params_file) {
    int i;
    *params_file = NULL;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--parameters") == 0 && i + 1 < argc) {
            *params_file = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            strncpy(pf->mode, argv[++i], 31); pf->mode[31] = 0; pf->has_mode = 1;
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            strncpy(pf->data, argv[++i], 255); pf->data[255] = 0; pf->has_data = 1;
        } else if (strcmp(argv[i], "--sdata") == 0 && i + 1 < argc) {
            strncpy(pf->sdata, argv[++i], 255); pf->sdata[255] = 0; pf->has_sdata = 1;
        } else if (strcmp(argv[i], "--cells") == 0 && i + 1 < argc) {
            pf->cells = atoi(argv[++i]); pf->has_cells = 1;
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            pf->epochs = atoi(argv[++i]); pf->has_epochs = 1;
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            pf->lr = atof(argv[++i]); pf->has_lr = 1;
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            pf->seq = atoi(argv[++i]); pf->has_seq = 1;
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            pf->seed = (unsigned int)atoi(argv[++i]); pf->has_seed = 1;
        } else if (strcmp(argv[i], "--vol_mode") == 0 && i + 1 < argc) {
            strncpy(pf->vol_mode, argv[++i], 31); pf->vol_mode[31] = 0; pf->has_vol_mode = 1;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            strncpy(pf->model, argv[++i], 255); pf->model[255] = 0; pf->has_model = 1;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }
}

/* ==================== Train Mode ==================== */
static int run_train(ParamsFile *pf) {
    DataRow *raw = NULL, *sec_raw = NULL;
    int num_raw = 0, sec_num_raw = 0;
    WalkFwdResult wfr;

    /* Set globals */
    if (pf->has_train_ratio) g_train_ratio = pf->train_ratio;
    if (pf->has_volume_return_clip) g_volume_return_clip = pf->volume_return_clip;

    /* Load primary data (daily bars) */
    printf("Loading primary data: %s\n", pf->data);
    if (data_load_raw(pf->data, &raw, &num_raw) != 0) {
        fprintf(stderr, "Failed to load primary data.\n");
        return -1;
    }
    printf("  Loaded %d daily bars (%s - %s)\n", num_raw, raw[0].date, raw[num_raw - 1].date);

    /* Sanity check data */
    {
        int bad_ohlc = 0, bad_order = 0, t;
        for (t = 1; t < num_raw; t++) {
            if (raw[t].high < raw[t].open || raw[t].high < raw[t].close ||
                raw[t].low  > raw[t].open || raw[t].low  > raw[t].close ||
                raw[t].open <= 0 || raw[t].close <= 0) bad_ohlc++;
            if (strcmp(raw[t].date, raw[t-1].date) < 0) bad_order++;
        }
        printf("Sanity: bad_ohlc=%d bad_order=%d\n", bad_ohlc, bad_order);
    }

    /* Load secondary data if specified */
    if (pf->has_sdata && pf->sdata[0] != 0) {
        printf("Loading secondary data: %s\n", pf->sdata);
        if (data_load_raw(pf->sdata, &sec_raw, &sec_num_raw) != 0) {
            fprintf(stderr, "Warning: Failed to load secondary data, continuing without it.\n");
            sec_raw = NULL; sec_num_raw = 0;
        } else {
            printf("  Loaded %d daily bars\n", sec_num_raw);
        }
    }

    /* Run walk-forward */
    wfr = walkfwd_run(raw, num_raw, sec_raw, sec_num_raw, pf);

    /* Print report */
    walkfwd_print_report(&wfr);

    /* Cleanup */
    walkfwd_free(&wfr);
    if (raw) free(raw);
    if (sec_raw) free(sec_raw);

    return 0;
}

/* ==================== Predict Mode ==================== */
static int run_predict(ParamsFile *pf) {
    DataRow *raw = NULL;
    int num_raw = 0;
    int seq_len, num_cells, num_outputs, h_period;
    LSTMNetwork *net;
    Dataset *ds;
    double *preds;

    if (!pf->has_model || pf->model[0] == 0) {
        fprintf(stderr, "Error: --model <file> required for predict mode.\n");
        return -1;
    }

    seq_len = pf->has_seq ? pf->seq : 10;
    num_cells = pf->has_cells ? pf->cells : 63;
    num_outputs = pf->has_output ? pf->output : 1;

    printf("Loading data: %s\n", pf->data);
    if (data_load_raw(pf->data, &raw, &num_raw) != 0) {
        fprintf(stderr, "Failed to load data.\n");
        return -1;
    }

    ds = data_build_dataset_direction(raw, num_raw, seq_len, num_outputs);
    if (!ds) {
        fprintf(stderr, "Failed to build dataset.\n");
        free(raw);
        return -1;
    }

    data_normalize_features(ds);

    net = lstm_init(num_cells, ds->num_features, num_outputs, 42);
    net->output_activation = ACTIVATION_SIGMOID;

    printf("Loading model: %s\n", pf->model);
    if (model_load(pf->model, net) != 0) {
        fprintf(stderr, "Failed to load model.\n");
        data_free_dataset(ds);
        lstm_free(net);
        free(raw);
        return -1;
    }

    /* Demo predictions on last 5 samples */
    demo_direction(net, ds, num_outputs);

    /* Cleanup */
    preds = NULL;
    data_free_dataset(ds);
    lstm_free(net);
    free(raw);

    return 0;
}

/* ==================== Main ==================== */
int main(int argc, char **argv) {
    ParamsFile pf;
    const char *params_file = NULL;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    memset(&pf, 0, sizeof(ParamsFile));

    /* Parse CLI first to get --parameters */
    parse_cli(argc, argv, &pf, &params_file);

    /* Load parameters file if specified */
    if (params_file) {
        ParamsFile file_params;
        parse_parameters_file(params_file, &file_params);

        /* Apply file params as defaults, CLI overrides */
        if (!pf.has_cells && file_params.has_cells) { pf.cells = file_params.cells; pf.has_cells = 1; }
        if (!pf.has_epochs && file_params.has_epochs) { pf.epochs = file_params.epochs; pf.has_epochs = 1; }
        if (!pf.has_lr && file_params.has_lr) { pf.lr = file_params.lr; pf.has_lr = 1; }
        if (!pf.has_seq && file_params.has_seq) { pf.seq = file_params.seq; pf.has_seq = 1; }
        if (!pf.has_mode && file_params.has_mode) { strcpy(pf.mode, file_params.mode); pf.has_mode = 1; }
        if (!pf.has_output && file_params.has_output) { pf.output = file_params.output; pf.has_output = 1; }
        if (!pf.has_h && file_params.has_h) { pf.h = file_params.h; pf.has_h = 1; }
        if (!pf.has_wd && file_params.has_wd) { pf.wd = file_params.wd; pf.has_wd = 1; }
        if (!pf.has_patience && file_params.has_patience) { pf.patience = file_params.patience; pf.has_patience = 1; }
        if (!pf.has_data && file_params.has_data) { strcpy(pf.data, file_params.data); pf.has_data = 1; }
        if (!pf.has_sdata && file_params.has_sdata) { strcpy(pf.sdata, file_params.sdata); pf.has_sdata = 1; }
        if (!pf.has_model && file_params.has_model) { strcpy(pf.model, file_params.model); pf.has_model = 1; }
        if (!pf.has_seed && file_params.has_seed) { pf.seed = file_params.seed; pf.has_seed = 1; }
        if (!pf.has_batch_size && file_params.has_batch_size) { pf.batch_size = file_params.batch_size; pf.has_batch_size = 1; }
        if (!pf.has_momentum && file_params.has_momentum) { pf.momentum = file_params.momentum; pf.has_momentum = 1; }
        if (!pf.has_clip_min && file_params.has_clip_min) { pf.clip_min = file_params.clip_min; pf.has_clip_min = 1; }
        if (!pf.has_clip_max && file_params.has_clip_max) { pf.clip_max = file_params.clip_max; pf.has_clip_max = 1; }
        if (!pf.has_train_ratio && file_params.has_train_ratio) { pf.train_ratio = file_params.train_ratio; pf.has_train_ratio = 1; }
        if (!pf.has_volume_return_clip && file_params.has_volume_return_clip) { pf.volume_return_clip = file_params.volume_return_clip; pf.has_volume_return_clip = 1; }
        if (!pf.has_forget_gate_bias_init && file_params.has_forget_gate_bias_init) { pf.forget_gate_bias_init = file_params.forget_gate_bias_init; pf.has_forget_gate_bias_init = 1; }
        if (!pf.has_prediction_threshold && file_params.has_prediction_threshold) { pf.prediction_threshold = file_params.prediction_threshold; pf.has_prediction_threshold = 1; }

        /* New params from file */
        if (!pf.has_wf_train_months && file_params.has_wf_train_months) { pf.wf_train_months = file_params.wf_train_months; pf.has_wf_train_months = 1; }
        if (!pf.has_wf_test_months && file_params.has_wf_test_months) { pf.wf_test_months = file_params.wf_test_months; pf.has_wf_test_months = 1; }
        if (!pf.has_wf_shift && file_params.has_wf_shift) { pf.wf_shift = file_params.wf_shift; pf.has_wf_shift = 1; }
        if (!pf.has_vol_mode && file_params.has_vol_mode) { strcpy(pf.vol_mode, file_params.vol_mode); pf.has_vol_mode = 1; }
        if (!pf.has_ewma_lambda && file_params.has_ewma_lambda) { pf.ewma_lambda = file_params.ewma_lambda; pf.has_ewma_lambda = 1; }
        if (!pf.has_cost_spread && file_params.has_cost_spread) { pf.cost_spread = file_params.cost_spread; pf.has_cost_spread = 1; }
        if (!pf.has_cost_commission && file_params.has_cost_commission) { pf.cost_commission = file_params.cost_commission; pf.has_cost_commission = 1; }
        if (!pf.has_threshold_k && file_params.has_threshold_k) { pf.threshold_k = file_params.threshold_k; pf.has_threshold_k = 1; }
        if (!pf.has_val_ratio && file_params.has_val_ratio) { pf.val_ratio = file_params.val_ratio; pf.has_val_ratio = 1; }
    }

    /* Apply defaults */
    if (!pf.has_patience) { pf.patience = 8; pf.has_patience = 1; }
    if (!pf.has_vol_mode) { strcpy(pf.vol_mode, "garch"); pf.has_vol_mode = 1; }

    /* Validate */
    if (!pf.has_mode) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(pf.mode, "train") == 0) {
        return run_train(&pf);
    } else if (strcmp(pf.mode, "predict") == 0) {
        return run_predict(&pf);
    } else {
        fprintf(stderr, "Unknown mode: %s\n", pf.mode);
        print_usage(argv[0]);
        return 1;
    }
}
