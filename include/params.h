#ifndef PARAMS_H
#define PARAMS_H

#include "utils.h"

typedef struct {
    int cells; int has_cells;
    int epochs; int has_epochs;
    double lr; int has_lr;
    int seq; int has_seq;
    char mode[32]; int has_mode;
    int output; int has_output;
    char datetime[32]; int has_datetime;
    char from[32]; int has_from;
    char to[32]; int has_to;
    char data[256]; int has_data;
    char model[256]; int has_model;
    int h; int has_h;
    double wd; int has_wd;
    int patience; int has_patience;
    char sdata[256]; int has_sdata;
    int batch_size; int has_batch_size;
    double momentum; int has_momentum;
    unsigned int seed; int has_seed;
    double clip_min; int has_clip_min;
    double clip_max; int has_clip_max;
    double train_ratio; int has_train_ratio;
    double volume_return_clip; int has_volume_return_clip;
    double forget_gate_bias_init; int has_forget_gate_bias_init;
    double prediction_threshold; int has_prediction_threshold;

    /* --- NEW: Walk-forward --- */
    int wf_train_months; int has_wf_train_months;
    int wf_test_months;  int has_wf_test_months;
    int wf_shift;        int has_wf_shift;

    /* --- NEW: Volatility --- */
    char vol_mode[32]; int has_vol_mode;
    double ewma_lambda; int has_ewma_lambda;

    /* --- NEW: Backtest costs --- */
    double cost_spread;     int has_cost_spread;
    double cost_commission; int has_cost_commission;

    /* --- NEW: Dynamic threshold --- */
    double threshold_k; int has_threshold_k;

    /* --- NEW: Validation split --- */
    double val_ratio; int has_val_ratio;
} ParamsFile;

void parse_parameters_file(const char *path, ParamsFile *pf);

#endif /* PARAMS_H */
