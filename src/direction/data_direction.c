#include "../../include/direction/data_direction.h"
#include <stdlib.h>

/* Build dataset for a window of the full raw series, computing features on the
 * FULL raw array so that sequential indicators (RSI, MACD, BB, ATR, Stoch)
 * have proper warm-up from the start of the series. This eliminates the
 * train/test/predict feature divergence where the same date got different
 * feature values in different windows.
 *
 * bar_offset: index of first bar in the window (relative to full raw)
 * window_len: number of bars in the window
 * Samples are created for bars [bar_offset, bar_offset + window_len - seq - h)
 * Sample i, timestep s maps to raw bar (bar_offset + i + s). */
Dataset *data_build_dataset_direction_window(DataRow *raw, int num_raw,
                                              int seq_len, int prediction_horizon,
                                              int bar_offset, int window_len) {
    int num_outputs = prediction_horizon;
    int num_primary = NUM_RAW_FEATURES;
    int num_samples = window_len - seq_len - prediction_horizon;
    Dataset *ds;
    double **raw_features;
    double *feat_min, *feat_max;
    char **feat_names;
    int i, s, k, fi;

    if (num_samples <= 0 || bar_offset < 0 || bar_offset + window_len > num_raw) {
        fprintf(stderr, "Error: invalid window offset=%d len=%d for num_raw=%d\n",
                bar_offset, window_len, num_raw);
        return NULL;
    }

    /* Compute features on the FULL raw series — this is the key fix */
    feat_min = (double *)utils_safe_malloc((size_t)num_primary * sizeof(double));
    feat_max = (double *)utils_safe_malloc((size_t)num_primary * sizeof(double));
    feat_names = (char **)utils_safe_malloc((size_t)num_primary * sizeof(char *));
    raw_features = data_compute_raw_features(raw, num_raw, num_primary,
                                              feat_min, feat_max, feat_names);

    ds = (Dataset *)malloc(sizeof(Dataset));
    if (!ds) { data_free_raw_features(raw_features, num_raw); free(feat_min); free(feat_max); for (fi = 0; fi < num_primary; fi++) free(feat_names[fi]); free(feat_names); return NULL; }

    ds->num_samples = num_samples;
    ds->num_features = num_primary;
    ds->num_outputs = num_outputs;
    ds->sequence_length = seq_len;
    ds->mode = MODE_DIRECTION;
    ds->raw_data = raw;
    ds->num_raw = num_raw;
    ds->bar_offset = bar_offset;

    ds->features = (double **)utils_safe_malloc((size_t)num_samples * sizeof(double *));
    ds->labels = (double **)utils_safe_malloc((size_t)num_samples * sizeof(double *));
    ds->feat_min = feat_min;
    ds->feat_max = feat_max;
    ds->feat_names = feat_names;
    ds->label_min = (double *)utils_safe_malloc((size_t)num_outputs * sizeof(double));
    ds->label_max = (double *)utils_safe_malloc((size_t)num_outputs * sizeof(double));

    for (i = 0; i < num_samples; i++) {
        ds->features[i] = (double *)utils_safe_malloc((size_t)(seq_len * num_primary) * sizeof(double));
        ds->labels[i] = (double *)utils_safe_malloc((size_t)(seq_len * num_outputs) * sizeof(double));
        for (s = 0; s < seq_len; s++) {
            int ti = bar_offset + i + s;
            for (fi = 0; fi < num_primary; fi++)
                ds->features[i][s * num_primary + fi] = raw_features[ti][fi];
            for (k = 0; k < num_outputs; k++) {
                int idx_base = ti;
                int idx_next = ti + 1 + k;
                ds->labels[i][s * num_outputs + k] =
                    (raw[idx_next].close > raw[idx_base].close) ? 1.0 : 0.0;
            }
        }
    }

    for (k = 0; k < num_outputs; k++) {
        ds->label_min[k] = 0.0;
        ds->label_max[k] = 1.0;
    }

    data_free_raw_features(raw_features, num_raw);
    return ds;
}

Dataset *data_build_dataset_direction_ext(DataRow *raw, int num_raw, int seq_len, int prediction_horizon,
                                           double **sec_features, int num_sec_features,
                                           double *sec_feat_min, double *sec_feat_max,
                                           char **sec_feat_names) {
    int num_outputs = prediction_horizon;
    Dataset *ds;
    int i, s, k;

    ds = data_alloc_dataset_ext(raw, num_raw, seq_len, prediction_horizon, num_outputs, MODE_DIRECTION,
                                sec_features, num_sec_features, sec_feat_min, sec_feat_max, sec_feat_names);
    if (!ds) return NULL;

    for (i = 0; i < ds->num_samples; i++) {
        for (s = 0; s < seq_len; s++) {
            int si = i + s;
            for (k = 0; k < prediction_horizon; k++) {
                int idx_base = si;           /* close[t]     — cumulative from base */
                int idx_next = si + 1 + k;   /* close[t+1+k] */
                ds->labels[i][s * num_outputs + k] =
                    (raw[idx_next].close > raw[idx_base].close) ? 1.0 : 0.0;
            }
        }
    }

    for (k = 0; k < num_outputs; k++) {
        ds->label_min[k] = 0.0;
        ds->label_max[k] = 1.0;
    }

    return ds;
}

Dataset *data_build_dataset_direction(DataRow *raw, int num_raw, int seq_len, int prediction_horizon) {
    return data_build_dataset_direction_ext(raw, num_raw, seq_len, prediction_horizon,
                                            NULL, 0, NULL, NULL, NULL);
}

void data_normalize_labels_direction(Dataset *ds) {
    /* No-op: labels are already 0/1, no normalization needed */
    (void)ds;
}
