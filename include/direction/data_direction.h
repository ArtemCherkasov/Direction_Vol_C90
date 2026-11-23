#ifndef DATA_DIRECTION_H
#define DATA_DIRECTION_H

#include "data.h"

/* Build dataset with corrected direction labels:
 * label[t][k] = 1 if close[t+1+k] > close[t+k] else 0
 * (compares consecutive future bars, NOT against close[t]) */
Dataset *data_build_dataset_direction(DataRow *raw, int num_raw, int seq_len, int prediction_horizon);

/* Build dataset for a window of the full raw series, computing features on the
 * FULL raw array so sequential indicators have proper warm-up from series start.
 * bar_offset: first bar index in the window (relative to full raw)
 * window_len: number of bars in the window */
Dataset *data_build_dataset_direction_window(DataRow *raw, int num_raw,
                                              int seq_len, int prediction_horizon,
                                              int bar_offset, int window_len);

Dataset *data_build_dataset_direction_ext(DataRow *raw, int num_raw, int seq_len, int prediction_horizon,
                                           double **sec_features, int num_sec_features,
                                           double *sec_feat_min, double *sec_feat_max,
                                           char **sec_feat_names);

/* Normalize direction labels (no-op, already in [0, 1]) */
void data_normalize_labels_direction(Dataset *ds);

#endif /* DATA_DIRECTION_H */
