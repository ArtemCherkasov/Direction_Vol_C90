#ifndef DATA_H
#define DATA_H

#include "utils.h"

/* ==================== Mode constants ==================== */
#define MODE_OHLC 0
#define MODE_DIRECTION 1

/* ==================== Feature count ==================== */
#define NUM_BASE_FEATURES 13      /* base + technical + temporal features */
#define NUM_MTF_FEATURES  13      /* multi-timeframe features (D2 aggregation) */
#define NUM_RAW_FEATURES  (NUM_BASE_FEATURES + NUM_MTF_FEATURES)  /* 26 */

/* ==================== Configurable globals (set from parameters file) ==================== */
extern double g_train_ratio;        /* default 0.8 */
extern double g_volume_return_clip; /* default 10.0 */

/* ==================== Raw Data Row ==================== */
typedef struct {
    char date[16];
    char time_[16];
    double open;
    double high;
    double low;
    double close;
    int tickvol;
    int vol;
    int spread;
} DataRow;

/* ==================== Dataset ==================== */
typedef struct {
    double **features;   /* features[sample_idx][feature_dim] */
    double **labels;     /* labels[sample_idx][sequence_length * num_outputs] */
    int num_samples;
    int num_features;
    int num_outputs;
    int sequence_length;
    int mode;            /* 0=ohlc, 1=direction */

    /* Normalization parameters */
    double *feat_min;
    double *feat_max;
    char **feat_names;

    /* Label normalization (symmetric scaling by max_abs, per output) */
    double *label_min;
    double *label_max;

    /* Raw data for reference */
    DataRow *raw_data;
    int num_raw;
} Dataset;

/* ==================== Common Functions ==================== */

/* Load raw data from MetaTrader CSV (tab-delimited, quoted) */
int data_load_raw(const char *filepath, DataRow **out_rows, int *out_count);

/* Aggregate every h_period consecutive bars (no time alignment). */
/* Used for MTF features (D1 -> D2) where bars are already evenly spaced. */
void data_aggregate_raw_consecutive(DataRow *raw, int num_raw, int h_period,
                                    DataRow **out_raw, int *out_count);

/* Compute NUM_RAW_FEATURES features for all rows. */
double **data_compute_raw_features(DataRow *raw, int num_raw, int num_features,
                                    double *feat_min, double *feat_max,
                                    char **feat_names);

/* Free raw features array */
void data_free_raw_features(double **raw_features, int num_raw);

/* Allocate Dataset struct with features filled (no labels yet) */
Dataset *data_alloc_dataset(DataRow *raw, int num_raw, int seq_len,
                             int prediction_horizon, int num_outputs, int mode);

/* Intersect two chronologically-sorted raw arrays by date+time key */
int data_intersect_raw(DataRow *pri, int n_pri, DataRow *sec, int n_sec,
                       DataRow **out_pri, DataRow **out_sec, int *out_n);

/* Allocate Dataset with optional secondary features appended after primary */
Dataset *data_alloc_dataset_ext(DataRow *raw, int num_raw, int seq_len,
                                int prediction_horizon, int num_outputs, int mode,
                                double **sec_features, int num_sec_features,
                                double *sec_feat_min, double *sec_feat_max,
                                char **sec_feat_names);

/* Free dataset */
void data_free_dataset(Dataset *ds);

/* Create training/test split (chronological) */
typedef struct {
    Dataset *train;
    Dataset *test;
    int test_start_sample;  /* index of first test sample in original dataset (train_count + purge) */
} DatasetSplit;

DatasetSplit data_split_train_test(Dataset *ds, double train_ratio);

/* Save normalization params to text file */
int data_save_normalization(const char *filepath, Dataset *ds);

/* Load normalization params from text file */
int data_load_normalization(const char *filepath, Dataset *ds);

/* Normalize features in-place using feat_min/feat_max (full dataset) */
void data_normalize_features(Dataset *ds);

/* Free a train/test split (does not free shared normalization params) */
void data_free_split(DatasetSplit *split);

/* ==================== NEW: Train-only normalization (no leak) ==================== */

/* Compute min/max on a specific sample range [start, end) only.
 * Stores results in ds->feat_min / ds->feat_max. */
void data_compute_train_stats(Dataset *ds, int start_sample, int end_sample);

/* Normalize features in-place using provided min/max arrays.
 * Used to normalize test/val with TRAIN stats (no leak). */
void data_normalize_with_stats(Dataset *ds, const double *min_arr, const double *max_arr);

/* ==================== NEW: Train/Val split ==================== */

typedef struct {
    Dataset *train;
    Dataset *val;
} ValSplit;

/* Split train dataset into train/val by chronological tail.
 * val_count = (int)(train_count * val_ratio).
 * train = first (train_count - val_count) samples,
 * val   = last val_count samples. */
ValSplit data_split_train_val(Dataset *train_ds, double val_ratio);
void data_free_val_split(ValSplit *vs);

/* ==================== NEW: Date boundary helpers ==================== */

/* Find row index for a date+time boundary (binary search).
 * Returns index of first row >= boundary, or num_raw if all before. */
int data_find_date_boundary(DataRow *raw, int num_raw, const char *date, const char *time_);

/* Count bars in a date range [from_date, to_date] inclusive. */
int data_count_range(DataRow *raw, int num_raw, const char *from_date, const char *to_date);

/* Add N months to a date string "YYYY.MM.DD" -> out. */
void data_add_months(const char *date, int months, char *out, int out_sz);

/* Append a per-bar feature (e.g. predictive volatility) to every sample/timestep.
 * vol_per_bar[t] is the value for raw bar t (length = num_raw).
 * Sample i, timestep s uses raw bar (i+s), so the appended feature = vol_per_bar[i+s].
 * Expands features array from num_features to num_features+1 per timestep.
 * Reallocates feat_min/feat_max/feat_names. */
void data_append_vol_feature(Dataset *ds, const double *vol_per_bar, int num_raw,
                              const char *feat_name);

#endif /* DATA_H */
