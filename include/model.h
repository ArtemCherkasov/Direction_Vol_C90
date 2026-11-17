#ifndef MODEL_H
#define MODEL_H

#include "lstm.h"
#include "data.h"
#include "types.h"
#include "vol.h"

/* Save network + normalization + GARCH params + train window dates */
int model_save(const char *filepath, LSTMNetwork *net,
               double epochs_trained, double best_train_loss,
               double best_val_loss, double val_accuracy,
               Dataset *norm_params, int h_period,
               const GarchParams *garch,
               const char *train_start_date, const char *train_end_date);

/* Load network from file */
int model_load(const char *filepath, LSTMNetwork *net);

/* Load GARCH params from model file. Returns 0 on success, -1 if not found. */
int model_load_garch(const char *filepath, GarchParams *garch);

/* Load normalization stats from model file into arrays. */
int model_load_normalization(const char *filepath,
                              double *feat_min, double *feat_max, int num_features,
                              double *label_min, double *label_max, int num_outputs);

#endif /* MODEL_H */
