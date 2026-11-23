#ifndef TRAIN_H
#define TRAIN_H

#include "lstm.h"
#include "data.h"
#include "types.h"
#include "vol.h"

/* ==================== Training Result ==================== */
typedef struct {
    int best_epoch;
    double best_val_loss;
    double train_loss_at_best;
    double val_loss_at_best;
    int epochs_actual;
} TrainResult;

/* Train LSTM on a train/val split with val-loss model selection.
 * Model selection: save epoch with lowest val_loss (NOT test loss).
 * Early stopping: stop if val_loss hasn't improved for patience epochs. */
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
                       const char *train_end_date);

/* ==================== Loss/Accuracy (last timestep only) ==================== */

double train_loss_last_timestep(LSTMNetwork *net, Dataset *ds);
double train_accuracy_last_timestep(LSTMNetwork *net, Dataset *ds);
double train_brier_score(LSTMNetwork *net, Dataset *ds);
double train_log_loss(LSTMNetwork *net, Dataset *ds);

#endif /* TRAIN_H */
