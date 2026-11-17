/*
 * T-L3: LSTM learnability.
 * Synthetic series where label = 1 if Friday (dow=4) else 0.
 * After training, val_loss < 0.3. If model can't learn this → bug in lstm/train.
 */
#include "test_common.h"

/* Simple date increment: day 0 = 2020.01.01 (Wednesday) */
static void make_date(int day_num, char *out) {
    int yr = 2020, mo = 1, dy = 1;
    int days_in_month[] = {31,29,31,30,31,30,31,31,30,31,30,31};
    int d;
    for (d = 0; d < day_num; d++) {
        dy++;
        if (dy > days_in_month[mo - 1]) {
            dy = 1; mo++;
            if (mo > 12) { mo = 1; yr++; }
        }
    }
    sprintf(out, "%04d.%02d.%02d", yr, mo, dy);
}

int test_tl3_run(void) {
    int num_raw = 300, seq = 5, h = 1;
    DataRow *raw;
    Dataset *ds;
    DatasetSplit split;
    ValSplit vs;
    LSTMNetwork *net;
    TrainResult tr;
    int i;
    double val_loss;

    raw = (DataRow *)malloc(num_raw * sizeof(DataRow));

    /* Build closes: up on Friday, down otherwise.
     * dow encoding: 0=Mon..4=Fri..6=Sun (from data.c day_of_week + conversion) */
    {
        double close = 1.0;
        for (i = 0; i < num_raw; i++) {
            int yr, mo, dy, dow_z, dow;
            make_date(i, raw[i].date);
            strcpy(raw[i].time_, "00:00");

            /* Compute dow the same way as data.c */
            sscanf(raw[i].date, "%d.%d.%d", &yr, &mo, &dy);
            {  /* Zeller's */
                int y = yr, m = mo;
                if (m < 3) { m += 12; y--; }
                dow_z = (dy + 13*(m+1)/5 + y + y/4 - y/100 + y/400) % 7;
            }
            dow = (dow_z >= 2) ? (dow_z - 2) : (dow_z + 5);

            raw[i].close = close;
            raw[i].open  = close;
            raw[i].high  = close + 0.005;
            raw[i].low   = close - 0.005;
            raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;

            /* Next bar direction: up on Friday (dow=4) */
            if (dow == 4)
                close += 0.01;   /* next bar up */
            else
                close -= 0.005;  /* next bar down */
        }
    }

    ds = data_build_dataset_direction(raw, num_raw, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); free(raw); return 1; }

    /* Split 80/20 */
    split = data_split_train_test(ds, 0.8);
    if (!split.train || !split.test) {
        fprintf(stderr, "  FAIL: split failed\n");
        data_free_dataset(ds); free(raw); return 1;
    }

    /* Split train into train/val (15% val) */
    vs = data_split_train_val(split.train, 0.15);
    if (!vs.train || !vs.val) {
        fprintf(stderr, "  FAIL: val split failed\n");
        data_free_split(&split); data_free_dataset(ds); free(raw); return 1;
    }

    printf("  train=%d val=%d test=%d\n", vs.train->num_samples, vs.val->num_samples, split.test->num_samples);

    /* Normalize with train stats */
    data_compute_train_stats(vs.train, 0, vs.train->num_samples);
    data_normalize_with_stats(vs.train, vs.train->feat_min, vs.train->feat_max);
    data_normalize_with_stats(vs.val, vs.train->feat_min, vs.train->feat_max);

    /* Init and train LSTM */
    net = lstm_init(8, ds->num_features, h, 42);
    net->output_activation = ACTIVATION_SIGMOID;

    tr = train_run(net, vs.train, vs.val,
                   0.01, 0.9, 16, 0.0001,
                   40, 20, 1, NULL);  /* no model save */

    val_loss = tr.best_val_loss;
    printf("  best_epoch=%d best_val_loss=%.6f epochs_actual=%d\n",
           tr.best_epoch, val_loss, tr.epochs_actual);

    lstm_free(net);
    data_free_val_split(&vs);
    data_free_split(&split);
    data_free_dataset(ds);
    free(raw);

    if (val_loss >= 0.3) {
        fprintf(stderr, "  FAIL: val_loss %.4f >= 0.3 — model cannot learn trivial pattern\n", val_loss);
        return 1;
    }

    printf("T-L3 PASS: val_loss=%.4f < 0.3 — model learned Friday pattern\n", val_loss);
    return 0;
}
