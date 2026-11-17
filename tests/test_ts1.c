/*
 * T-S1: Label ground truth.
 * closes = [1,2,1,3,2,4,1,5,3,6], seq=2, h=1.
 * Label formula: label[i][s] = (close[i+s+1] > close[i+s]) ? 1.0 : 0.0
 * Hardcoded expected table, exact match.
 * Verifies the linked data_direction.c formula.
 */
#include "test_common.h"

int test_ts1_run(void) {
    double closes[10] = {1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 3.0, 6.0};
    DataRow raw[10];
    Dataset *ds;
    int i, s;
    int seq = 2, h = 1;
    /* Expected labels: sample i, timestep s → (close[i+s+1] > close[i+s]) ? 1 : 0
     * Sample 0: s0=(2>1)=1  s1=(1>2)=0
     * Sample 1: s0=(1>2)=0  s1=(3>1)=1
     * Sample 2: s0=(3>1)=1  s1=(2>3)=0
     * Sample 3: s0=(2>3)=0  s1=(4>2)=1
     * Sample 4: s0=(4>2)=1  s1=(1>4)=0
     * Sample 5: s0=(1>4)=0  s1=(5>1)=1
     * Sample 6: s0=(5>1)=1  s1=(3>5)=0
     */
    double exp_labels[7][2] = {
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0}
    };

    for (i = 0; i < 10; i++) {
        sprintf(raw[i].date, "2020.01.%02d", i + 1);
        strcpy(raw[i].time_, "00:00");
        raw[i].open  = closes[i];
        raw[i].high  = closes[i] + 0.01;
        raw[i].low   = closes[i] - 0.01;
        raw[i].close = closes[i];
        raw[i].tickvol = 1000; raw[i].vol = 500; raw[i].spread = 10;
    }

    ds = data_build_dataset_direction(raw, 10, seq, h);
    if (!ds) { fprintf(stderr, "  FAIL: cannot build dataset\n"); return 1; }

    /* num_samples = 10 - 2 - 1 = 7 */
    ASSERT_INT(ds->num_samples, 7);
    ASSERT_INT(ds->sequence_length, seq);
    ASSERT_INT(ds->num_outputs, h);

    for (i = 0; i < 7; i++) {
        for (s = 0; s < seq; s++) {
            double got = ds->labels[i][s * h];
            double exp = exp_labels[i][s];
            if (got != exp) {
                fprintf(stderr, "  FAIL: sample %d step %d: expected %.1f got %.1f\n",
                        i, s, exp, got);
                data_free_dataset(ds);
                return 1;
            }
        }
    }

    data_free_dataset(ds);
    printf("T-S1 PASS: 7 samples x 2 steps, all labels match formula (close[i+s+1] > close[i+s])\n");
    return 0;
}
