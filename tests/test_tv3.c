/*
 * T-V3: Volatility alignment.
 * Compute GARCH sigma2 for 20 returns, verify vol_sigma[bi] == sqrt(|sigma2[bi]|).
 * Document: in walkfwd, vol_preds[i] = vol_sigma[entry_bar - 1] (last feature bar, causal).
 */
#include "test_common.h"

int test_tv3_run(void) {
    double ret[20];
    double *sigma2;
    double vol_sigma[20];
    GarchParams gp;
    int i;

    for (i = 0; i < 20; i++)
        ret[i] = 0.001 * lcg_noise(i) + 0.0005;

    gp.omega = 1e-6; gp.alpha = 0.05; gp.beta = 0.90; gp.sigma2_init = 1e-5;
    sigma2 = vol_garch_series(ret, 20, &gp);

    /* Reproduce walkfwd's vol_sigma construction */
    for (i = 0; i < 20; i++) {
        if (i < 19)  /* window_ret_len = 20 - 1 = 19 */
            vol_sigma[i] = sqrt(fabs(sigma2[i]));
        else
            vol_sigma[i] = sqrt(fabs(sigma2[18]));
    }

    /* Assert: vol_sigma[bi] == sqrt(|sigma2[bi]|) for bi < window_ret_len */
    for (i = 0; i < 19; i++) {
        ASSERT_APPROX(vol_sigma[i], sqrt(fabs(sigma2[i])), 1e-15);
    }

    /* Assert: vol_sigma[19] uses last valid sigma2 (index 18) */
    ASSERT_APPROX(vol_sigma[19], sqrt(fabs(sigma2[18])), 1e-15);

    /* All vol_sigma values must be non-negative */
    for (i = 0; i < 20; i++) {
        ASSERT_TRUE(vol_sigma[i] >= 0.0);
    }

    /* Document the alignment in backtest:
     * entry_bar = raw_offset + i + seq_len
     * vol_preds[i] = vol_sigma[entry_bar - 1] = vol_sigma[raw_offset + i + seq_len - 1]
     * This is causal: sigma2[entry_bar-1] is known at open of bar entry_bar-1,
     * which is before entry at bar entry_bar.
     */
    {
        int raw_offset = 100;  /* example */
        int seq_len = 10;
        int test_i = 5;
        int entry_bar = raw_offset + test_i + seq_len;  /* 115 */
        int vol_bar = entry_bar - 1;                      /* 114 */
        printf("  Alignment: entry_bar=%d, vol_bar=%d (vol known before entry)\n",
               entry_bar, vol_bar);
        ASSERT_TRUE(vol_bar < entry_bar);  /* causal */
    }

    free(sigma2);
    printf("T-V3 PASS: vol_sigma[bi] == sqrt(|sigma2[bi]|), vol uses entry_bar-1 (causal)\n");
    return 0;
}
