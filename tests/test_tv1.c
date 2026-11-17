/*
 * T-V1: Volatility causality.
 * 50 returns, compute EWMA and GARCH series.
 * Modify returns[25..49] → assert sigma2[0..24] unchanged.
 * sigma2[t] depends only on returns[0..t-1].
 */
#include "test_common.h"

#define TV1_N 50
#define TV1_MOD 25

int test_tv1_run(void) {
    double ret1[TV1_N], ret2[TV1_N];
    double *s2_ewma1, *s2_ewma2, *s2_garch1, *s2_garch2;
    GarchParams gp;
    double lambda = 0.94;
    int i;
    int violations_ewma = 0, violations_garch = 0;

    for (i = 0; i < TV1_N; i++) {
        ret1[i] = 0.001 * lcg_noise(i) + 0.0005;
        ret2[i] = ret1[i];
    }

    /* EWMA causality */
    s2_ewma1 = vol_ewma(ret1, TV1_N, lambda);
    ret2[TV1_MOD] += 0.01;  /* modify return at bar 25 */
    s2_ewma2 = vol_ewma(ret2, TV1_N, lambda);

    for (i = 0; i < TV1_MOD; i++) {
        double diff = fabs(s2_ewma1[i] - s2_ewma2[i]);
        if (diff > 1e-15) {
            if (violations_ewma < 3)
                fprintf(stderr, "  EWMA leak at %d: %.15g vs %.15g diff=%.3e\n",
                        i, s2_ewma1[i], s2_ewma2[i], diff);
            violations_ewma++;
        }
    }
    free(s2_ewma1);
    free(s2_ewma2);

    /* GARCH causality */
    for (i = 0; i < TV1_N; i++) ret2[i] = ret1[i];  /* reset */
    gp.omega = 1e-6; gp.alpha = 0.05; gp.beta = 0.90;
    gp.sigma2_init = 1e-5;

    s2_garch1 = vol_garch_series(ret1, TV1_N, &gp);
    ret2[TV1_MOD] += 0.01;
    s2_garch2 = vol_garch_series(ret2, TV1_N, &gp);

    for (i = 0; i < TV1_MOD; i++) {
        double diff = fabs(s2_garch1[i] - s2_garch2[i]);
        if (diff > 1e-15) {
            if (violations_garch < 3)
                fprintf(stderr, "  GARCH leak at %d: %.15g vs %.15g diff=%.3e\n",
                        i, s2_garch1[i], s2_garch2[i], diff);
            violations_garch++;
        }
    }
    free(s2_garch1);
    free(s2_garch2);

    if (violations_ewma > 0 || violations_garch > 0) {
        fprintf(stderr, "  FAIL: EWMA leaks=%d GARCH leaks=%d\n", violations_ewma, violations_garch);
        return 1;
    }

    printf("T-V1 PASS: EWMA and GARCH causal — modifying ret[%d] doesn't affect sigma2[0..%d]\n",
           TV1_MOD, TV1_MOD - 1);
    return 0;
}
