/*
 * T-V2: Volatility recurrence — verify EWMA and GARCH match their formulas.
 * 5 known returns, lambda=0.94. Check each sigma2[i] against hand-computed formula.
 * GARCH with known params {omega=1e-6, alpha=0.05, beta=0.90, sigma2_init=1e-5}.
 */
#include "test_common.h"

int test_tv2_run(void) {
    double ret[5] = {0.01, -0.02, 0.005, 0.03, -0.01};
    double lambda = 0.94;
    double *s2;
    double exp_s2[5];
    int i;

    /* --- EWMA --- */
    s2 = vol_ewma(ret, 5, lambda);

    /* Formula: sigma2[0] = ret[0]^2, sigma2[i] = lambda*sigma2[i-1] + (1-lambda)*ret[i-1]^2 */
    exp_s2[0] = ret[0] * ret[0];
    for (i = 1; i < 5; i++)
        exp_s2[i] = lambda * exp_s2[i-1] + (1.0 - lambda) * ret[i-1] * ret[i-1];

    for (i = 0; i < 5; i++)
        ASSERT_APPROX(s2[i], exp_s2[i], 1e-15);

    /* Hardcoded spot checks */
    ASSERT_APPROX(s2[0], 0.0001, 1e-15);       /* 0.01^2 */
    ASSERT_APPROX(s2[1], 0.0001, 1e-15);       /* 0.94*0.0001 + 0.06*0.0001 = 0.0001 */
    ASSERT_APPROX(s2[2], 0.000118, 1e-12);     /* 0.94*0.0001 + 0.06*0.0004 */

    free(s2);

    /* --- GARCH --- */
    {
        GarchParams gp;
        gp.omega = 1e-6; gp.alpha = 0.05; gp.beta = 0.90; gp.sigma2_init = 1e-5;

        s2 = vol_garch_series(ret, 5, &gp);

        /* Formula: sigma2[0] = sigma2_init, sigma2[i] = omega + alpha*ret[i-1]^2 + beta*sigma2[i-1] */
        exp_s2[0] = gp.sigma2_init;
        for (i = 1; i < 5; i++)
            exp_s2[i] = gp.omega + gp.alpha * ret[i-1] * ret[i-1] + gp.beta * exp_s2[i-1];

        for (i = 0; i < 5; i++)
            ASSERT_APPROX(s2[i], exp_s2[i], 1e-15);

        /* Hardcoded spot checks */
        ASSERT_APPROX(s2[0], 1e-5, 1e-15);
        /* s2[1] = 1e-6 + 0.05*0.0001 + 0.90*1e-5 = 0.000015 */
        ASSERT_APPROX(s2[1], 0.000015, 1e-15);
        /* s2[2] = 1e-6 + 0.05*0.0004 + 0.90*0.000015 = 0.0000345 */
        ASSERT_APPROX(s2[2], 0.0000345, 1e-15);

        free(s2);
    }

    printf("T-V2 PASS: EWMA and GARCH sigma2 match recurrence formula to 1e-15\n");
    return 0;
}
