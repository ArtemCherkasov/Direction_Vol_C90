# Direction_Vol_C90

LSTM + GARCH walk-forward backtest for EURUSD daily direction prediction.

## Build

```bash
cmake -B build -S .
cmake --build build
```

Requires: CMake 3.22+, GCC with C99 support.

## Run

### Train (walk-forward backtest)
```bash
./build/Direction_Vol_C90 --mode train --parameters parameters.cfg
```

### Predict (load saved model)
```bash
./build/Direction_Vol_C90 --mode predict --parameters parameters.cfg --model model_wf_window1.txt
```

## Tests

```bash
./build/test_all          # quick tests
./build/test_all --all    # all tests including slow
```

## Pipeline

1. Load EURUSD daily bars from CSV
2. Compute 26 raw features (returns, RSI, MACD, BB, ATR, Stoch, MTF) on full series
3. Fit GARCH(1,1) on train log-returns → predictive volatility (feature 27)
4. Walk-forward: 24-month train, 3-month test, 3-month shift
5. Train LSTM (early stopping on val loss, best weights reloaded)
6. Backtest: close-to-close execution, vol-targeted position sizing
7. Pooled aggregate metrics across all windows

## Key Design Decisions

- **Close-to-close execution**: trade entry/exit uses close prices. Entry is at
  `close[t]` — the same price used in features for bar `t`. For daily FX this is
  optimistic: the close cannot be known before it is printed. The label and PnL
  are consistent with each other, but results should be validated with an
  `open[t+1]` execution sensitivity test.
- **Prediction horizon**: `output = N` sets `num_outputs = N` and `hold_bars = N`.
  Backtest uses output `k = hold_bars - 1` (the N-bar horizon), not `k = 0`.
- **Non-overlapping entries**: with `hold_bars > 1`, trades enter only on every
  `hold_bars`-th bar, ensuring at most one position is open at any time.
- **Features on full series**: indicators (RSI, MACD, etc.) are computed once on
  the entire raw series, not per-window, so the same date gets identical feature
  values across windows.
- **Purge gap**: `seq_len` samples skipped between train and test to prevent feature overlap
- **Train-only normalization**: feat_min/feat_max computed on train only, applied to val/test
- **Pooled metrics**: all trades from all windows pooled into one metrics_compute call
- **Best weight restoration**: model_load called after train_run to reload best epoch weights

## Config (parameters.cfg)

Key parameters: `cells`, `epochs`, `patience`, `seq`, `lr`, `output`,
`max_windows`, `wf_train_months`, `wf_test_months`, `wf_shift`, `vol_mode`,
`val_ratio`, `cost_spread`, `threshold_k`, `prediction_threshold`.
