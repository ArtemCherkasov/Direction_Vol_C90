# Direction_Vol_C90

LSTM + GARCH walk-forward backtest for EURUSD daily direction prediction.

## Build

```bash
cd build_linux
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

Requires: CMake 3.22+, GCC with C99 support, WSL (on Windows).

## Run

### Train (walk-forward backtest)
```bash
./Direction_Vol_C90 --mode train --parameters ../parameters.txt
```

### Predict (load saved model)
```bash
./Direction_Vol_C90 --mode predict --parameters ../parameters.txt --model model_wf_window1.txt
```

## Tests

```bash
./test_all          # quick tests (37)
./test_all --all    # all tests including slow (41)
```

## Pipeline

1. Load EURUSD daily bars from CSV
2. Compute 26 raw features (returns, RSI, MACD, BB, ATR, Stoch, MTF)
3. Fit GARCH(1,1) on train log-returns → predictive volatility (feature 27)
4. Walk-forward: 24-month train, 3-month test, 3-month shift
5. Train LSTM (early stopping on val loss, best weights reloaded)
6. Backtest: close-to-close execution, vol-targeted position sizing
7. Pooled aggregate metrics across all windows

## Key Design Decisions

- **Close-to-close execution**: trade entry/exit uses close prices, matching the label exactly
- **Purge gap**: `seq_len` samples skipped between train and test to prevent feature overlap
- **Train-only normalization**: feat_min/feat_max computed on train only, applied to val/test
- **Pooled metrics**: all trades from all windows pooled into one metrics_compute call
- **Best weight restoration**: model_load called after train_run to reload best epoch weights

## Config (parameters.txt)

Key parameters: `cells`, `epochs`, `patience`, `seq`, `lr`, `output`, `hold_bars`,
`max_windows`, `wf_train_months`, `wf_test_months`, `wf_shift`, `vol_mode`, `val_ratio`.
