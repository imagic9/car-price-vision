# Logo-mask ablation summary

## test (n_used=1094, n_front_view=1094)

| condition | n | mae_years | mape (%) | r2_price_log |
|---|---|---|---|---|
| none | 1094 | 1.044 | 21.14 | 0.8888 |
| badge | 1094 | 1.338 | 29.62 | 0.8069 |
| control | 1094 | 1.064 | 21.92 | 0.8874 |

| shift vs none | mean \|Δyear\| | median \|Δyear\| | mean \|Δlog_price\| | median \|Δlog_price\| | frac >10% | frac >25% |
|---|---|---|---|---|---|---|
| badge | 0.900 | 0.681 | 0.2006 | 0.1637 | 0.676 | 0.314 |
| control | 0.172 | 0.135 | 0.0387 | 0.0282 | 0.068 | 0.005 |

| ablation delta | mae_years | mape (%) | r2_price_log |
|---|---|---|---|
| badge_minus_none | +0.294 | +8.49 | -0.0819 |
| control_minus_none | +0.021 | +0.78 | -0.0014 |
| badge_vs_control | +0.274 | +7.71 | -0.0805 |

## holdout (n_used=1284, n_front_view=1284)

| condition | n | mae_years | mape (%) | r2_price_log |
|---|---|---|---|---|
| none | 1284 | 1.567 | 36.60 | 0.7210 |
| badge | 1284 | 1.720 | 39.38 | 0.6772 |
| control | 1284 | 1.586 | 37.56 | 0.7155 |

| shift vs none | mean \|Δyear\| | median \|Δyear\| | mean \|Δlog_price\| | median \|Δlog_price\| | frac >10% | frac >25% |
|---|---|---|---|---|---|---|
| badge | 0.741 | 0.558 | 0.1707 | 0.1285 | 0.582 | 0.231 |
| control | 0.178 | 0.140 | 0.0406 | 0.0312 | 0.066 | 0.003 |

| ablation delta | mae_years | mape (%) | r2_price_log |
|---|---|---|---|
| badge_minus_none | +0.153 | +2.78 | -0.0437 |
| control_minus_none | +0.019 | +0.96 | -0.0055 |
| badge_vs_control | +0.134 | +1.82 | -0.0383 |
