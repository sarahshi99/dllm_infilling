# Proportional Length Widening Audit

Decision: `needs_expanded_grid_smoke`.

## Overview

- rows: `1033`
- max available probe length: `24`
- baseline average absolute length error: `3.0068`

## Best Policy

| Policy | Promoted | Improved | Worsened | Failed True-Long Improved | Short Promoted | Avg Abs Error Delta |
|---|---:|---:|---:|---:|---:|---:|
| `base0p985_slope0p02_min0p85_minbase12_maxx1p5_noraw` | `5` | `0` | `5` | `0` | `1` | `0.0077` |

## Top Policies

| Policy | Score | Promoted | Improved | Worsened | 17-24 improved | 25+ improved | Short promoted |
|---|---:|---:|---:|---:|---:|---:|---:|
| `base0p985_slope0p02_min0p85_minbase12_maxx1p5_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p85_minbase12_maxx1p5_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p85_minbase12_maxx2_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p85_minbase12_maxx2_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p85_minbase12_maxx3_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p85_minbase12_maxx3_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p88_minbase12_maxx1p5_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p88_minbase12_maxx1p5_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p88_minbase12_maxx2_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p88_minbase12_maxx2_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p88_minbase12_maxx3_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p88_minbase12_maxx3_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p9_minbase12_maxx1p5_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p9_minbase12_maxx1p5_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p9_minbase12_maxx2_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p9_minbase12_maxx2_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p9_minbase12_maxx3_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p9_minbase12_maxx3_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p93_minbase12_maxx1p5_noraw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |
| `base0p985_slope0p02_min0p93_minbase12_maxx1p5_raw` | `-17` | `5` | `0` | `5` | `0` | `0` | `1` |

## Interpretation

This audit is CPU-only and replays stored probe scores. It does not prove a pass-rate gain.
Oracle length and pass/fail are used only for offline accounting, not as policy inputs.
The current stored probe grid is probably too narrow for the user's intended long-length widening idea; a small expanded-grid GPU smoke would be the next meaningful test.
