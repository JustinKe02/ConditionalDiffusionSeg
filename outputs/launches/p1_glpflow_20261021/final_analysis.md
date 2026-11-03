# GLP-Flow batch completed

| Arm | Best epoch | mIoU % | Mono % | Few % | Thin mean % | FG boundary F1 % | few→mono % | 3408 mIoU % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 100 | 85.26 | 67.17 | 76.64 | 71.91 | 67.86 | 15.47 | 85.26 |
| E1 | 50 | 88.32 | 77.02 | 78.19 | 77.61 | 71.06 | 17.48 | 88.32 |
| E2 | 60 | 86.70 | 74.28 | 74.93 | 74.61 | 55.95 | 18.67 | 86.73 |
| E3 | 50 | 86.96 | 74.68 | 75.53 | 75.10 | 54.07 | 18.44 | 86.99 |

Main screen passed: False

```
{
 "miou_E3_minus_E1": false,
 "thin_mean_E3_minus_E1": false,
 "few_to_mono_E3_below_E1": false,
 "miou_E3_minus_E0": true,
 "miou_E3_minus_E2": false,
 "boundary_E3_minus_E1": false,
 "seed3408_E3_above_E1_and_E2": false
}
```

Single training seed 42; val25 used for selection; no test evaluation. Without-m109 values are descriptive only.
