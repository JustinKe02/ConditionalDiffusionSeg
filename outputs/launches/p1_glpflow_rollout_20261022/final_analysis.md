# Sampler-aligned E3R completed

| Arm | Best epoch | mIoU % | Mono % | Few % | Thin mean % | FG boundary F1 % | few→mono % |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 50 | 88.32 | 77.02 | 78.19 | 77.61 | 71.06 | 17.48 |
| E3 | 50 | 86.96 | 74.68 | 75.53 | 75.10 | 54.07 | 18.44 |
| E3R | 60 | 88.56 | 77.21 | 78.81 | 78.01 | 74.09 | 17.64 |

E3R 3408 mIoU: 88.56%
Screen passed: False

```
{
 "miou_E3R_minus_E1": false,
 "thin_mean_E3R_minus_E1": false,
 "boundary_E3R_minus_E1": true,
 "few_to_mono_E3R_below_E1": false,
 "miou_E3R_minus_E3": true,
 "seed3408_E3R_above_E1": true
}
```

Single training seed 42; val25 selection; E1/E3 reused from p1_glpflow_20261021; no test.
