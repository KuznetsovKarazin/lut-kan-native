# Shape-constrained LUT follow-up

Synthetic confirmation seed 20261108; 144 cases, 4,896 model evaluations; 8 new virtual units × 3 repeats × 2 budgets × 3 scenarios. No physical sensor or MCU timings are contained in this report.

## Equal-payload uniform LUT comparison

L32/int16: 88 bytes for either method. Max is the largest absolute error across all held-out inputs and 24 calibration/noise cases in a row.

| Scenario | Budget | Uniform MAE | Monotone MAE | Uniform max | Monotone max |
|---|---:|---:|---:|---:|---:|
| ntc | 30 | 0.088845 | 0.078804 | 6.506957 | 1.737615 |
| ntc | 75 | 0.058130 | 0.058130 | 0.576794 | 0.576794 |
| ntc_frontend | 30 | 0.096144 | 0.079676 | 8.267284 | 1.855882 |
| ntc_frontend | 75 | 0.058949 | 0.058949 | 0.573180 | 0.573180 |
| ldr | 30 | 0.004350 | 0.004823 | 0.044607 | 0.082980 |
| ldr | 75 | 0.003187 | 0.003187 | 0.021197 | 0.021197 |

NTC errors are in °C; LDR errors in log10(lux). All monotone LUTs preserve the declared direction after float32/int16 encoding. Monotonicity does not bound the physical error, and it can worsen validation-selected mean error in other sizes, notably L64/B75. See the complete table below.

## What to carry forward

- Monotonicity is useful as a shape safeguard at no additional inference payload. It substantially reduces severe sparse-fit outliers, but is not a universal MAE improvement.
- At L32/B30, the paired 95% unit-bootstrap intervals for mean MAE differences include zero for both NTC scenarios (see paired.json). The reduction of the observed largest errors should not be presented as a statistically established mean-accuracy gain.
- Nonuniform knots can improve approximation, especially in the earlier failed cases. They add 4L knot bytes and binary search at inference. Their accuracy cannot be compared as if payload and execution cost were unchanged.
- Strong physics/polynomial/cubic comparators remain necessary. This follow-up does not establish a resource win: real-board timings are the next required observation.
- The firmware fixture is predeclared (new seed, distorted NTC, unit0/repeat0, B75), not the best test case. Its C++ kernels were checked against exported Python references; cross-compilation is distinct from execution on a board.

## Known failures from the original pilot (regression examples only)

| Scenario | Method | L | Payload | MAE | Max |
|---|---|---:|---:|---:|---:|
| ntc | direct_lut | 32 | 88 | 0.236800 | 4.294151 |
| ntc | monotone_lut | 32 | 88 | 0.228156 | 2.365750 |
| ntc | nonuniform_monotone | 32 | 216 | 0.057086 | 0.333619 |
| ldr | direct_lut | 64 | 152 | 0.037487 | 0.752869 |
| ldr | monotone_lut | 64 | 152 | 0.005302 | 0.047913 |
| ldr | nonuniform_monotone | 64 | 408 | 0.004218 | 0.016542 |

These two previously observed failures were kept separate from new confirmation data. They cannot be treated as independent evidence for a newly chosen method.

## Complete summary

| Scenario | B | Method | L | Encoding | Mean payload B | MAE | Worst max |
|---|---:|---|---:|---|---:|---:|---:|
| ldr | 30 | cubic_spline | 8 | float32 | 48.0 | 0.005229 | 0.049993 |
| ldr | 30 | cubic_spline | 16 | float32 | 80.0 | 0.002772 | 0.022338 |
| ldr | 30 | cubic_spline | 32 | float32 | 144.0 | 0.003962 | 0.039629 |
| ldr | 30 | cubic_spline | 64 | float32 | 272.0 | 0.003971 | 0.105545 |
| ldr | 30 | direct_lut | 8 | float32 | 48.0 | 0.014607 | 0.117752 |
| ldr | 30 | direct_lut | 8 | int16 | 40.0 | 0.014607 | 0.117748 |
| ldr | 30 | direct_lut | 16 | float32 | 80.0 | 0.004757 | 0.035360 |
| ldr | 30 | direct_lut | 16 | int16 | 56.0 | 0.004757 | 0.035357 |
| ldr | 30 | direct_lut | 32 | float32 | 144.0 | 0.004350 | 0.044614 |
| ldr | 30 | direct_lut | 32 | int16 | 88.0 | 0.004350 | 0.044607 |
| ldr | 30 | direct_lut | 64 | float32 | 272.0 | 0.003697 | 0.039852 |
| ldr | 30 | direct_lut | 64 | int16 | 152.0 | 0.003697 | 0.039861 |
| ldr | 30 | kan_edge | 8 | float32 | 52.0 | 0.005321 | 0.049757 |
| ldr | 30 | kan_edge | 16 | float32 | 84.0 | 0.002833 | 0.022361 |
| ldr | 30 | kan_edge | 32 | float32 | 148.0 | 0.004269 | 0.043224 |
| ldr | 30 | kan_edge | 64 | float32 | 276.0 | 0.003967 | 0.105547 |
| ldr | 30 | monotone_lut | 8 | float32 | 48.0 | 0.014607 | 0.117752 |
| ldr | 30 | monotone_lut | 8 | int16 | 40.0 | 0.014607 | 0.117748 |
| ldr | 30 | monotone_lut | 16 | float32 | 80.0 | 0.004757 | 0.035360 |
| ldr | 30 | monotone_lut | 16 | int16 | 56.0 | 0.004757 | 0.035357 |
| ldr | 30 | monotone_lut | 32 | float32 | 144.0 | 0.004823 | 0.082978 |
| ldr | 30 | monotone_lut | 32 | int16 | 88.0 | 0.004823 | 0.082980 |
| ldr | 30 | monotone_lut | 64 | float32 | 272.0 | 0.003771 | 0.039852 |
| ldr | 30 | monotone_lut | 64 | int16 | 152.0 | 0.003771 | 0.039861 |
| ldr | 30 | nonuniform_monotone | 8 | float32 | 80.0 | 0.005737 | 0.028967 |
| ldr | 30 | nonuniform_monotone | 8 | int16 | 72.0 | 0.005737 | 0.028966 |
| ldr | 30 | nonuniform_monotone | 16 | float32 | 144.0 | 0.003452 | 0.040410 |
| ldr | 30 | nonuniform_monotone | 16 | int16 | 120.0 | 0.003453 | 0.040406 |
| ldr | 30 | nonuniform_monotone | 32 | float32 | 272.0 | 0.003334 | 0.031673 |
| ldr | 30 | nonuniform_monotone | 32 | int16 | 216.0 | 0.003334 | 0.031694 |
| ldr | 30 | nonuniform_monotone | 64 | float32 | 528.0 | 0.004246 | 0.041436 |
| ldr | 30 | nonuniform_monotone | 64 | int16 | 408.0 | 0.004247 | 0.041441 |
| ldr | 30 | physics | 0 | float32 | 28.0 | 0.001474 | 0.015470 |
| ldr | 30 | polynomial | 0 | float32 | 54.7 | 0.003053 | 0.023320 |
| ldr | 75 | cubic_spline | 8 | float32 | 48.0 | 0.004684 | 0.059592 |
| ldr | 75 | cubic_spline | 16 | float32 | 80.0 | 0.002083 | 0.017950 |
| ldr | 75 | cubic_spline | 32 | float32 | 144.0 | 0.002649 | 0.046413 |
| ldr | 75 | cubic_spline | 64 | float32 | 272.0 | 0.002800 | 0.028315 |
| ldr | 75 | direct_lut | 8 | float32 | 48.0 | 0.014170 | 0.125547 |
| ldr | 75 | direct_lut | 8 | int16 | 40.0 | 0.014170 | 0.125547 |
| ldr | 75 | direct_lut | 16 | float32 | 80.0 | 0.004020 | 0.041168 |
| ldr | 75 | direct_lut | 16 | int16 | 56.0 | 0.004021 | 0.041173 |
| ldr | 75 | direct_lut | 32 | float32 | 144.0 | 0.003187 | 0.021190 |
| ldr | 75 | direct_lut | 32 | int16 | 88.0 | 0.003187 | 0.021197 |
| ldr | 75 | direct_lut | 64 | float32 | 272.0 | 0.002853 | 0.027623 |
| ldr | 75 | direct_lut | 64 | int16 | 152.0 | 0.002853 | 0.027618 |
| ldr | 75 | kan_edge | 8 | float32 | 52.0 | 0.004758 | 0.059342 |
| ldr | 75 | kan_edge | 16 | float32 | 84.0 | 0.002113 | 0.017821 |
| ldr | 75 | kan_edge | 32 | float32 | 148.0 | 0.002584 | 0.042621 |
| ldr | 75 | kan_edge | 64 | float32 | 276.0 | 0.002786 | 0.027697 |
| ldr | 75 | monotone_lut | 8 | float32 | 48.0 | 0.014170 | 0.125547 |
| ldr | 75 | monotone_lut | 8 | int16 | 40.0 | 0.014170 | 0.125547 |
| ldr | 75 | monotone_lut | 16 | float32 | 80.0 | 0.004020 | 0.041168 |
| ldr | 75 | monotone_lut | 16 | int16 | 56.0 | 0.004021 | 0.041173 |
| ldr | 75 | monotone_lut | 32 | float32 | 144.0 | 0.003187 | 0.021190 |
| ldr | 75 | monotone_lut | 32 | int16 | 88.0 | 0.003187 | 0.021197 |
| ldr | 75 | monotone_lut | 64 | float32 | 272.0 | 0.003035 | 0.031427 |
| ldr | 75 | monotone_lut | 64 | int16 | 152.0 | 0.003035 | 0.031414 |
| ldr | 75 | nonuniform_monotone | 8 | float32 | 80.0 | 0.005627 | 0.044839 |
| ldr | 75 | nonuniform_monotone | 8 | int16 | 72.0 | 0.005626 | 0.044846 |
| ldr | 75 | nonuniform_monotone | 16 | float32 | 144.0 | 0.002604 | 0.061345 |
| ldr | 75 | nonuniform_monotone | 16 | int16 | 120.0 | 0.002604 | 0.061338 |
| ldr | 75 | nonuniform_monotone | 32 | float32 | 272.0 | 0.002088 | 0.021054 |
| ldr | 75 | nonuniform_monotone | 32 | int16 | 216.0 | 0.002088 | 0.021054 |
| ldr | 75 | nonuniform_monotone | 64 | float32 | 528.0 | 0.002561 | 0.032630 |
| ldr | 75 | nonuniform_monotone | 64 | int16 | 408.0 | 0.002561 | 0.032645 |
| ldr | 75 | physics | 0 | float32 | 28.0 | 0.001229 | 0.013859 |
| ldr | 75 | polynomial | 0 | float32 | 55.3 | 0.002402 | 0.021758 |
| ntc | 30 | cubic_spline | 8 | float32 | 48.0 | 0.092310 | 0.705402 |
| ntc | 30 | cubic_spline | 16 | float32 | 80.0 | 0.055148 | 0.517604 |
| ntc | 30 | cubic_spline | 32 | float32 | 144.0 | 0.056119 | 0.544865 |
| ntc | 30 | cubic_spline | 64 | float32 | 272.0 | 0.059781 | 0.692433 |
| ntc | 30 | direct_lut | 8 | float32 | 48.0 | 0.413413 | 2.990568 |
| ntc | 30 | direct_lut | 8 | int16 | 40.0 | 0.413416 | 2.990687 |
| ntc | 30 | direct_lut | 16 | float32 | 80.0 | 0.103423 | 0.884887 |
| ntc | 30 | direct_lut | 16 | int16 | 56.0 | 0.103424 | 0.885420 |
| ntc | 30 | direct_lut | 32 | float32 | 144.0 | 0.088844 | 6.507376 |
| ntc | 30 | direct_lut | 32 | int16 | 88.0 | 0.088845 | 6.506957 |
| ntc | 30 | direct_lut | 64 | float32 | 272.0 | 0.059380 | 0.661991 |
| ntc | 30 | direct_lut | 64 | int16 | 152.0 | 0.059371 | 0.661296 |
| ntc | 30 | kan_edge | 8 | float32 | 52.0 | 0.091523 | 0.673703 |
| ntc | 30 | kan_edge | 16 | float32 | 84.0 | 0.055305 | 0.517631 |
| ntc | 30 | kan_edge | 32 | float32 | 148.0 | 0.063207 | 0.807082 |
| ntc | 30 | kan_edge | 64 | float32 | 276.0 | 0.059915 | 0.683767 |
| ntc | 30 | monotone_lut | 8 | float32 | 48.0 | 0.413413 | 2.990568 |
| ntc | 30 | monotone_lut | 8 | int16 | 40.0 | 0.413416 | 2.990687 |
| ntc | 30 | monotone_lut | 16 | float32 | 80.0 | 0.103423 | 0.884908 |
| ntc | 30 | monotone_lut | 16 | int16 | 56.0 | 0.103424 | 0.885381 |
| ntc | 30 | monotone_lut | 32 | float32 | 144.0 | 0.078805 | 1.737836 |
| ntc | 30 | monotone_lut | 32 | int16 | 88.0 | 0.078804 | 1.737615 |
| ntc | 30 | monotone_lut | 64 | float32 | 272.0 | 0.059533 | 0.661991 |
| ntc | 30 | monotone_lut | 64 | int16 | 152.0 | 0.059524 | 0.661296 |
| ntc | 30 | nonuniform_monotone | 8 | float32 | 80.0 | 0.152801 | 0.892417 |
| ntc | 30 | nonuniform_monotone | 8 | int16 | 72.0 | 0.152805 | 0.892186 |
| ntc | 30 | nonuniform_monotone | 16 | float32 | 144.0 | 0.067443 | 0.807225 |
| ntc | 30 | nonuniform_monotone | 16 | int16 | 120.0 | 0.067438 | 0.806549 |
| ntc | 30 | nonuniform_monotone | 32 | float32 | 272.0 | 0.062238 | 0.773854 |
| ntc | 30 | nonuniform_monotone | 32 | int16 | 216.0 | 0.062235 | 0.773615 |
| ntc | 30 | nonuniform_monotone | 64 | float32 | 528.0 | 0.080442 | 1.505680 |
| ntc | 30 | nonuniform_monotone | 64 | int16 | 408.0 | 0.080447 | 1.505686 |
| ntc | 30 | physics | 0 | float32 | 32.0 | 0.044092 | 0.415871 |
| ntc | 30 | polynomial | 0 | float32 | 55.0 | 0.055233 | 0.587550 |
| ntc | 75 | cubic_spline | 8 | float32 | 48.0 | 0.088532 | 1.024435 |
| ntc | 75 | cubic_spline | 16 | float32 | 80.0 | 0.046256 | 0.423164 |
| ntc | 75 | cubic_spline | 32 | float32 | 144.0 | 0.046889 | 0.462528 |
| ntc | 75 | cubic_spline | 64 | float32 | 272.0 | 0.051047 | 0.629007 |
| ntc | 75 | direct_lut | 8 | float32 | 48.0 | 0.408562 | 3.201065 |
| ntc | 75 | direct_lut | 8 | int16 | 40.0 | 0.408552 | 3.201189 |
| ntc | 75 | direct_lut | 16 | float32 | 80.0 | 0.098740 | 0.852515 |
| ntc | 75 | direct_lut | 16 | int16 | 56.0 | 0.098743 | 0.852200 |
| ntc | 75 | direct_lut | 32 | float32 | 144.0 | 0.058125 | 0.576271 |
| ntc | 75 | direct_lut | 32 | int16 | 88.0 | 0.058130 | 0.576794 |
| ntc | 75 | direct_lut | 64 | float32 | 272.0 | 0.052457 | 0.608297 |
| ntc | 75 | direct_lut | 64 | int16 | 152.0 | 0.052454 | 0.608356 |
| ntc | 75 | kan_edge | 8 | float32 | 52.0 | 0.085839 | 0.721724 |
| ntc | 75 | kan_edge | 16 | float32 | 84.0 | 0.046620 | 0.423833 |
| ntc | 75 | kan_edge | 32 | float32 | 148.0 | 0.047445 | 0.462565 |
| ntc | 75 | kan_edge | 64 | float32 | 276.0 | 0.051081 | 0.614968 |
| ntc | 75 | monotone_lut | 8 | float32 | 48.0 | 0.408562 | 3.201065 |
| ntc | 75 | monotone_lut | 8 | int16 | 40.0 | 0.408552 | 3.201189 |
| ntc | 75 | monotone_lut | 16 | float32 | 80.0 | 0.098740 | 0.852515 |
| ntc | 75 | monotone_lut | 16 | int16 | 56.0 | 0.098743 | 0.852200 |
| ntc | 75 | monotone_lut | 32 | float32 | 144.0 | 0.058125 | 0.576271 |
| ntc | 75 | monotone_lut | 32 | int16 | 88.0 | 0.058130 | 0.576794 |
| ntc | 75 | monotone_lut | 64 | float32 | 272.0 | 0.060939 | 0.943019 |
| ntc | 75 | monotone_lut | 64 | int16 | 152.0 | 0.060944 | 0.943612 |
| ntc | 75 | nonuniform_monotone | 8 | float32 | 80.0 | 0.149806 | 0.968449 |
| ntc | 75 | nonuniform_monotone | 8 | int16 | 72.0 | 0.149798 | 0.968288 |
| ntc | 75 | nonuniform_monotone | 16 | float32 | 144.0 | 0.054901 | 0.515604 |
| ntc | 75 | nonuniform_monotone | 16 | int16 | 120.0 | 0.054899 | 0.515559 |
| ntc | 75 | nonuniform_monotone | 32 | float32 | 272.0 | 0.049313 | 0.535171 |
| ntc | 75 | nonuniform_monotone | 32 | int16 | 216.0 | 0.049321 | 0.535193 |
| ntc | 75 | nonuniform_monotone | 64 | float32 | 528.0 | 0.051014 | 0.765316 |
| ntc | 75 | nonuniform_monotone | 64 | int16 | 408.0 | 0.051021 | 0.765122 |
| ntc | 75 | physics | 0 | float32 | 32.0 | 0.041619 | 0.422937 |
| ntc | 75 | polynomial | 0 | float32 | 55.7 | 0.047989 | 0.523620 |
| ntc_frontend | 30 | cubic_spline | 8 | float32 | 48.0 | 0.099707 | 0.708989 |
| ntc_frontend | 30 | cubic_spline | 16 | float32 | 80.0 | 0.054687 | 0.521237 |
| ntc_frontend | 30 | cubic_spline | 32 | float32 | 144.0 | 0.061573 | 0.951070 |
| ntc_frontend | 30 | cubic_spline | 64 | float32 | 272.0 | 0.057156 | 0.627866 |
| ntc_frontend | 30 | direct_lut | 8 | float32 | 48.0 | 0.437675 | 3.150100 |
| ntc_frontend | 30 | direct_lut | 8 | int16 | 40.0 | 0.437666 | 3.150196 |
| ntc_frontend | 30 | direct_lut | 16 | float32 | 80.0 | 0.104755 | 0.970149 |
| ntc_frontend | 30 | direct_lut | 16 | int16 | 56.0 | 0.104767 | 0.969495 |
| ntc_frontend | 30 | direct_lut | 32 | float32 | 144.0 | 0.096152 | 8.267662 |
| ntc_frontend | 30 | direct_lut | 32 | int16 | 88.0 | 0.096144 | 8.267284 |
| ntc_frontend | 30 | direct_lut | 64 | float32 | 272.0 | 0.057782 | 0.657460 |
| ntc_frontend | 30 | direct_lut | 64 | int16 | 152.0 | 0.057791 | 0.658284 |
| ntc_frontend | 30 | kan_edge | 8 | float32 | 52.0 | 0.098223 | 0.709036 |
| ntc_frontend | 30 | kan_edge | 16 | float32 | 84.0 | 0.055255 | 0.521111 |
| ntc_frontend | 30 | kan_edge | 32 | float32 | 148.0 | 0.069734 | 2.824777 |
| ntc_frontend | 30 | kan_edge | 64 | float32 | 276.0 | 0.057400 | 0.614891 |
| ntc_frontend | 30 | monotone_lut | 8 | float32 | 48.0 | 0.437675 | 3.150100 |
| ntc_frontend | 30 | monotone_lut | 8 | int16 | 40.0 | 0.437666 | 3.150196 |
| ntc_frontend | 30 | monotone_lut | 16 | float32 | 80.0 | 0.104755 | 0.970149 |
| ntc_frontend | 30 | monotone_lut | 16 | int16 | 56.0 | 0.104767 | 0.969495 |
| ntc_frontend | 30 | monotone_lut | 32 | float32 | 144.0 | 0.079675 | 1.855961 |
| ntc_frontend | 30 | monotone_lut | 32 | int16 | 88.0 | 0.079676 | 1.855882 |
| ntc_frontend | 30 | monotone_lut | 64 | float32 | 272.0 | 0.057941 | 0.657460 |
| ntc_frontend | 30 | monotone_lut | 64 | int16 | 152.0 | 0.057950 | 0.658284 |
| ntc_frontend | 30 | nonuniform_monotone | 8 | float32 | 80.0 | 0.185077 | 0.863431 |
| ntc_frontend | 30 | nonuniform_monotone | 8 | int16 | 72.0 | 0.185090 | 0.863779 |
| ntc_frontend | 30 | nonuniform_monotone | 16 | float32 | 144.0 | 0.070468 | 0.734551 |
| ntc_frontend | 30 | nonuniform_monotone | 16 | int16 | 120.0 | 0.070453 | 0.734446 |
| ntc_frontend | 30 | nonuniform_monotone | 32 | float32 | 272.0 | 0.062954 | 0.791401 |
| ntc_frontend | 30 | nonuniform_monotone | 32 | int16 | 216.0 | 0.062945 | 0.791586 |
| ntc_frontend | 30 | nonuniform_monotone | 64 | float32 | 528.0 | 0.080729 | 1.417060 |
| ntc_frontend | 30 | nonuniform_monotone | 64 | int16 | 408.0 | 0.080726 | 1.417030 |
| ntc_frontend | 30 | physics | 0 | float32 | 32.0 | 0.299054 | 1.161962 |
| ntc_frontend | 30 | polynomial | 0 | float32 | 55.7 | 0.054126 | 0.463490 |
| ntc_frontend | 75 | cubic_spline | 8 | float32 | 48.0 | 0.094163 | 0.777677 |
| ntc_frontend | 75 | cubic_spline | 16 | float32 | 80.0 | 0.045560 | 0.456420 |
| ntc_frontend | 75 | cubic_spline | 32 | float32 | 144.0 | 0.049472 | 0.500220 |
| ntc_frontend | 75 | cubic_spline | 64 | float32 | 272.0 | 0.051679 | 0.584895 |
| ntc_frontend | 75 | direct_lut | 8 | float32 | 48.0 | 0.433637 | 3.552193 |
| ntc_frontend | 75 | direct_lut | 8 | int16 | 40.0 | 0.433642 | 3.552234 |
| ntc_frontend | 75 | direct_lut | 16 | float32 | 80.0 | 0.101533 | 0.998196 |
| ntc_frontend | 75 | direct_lut | 16 | int16 | 56.0 | 0.101537 | 0.998206 |
| ntc_frontend | 75 | direct_lut | 32 | float32 | 144.0 | 0.058960 | 0.573147 |
| ntc_frontend | 75 | direct_lut | 32 | int16 | 88.0 | 0.058949 | 0.573180 |
| ntc_frontend | 75 | direct_lut | 64 | float32 | 272.0 | 0.052199 | 0.567100 |
| ntc_frontend | 75 | direct_lut | 64 | int16 | 152.0 | 0.052205 | 0.567235 |
| ntc_frontend | 75 | kan_edge | 8 | float32 | 52.0 | 0.094477 | 0.777734 |
| ntc_frontend | 75 | kan_edge | 16 | float32 | 84.0 | 0.045467 | 0.456396 |
| ntc_frontend | 75 | kan_edge | 32 | float32 | 148.0 | 0.047669 | 0.456206 |
| ntc_frontend | 75 | kan_edge | 64 | float32 | 276.0 | 0.051392 | 0.571291 |
| ntc_frontend | 75 | monotone_lut | 8 | float32 | 48.0 | 0.433637 | 3.552193 |
| ntc_frontend | 75 | monotone_lut | 8 | int16 | 40.0 | 0.433642 | 3.552234 |
| ntc_frontend | 75 | monotone_lut | 16 | float32 | 80.0 | 0.101533 | 0.998196 |
| ntc_frontend | 75 | monotone_lut | 16 | int16 | 56.0 | 0.101537 | 0.998206 |
| ntc_frontend | 75 | monotone_lut | 32 | float32 | 144.0 | 0.058960 | 0.573147 |
| ntc_frontend | 75 | monotone_lut | 32 | int16 | 88.0 | 0.058949 | 0.573180 |
| ntc_frontend | 75 | monotone_lut | 64 | float32 | 272.0 | 0.061590 | 1.093594 |
| ntc_frontend | 75 | monotone_lut | 64 | int16 | 152.0 | 0.061588 | 1.093794 |
| ntc_frontend | 75 | nonuniform_monotone | 8 | float32 | 80.0 | 0.174559 | 0.900832 |
| ntc_frontend | 75 | nonuniform_monotone | 8 | int16 | 72.0 | 0.174561 | 0.900820 |
| ntc_frontend | 75 | nonuniform_monotone | 16 | float32 | 144.0 | 0.058041 | 0.522328 |
| ntc_frontend | 75 | nonuniform_monotone | 16 | int16 | 120.0 | 0.058046 | 0.521868 |
| ntc_frontend | 75 | nonuniform_monotone | 32 | float32 | 272.0 | 0.052024 | 0.505230 |
| ntc_frontend | 75 | nonuniform_monotone | 32 | int16 | 216.0 | 0.052023 | 0.506015 |
| ntc_frontend | 75 | nonuniform_monotone | 64 | float32 | 528.0 | 0.046568 | 0.549711 |
| ntc_frontend | 75 | nonuniform_monotone | 64 | int16 | 408.0 | 0.046572 | 0.550362 |
| ntc_frontend | 75 | physics | 0 | float32 | 32.0 | 0.298402 | 1.190970 |
| ntc_frontend | 75 | polynomial | 0 | float32 | 56.0 | 0.048579 | 0.429545 |
