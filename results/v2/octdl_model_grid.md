
### all classes (AMD/DME/ERM/NORMAL/RAO/RVO/VID)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease (111) | Logistic regression | 0.726 ± 0.002 | 0.661 [0.603, 0.722] | 0.579 |
| v2 all (243) | Logistic regression | 0.740 ± 0.005 | 0.652 [0.599, 0.715] | 0.592 |
| v2 texture (band x sector) (220) | Logistic regression | 0.716 ± 0.004 | 0.632 [0.573, 0.691] | 0.560 |
| v2 texture, no entropy (200) | Logistic regression | 0.696 ± 0.000 | 0.623 [0.564, 0.684] | 0.548 |
| v2 band-summary + disease (111) | RF (paper config) | 0.755 ± 0.004 | 0.609 [0.552, 0.668] | 0.608 |
| v2 band-summary + disease (111) | Gradient boosting | 0.816 ± 0.000 | 0.608 [0.548, 0.662] | 0.662 |
| v2 band-summary + disease (111) | Two-stage (GB) | 0.810 ± 0.000 | 0.602 [0.543, 0.658] | 0.650 |
| v2 all (243) | RF (paper config) | 0.772 ± 0.003 | 0.601 [0.532, 0.646] | 0.621 |
| v2 all (243) | Gradient boosting | 0.812 ± 0.003 | 0.592 [0.527, 0.638] | 0.652 |
| v2 all (243) | Two-stage (GB) | 0.810 ± 0.001 | 0.591 [0.533, 0.646] | 0.654 |
| v2 texture (band x sector) (220) | RF (paper config) | 0.761 ± 0.001 | 0.575 [0.506, 0.619] | 0.595 |
| v2 texture, no entropy (200) | Gradient boosting | 0.800 ± 0.004 | 0.572 [0.496, 0.615] | 0.630 |
| v2 texture (band x sector) (220) | Gradient boosting | 0.806 ± 0.002 | 0.571 [0.505, 0.613] | 0.629 |
| v2 texture, no entropy (200) | RF (paper config) | 0.757 ± 0.004 | 0.568 [0.513, 0.623] | 0.587 |
| v2 texture (band x sector) (220) | Two-stage (GB) | 0.801 ± 0.002 | 0.568 [0.502, 0.610] | 0.630 |
| v2 texture, no entropy (200) | Two-stage (GB) | 0.797 ± 0.004 | 0.564 [0.489, 0.603] | 0.620 |
| v2 disease-specific (23) | RF (paper config) | 0.580 ± 0.009 | 0.536 [0.466, 0.576] | 0.482 |
| v2 disease-specific (23) | Logistic regression | 0.409 ± 0.000 | 0.508 [0.467, 0.541] | 0.343 |
| paper + entropy (450) | Logistic regression | 0.541 ± 0.000 | 0.485 [0.420, 0.523] | 0.412 |
| paper + entropy (450) | RF (paper config) | 0.617 ± 0.006 | 0.477 [0.408, 0.525] | 0.482 |
| v2 disease-specific (23) | Gradient boosting | 0.675 ± 0.005 | 0.476 [0.405, 0.518] | 0.508 |
| v2 disease-specific (23) | Two-stage (GB) | 0.669 ± 0.004 | 0.474 [0.402, 0.516] | 0.504 |
| paper (300 grid H+S) (300) | RF (paper config) | 0.600 ± 0.009 | 0.464 [0.388, 0.491] | 0.469 |
| paper (300 grid H+S) (300) | Logistic regression | 0.472 ± 0.002 | 0.438 [0.378, 0.467] | 0.358 |
| paper + entropy (450) | Two-stage (GB) | 0.716 ± 0.004 | 0.419 [0.363, 0.453] | 0.487 |
| paper row-summary (40) (40) | RF (paper config) | 0.551 ± 0.001 | 0.418 [0.377, 0.460] | 0.399 |
| paper + entropy (450) | Gradient boosting | 0.721 ± 0.009 | 0.416 [0.364, 0.435] | 0.476 |
| paper (300 grid H+S) (300) | Gradient boosting | 0.705 ± 0.003 | 0.405 [0.362, 0.437] | 0.469 |
| paper row-summary (40) (40) | Logistic regression | 0.285 ± 0.002 | 0.403 [0.336, 0.441] | 0.254 |
| paper (300 grid H+S) (300) | Two-stage (GB) | 0.696 ± 0.001 | 0.389 [0.340, 0.411] | 0.456 |
| paper row-summary (40) (40) | Two-stage (GB) | 0.653 ± 0.004 | 0.349 [0.306, 0.379] | 0.398 |
| paper row-summary (40) (40) | Gradient boosting | 0.652 ± 0.002 | 0.342 [0.298, 0.368] | 0.387 |

### disease-only (AMD/DME/ERM/RAO/RVO/VID)

| Features | Model | Accuracy | Balanced acc [95% CI] | Macro-F1 |
|---|---|---|---|---|
| v2 band-summary + disease (111) | Logistic regression | 0.722 ± 0.000 | 0.648 [0.579, 0.706] | 0.551 |
| v2 all (243) | Logistic regression | 0.752 ± 0.006 | 0.642 [0.583, 0.717] | 0.575 |
| v2 texture (band x sector) (220) | Logistic regression | 0.731 ± 0.010 | 0.631 [0.576, 0.705] | 0.550 |
| v2 texture, no entropy (200) | Logistic regression | 0.707 ± 0.014 | 0.617 [0.558, 0.692] | 0.530 |
| v2 band-summary + disease (111) | Gradient boosting | 0.839 ± 0.000 | 0.595 [0.523, 0.653] | 0.659 |
| v2 band-summary + disease (111) | RF (paper config) | 0.775 ± 0.007 | 0.582 [0.517, 0.648] | 0.589 |
| v2 all (243) | RF (paper config) | 0.792 ± 0.001 | 0.580 [0.512, 0.644] | 0.603 |
| v2 texture (band x sector) (220) | RF (paper config) | 0.790 ± 0.006 | 0.558 [0.491, 0.620] | 0.583 |
| v2 all (243) | Gradient boosting | 0.827 ± 0.005 | 0.554 [0.490, 0.620] | 0.628 |
| v2 texture, no entropy (200) | RF (paper config) | 0.784 ± 0.003 | 0.550 [0.492, 0.623] | 0.580 |
| v2 texture (band x sector) (220) | Gradient boosting | 0.825 ± 0.002 | 0.540 [0.472, 0.598] | 0.610 |
| v2 texture, no entropy (200) | Gradient boosting | 0.818 ± 0.005 | 0.530 [0.468, 0.600] | 0.597 |
| v2 disease-specific (23) | RF (paper config) | 0.649 ± 0.001 | 0.526 [0.462, 0.585] | 0.479 |
| v2 disease-specific (23) | Logistic regression | 0.415 ± 0.011 | 0.516 [0.449, 0.544] | 0.328 |
| v2 disease-specific (23) | Gradient boosting | 0.733 ± 0.005 | 0.468 [0.412, 0.532] | 0.506 |
| paper + entropy (450) | RF (paper config) | 0.681 ± 0.002 | 0.468 [0.415, 0.534] | 0.480 |
| paper + entropy (450) | Logistic regression | 0.548 ± 0.001 | 0.456 [0.406, 0.514] | 0.382 |
| paper (300 grid H+S) (300) | RF (paper config) | 0.673 ± 0.003 | 0.445 [0.383, 0.518] | 0.464 |
| paper row-summary (40) (40) | Logistic regression | 0.296 ± 0.001 | 0.425 [0.353, 0.474] | 0.246 |
| paper (300 grid H+S) (300) | Logistic regression | 0.471 ± 0.008 | 0.420 [0.368, 0.470] | 0.328 |
| paper row-summary (40) (40) | RF (paper config) | 0.628 ± 0.001 | 0.413 [0.368, 0.467] | 0.405 |
| paper + entropy (450) | Gradient boosting | 0.768 ± 0.002 | 0.386 [0.345, 0.438] | 0.461 |
| paper (300 grid H+S) (300) | Gradient boosting | 0.758 ± 0.003 | 0.367 [0.329, 0.415] | 0.441 |
| paper row-summary (40) (40) | Gradient boosting | 0.725 ± 0.006 | 0.331 [0.297, 0.381] | 0.384 |
