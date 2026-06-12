# Getting started

### Minimal example
Three types of data are required to fit a model:

<!---  Table --->
| Variable     |Description
|--------------|----------|
| `stimuli`    | list/array of signed stimulus intensity values, where the sign codes the stimulus category and the absolute value codes the intensity.      |
| `choices`    | list/array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive stimulus category.         |
| `confidence` | list/array of confidence ratings. Confidence ratings must be normalized to [0; 1]. Discrete confidence ratings must be normalized accordingly (e.g., if confidence ratings are 1-4, subtract 1 and divide by 3).        |

When fitting individual participants, these are 1d lists / arrays with length `ntrials`. Note that 2d Group data are supported as well (-> link).

To quickly demonstrate ReMeta, we load a simple dataset as follows.
```python
import remeta
ds = remeta.load_dataset('default')  # load example dataset
```

Output:
```
..Generative model:
    Type 1 noise distribution: normal
    Type 2 noise type: report
    Type 2 noise distribution: beta_mode
..Generative parameters:
    type1_noise: 0.5
    type1_bias: -0.1
    type2_noise: 0.3
    type2_criteria: [0.25 0.5  0.75]
        [extra] Criterion bias: 0.
        [extra] Criterion-based confidence bias: 0.
..Descriptive statistics:
    No. subjects: 1
    No. samples: 2000
    No. of discrete confidence levels: 4
    Accuracy: 85.2% correct
    d': 2.1
    Choice bias: -3.9%
    Confidence: 0.65
    M-Ratio: 0.95
    AUROC2: 0.77
```

The output provides information how the dataset was generated and some descriptive statistics.

```python
rem = remeta.ReMeta()
rem.fit(ds.stimuli, ds.choices, ds.confidence, n_ratings=4)
```


Output (for fit):
```
+++ Type 1 level +++
  Subject-level estimation (MLE)
    .. finished (0.2 secs).
  Final report
    Parameters estimates (subject-level fit)
        [subject] type1_noise: 0.510 ± 0.018
        [subject] type1_bias: -0.099 ± 0.019
    [subject] Log-likelihood: -717.84 (per sample: -0.3589)
    [subject] Fitting time: 0.17 secs
Type 1 level finished
+++ Type 2 level +++
  Subject-level estimation (MLE)
    .. finished (128.9 secs).
  Final report
    Parameters estimates (subject-level fit)
        [subject] type2_noise: 0.312 ± 0.047
        [subject] type2_criteria: [0.278 ± 0.012, 0.505 ± 0.014, 0.738 ± 0.018]
            [extra] type2_criteria_bias: 0.016 ± 0.010
            [extra] type2_criteria_confidence_bias: -0.016 ± 0.010
    [subject] Log-likelihood: -2938.91 (per sample: -1.469)
    [subject] Fitting time: 53.63 secs
Type 2 level finished (128.9 secs)
```

Since the dataset is based on simulation, we know the true parameters of the underlying generative model (see first output), which are quite close to the fitted parameters.

We can access the fitted parameters by invoking the `summary()` method on the `ReMeta` instance:

```python
# Access fitted parameters
import numpy as np
result = rem.summary()
for k, v in result.params.items():
    print(f'{k}: {np.array2string(np.array(v), precision=3)}')
```

Ouput:
```
type1_noise: 0.51
type1_bias: -0.099
type2_noise: 0.312
type2_criteria: [0.278 0.505 0.738]
```

By default, the model fits parameters for type 1 noise (`type1_noise`) and a type 1 bias (`type1_bias`), as well as metacognitive 'type 2' noise (`type2_noise`) and confidence criteria (`type2_criteria`). Moreover, by default the model assumes that metacognitive noise occurs at the stage of the confidence report (setting `type2_noise_type='noisy_report'`) and that type 2 metacognitive noise can be described by a truncated normal distribution (setting `type2_noise_dist='truncated_norm_mode'`).

All settings can be changed via the `Configuration` object which is optionally passed to the `ReMeta` instance. For example, to change the metacognitive noisy type to "noisy-readout":

```python
cfg = remeta.Configuration()
cfg.type2_noise_type = 'noisy_readout'
rem = remeta.ReMeta(cfg)
...
```
