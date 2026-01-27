Go directly to:
- [**Installation**](https://github.com/coconeuro/remeta/blob/main/INSTALL.md)
- [**Basic Usage**](https://github.com/coconeuro/remeta/blob/main/demo/basic_usage.ipynb)
- [**Common use cases**](https://github.com/coconeuro/remeta/blob/main/demo/common_use_cases.ipynb)
- [**Group estimation and priors**](https://github.com/coconeuro/remeta/blob/main/demo/group_estimation_priors.ipynb)


# ReMeta Toolbox

The ReMeta toolbox allows researchers to estimate latent type 1 and type 2 parameters based on data of cognitive or perceptual decision-making tasks with two response categories. 


### Minimal example
Three types of data are required to fit a model:

<!---  Table --->
| Type       | Variable |Description
|------------|----------|----------|
| Stimuli    | `x_stim`   | list/array of signed stimulus intensity values, where the sign codes the stimulus category and the absolute value codes the intensity. The stimuli should be normalized to [-1; 1], although there is a setting (`normalize_stimuli_by_max`) to auto-normalize stimuli         |
| Choices    | `d_dec`    | list/array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive stimulus category.         |
| Confidence | `c_conf`   | list/array of confidence ratings. Confidence ratings must be normalized to [0; 1]. Discrete confidence ratings must be normalized accordingly (e.g., if confidence ratings are 1-4, subtract 1 and divide by 3).         |

A minimal example would be the following:
```python
# Minimal example
import remeta
ds = remeta.load_dataset('default')  # load example dataset
rem = remeta.ReMeta()
rem.fit(ds.x_stim, ds.d_dec, ds.c_conf)
```
Output (for load_dataset):
```
..Generative model:
    Type 2 noise type: noisy_report
    Type 2 noise distribution: truncated_norm_mode
..Generative parameters:
    type1_noise: 0.5
    type1_bias: -0.1
    type2_noise: 0.3
    type2_criteria: [0.2 0.2 0.2 0.2]
    Type 2 criteria (absolute): [0.2, 0.4, 0.6, 0.8]
    Criterion bias: 0
..Descriptive statistics:
    No. subjects: 1
    No. samples: 2000
    Performance: 86.5% correct
    Choice bias: -3.1%
    Confidence: 0.62
    M-Ratio: 0.58
    AUROC2: 0.69
```
Output (for fit):
```
+++ Type 1 level +++
  Subject-level estimation (MLE)
    .. finished (0.3 secs).
  Final report
    Parameters estimates (subject-level fit)
        [subject] type1_noise: 0.503
        [subject] type1_bias: -0.0821
    [subject] Neg. LL: 683.64
    [subject] Fitting time: 0.25 secs
Type 1 level finished

+++ Type 2 level +++
  Subject-level estimation (MLE)
        Grid search activated (grid size = 2048)
        Grid iteration 1000 / 2048
        Grid iteration 2000 / 2048
            [grid] type2_noise: 0.2641
            [grid] type2_criteria_0: 0.1667
            [grid] type2_criteria_1: 0.2 = gap | criterion = 0.4
            [grid] type2_criteria_2: 0.2 = gap | criterion = 0.6
            [grid] type2_criteria_3: 0.2 = gap | criterion = 0.8
        Grid neg. LL: 3636.9
        Grid runtime: 150.86 secs
    .. finished (199.1 secs).
  Final report
    Parameters estimates (subject-level fit)
        [subject] type2_noise: 0.288
        [subject] type2_criteria_0: 0.189
        [subject] type2_criteria_1: 0.21 = gap | criterion = 0.399
        [subject] type2_criteria_2: 0.206 = gap | criterion = 0.605
        [subject] type2_criteria_3: 0.194 = gap | criterion = 0.798
            [extra] type2_criteria_absolute: [0.189, 0.399, 0.605, 0.798]
            [extra] type2_criteria_bias: -0.00211
    [subject] Neg. LL: 3605.21
    [subject] Fitting time: 198.89 secs
Type 2 level finished
```

Since the dataset is based on simulation, we know the true parameters of the underlying generative model (see first output), which are quite close to the fitted parameters.

We can access the fitted parameters by invoking the `summary()` method on the `ReMeta` instance:

```python
# Access fitted parameters
import numpy as np
result = rem.summary()
for k, v in result.model.params.items():
    print(f'{k}: {np.array2string(np.array(v), precision=3)}')
```

Ouput:
```
type1_noise: 0.503
type1_bias: -0.082
type2_noise: 0.288
type2_criteria: [0.189 0.21  0.206 0.194]
```

By default, the model fits parameters for type 1 noise (`type1_noise`) and a type 1 bias (`type1_bias`), as well as metacognitive 'type 2' noise (`type2_noise`) and 4 confidence criteria (`type2_criteria`). Moreover, by default the model assumes that metacognitive noise occurs at the stage of the confidence report (setting `type2_noise_type='noisy_report'`) and that type 2 metacognitive noise can be described by a truncated normal distribution (setting `type2_noise_dist='truncated_norm_mode'`).

All settings can be changed via the `Configuration` object which is optionally passed to the `ReMeta` instance. For example, to change the metacognitive noisy type to "noisy-readout":

```python
cfg = remeta.Configuration()
cfg.type2_noise_type = 'noisy_readout'
rem = remeta.ReMeta(cfg)
...
```

### Supported parameters

_Type 1 parameters_:
- `type1_noise`: type 1 noise
- `type1_bias`: type 1 bias towards one of the two stimulus categories
- `type1_thresh`: a (sensory) threshold, building on the assumption that a certain minimal stimulus intensity is required to elicit behavior; use only if there are stimulus intensities close to threshold
- `type1_noise_heteroscedastic`: parameter to specify stimulus-dependent type 1 noise (e.g. multiplicative noise)

_Type 2 (metacognitive) parameters:_
- `type2_noise`: metacognitive noise
- `type2_criteria`: confidence criteria
- `type2_evidence_bias_mult`: optional multiplicative metacognitive bias

In addition, each type 1 parameter can be fitted in "duplex mode", such that separate values are fitted depending on the stimulus category.

A more detailed guide to use the toolbox is provided in the following Jupyter notebook: [**Basic Usage**](https://github.com/coconeuro/remeta/blob/main/demo/basic_usage.ipynb)
