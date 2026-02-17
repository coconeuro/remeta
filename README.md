# ReMeta Toolbox

The ReMeta ("Reverse engineering of Metacognition") toolbox allows researchers to estimate latent type 1 and type 2 parameters based on data of cognitive or perceptual decision-making tasks with two response categories. 

**Guide:** https://re-meta.github.io/

**Original paper:** Guggenmos, M. (2022). Reverse engineering of metacognition. eLife, 11. https://doi.org/10.7554/elife.75420

**Citation for Toolbox:** Guggenmos, M. (<year-of-release>). ReMeta toolbox (Version X.Y.Z) [Computer software]. GitHub. https://github.com/coconeuro/remeta


### Installation

Remeta is a Python toolbox requires a working Python installation. It should run with Python >=3.10.

Install the latest release with `pip`:
```
pip install remeta
```

Or the most recent code base via GitHub:
```
pip install git+https://github.com/m-guggenmos/remeta.git
```
(this command requires an installed Git, e.g. [gitforwindows](https://gitforwindows.org/))


### Minimal example
Three types of data are required to fit a model:

<!---  Table --->
| Type       | Variable     | Description                                                                                                                                                                                                                                                                                
|------------|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stimuli    | `stimuli`    | list/array of signed stimulus intensity values, where the sign codes the stimulus category and the absolute value codes the intensity. The stimuli should be roughly in the range [-1; 1], although there is a setting (`normalize_stimuli_by_max`) to auto-normalize stimuli |
| Choices    | `choices`    | list/array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive stimulus category.                                                                                                                                                         |
| Confidence | `confidence` | list/array of confidence ratings. Confidence ratings must be normalized to [0; 1]. Discrete confidence ratings must be normalized accordingly (e.g., if confidence ratings are 1-4, subtract 1 and divide by 3).                                                                           |

A minimal example would be the following:
```python
# Minimal example
import remeta
ds = remeta.load_dataset('default')  # load example dataset
rem = remeta.ReMeta()
rem.fit(ds.stimuli, ds.choices, ds.confidence)
```
Output (for load_dataset):
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
        [extra] Criterion bias: 0.0000
        [extra] Criterion-based confidence bias: 0.0000
..Descriptive statistics:
    No. subjects: 1
    No. samples: 2000
    Accuracy: 85.2% correct
    d': 2.1
    Choice bias: -3.9%
    Confidence: 0.53
    M-Ratio: 0.33
    AUROC2: 0.60
```
Output (for fit):
```
+++ Type 1 level +++
  Subject-level estimation (MLE)
    .. finished (0.1 secs).
  Final report
    Parameters estimates (subject-level fit)
        [subject] type1_noise: 0.510 ± 0.018
        [subject] type1_bias: -0.099 ± 0.019
    [subject] Log-likelihood: -717.84 (per sample: -0.3589)
    [subject] Fitting time: 0.13 secs
Type 1 level finished

+++ Type 2 level +++
  Subject-level estimation (MLE)
    .. finished (77.8 secs).
  Final report
    Parameters estimates (subject-level fit)
        [subject] type2_noise: 0.264 ± 0.031
        [subject] type2_criteria: [0.268 ± 0.014, 0.512 ± 0.012, 0.758 ± 0.008]
            [extra] type2_criteria_bias: 0.009 ± 0.008
            [extra] type2_criteria_confidence_bias: -0.009 ± 0.008
    [subject] Log-likelihood: -3425.60 (per sample: -1.713)
    [subject] Fitting time: 32.41 secs
Type 2 level finished
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
type2_noise: 0.264
type2_criteria: [0.268 0.512 0.758]
```

### Supported parameters

| Parameter | Description | Default  |
|----------|----------|----------|
|`type1_noise`|Type 1 noise| Enabled  |
|`type1_bias`|Choice bias| Enabled  |
|`type1_thresh`|Sensory threshold| Disabled |
|`type1_nonlinear_encoding_gain`|Nonlinear encoding| Disabled |
|`type1_nonlinear_encoding_scale`|Nonlinear encoding| Disabled |
|`type2_noise`|Metacognitive noise| Enabled  |
|`type2_evidence_bias`|Metacognitive evidence bias| Disabled |
|`type2_confidence_bias`|Metacognitive confidence bias| Disabled |
|`type2_criteria`|Confidence criteria| Enabled  |
