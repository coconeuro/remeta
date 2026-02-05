# Architecture

The ReMeta toolbox allows researchers to estimate latent parameters of observers that perform a decision-making task involving confidence ratings. *Type 1 parameters* refer to parameters that describe the decision-making process, while *Type 2 parameters* refer to parameters that describe the confidence rating process. The focus of Remeta is an estimate of **metacognitive noise**, i.e. how accurately observers can monitor their own performance, and **metacognitive bias**, i.e. whether observers tend to be over- or under-confident.

The basic architecture of the toolbox is shown in the figure below.

:::{figure} img/model.png
:align: center
:width: 75%
:::

Based on a stimulus input $x$, the observer computes a decision value $y$, the sign of which determines a type 1 decision $D$. This process can be characterized by a variety of type 1 parameters, including type 1 noise $\sigma_{1}$ (sensory or decisional noise) and a type 1 decision bias $\delta_1$.

The absolute value of the noisy decision value — labeled *type 1 evidence* $z_1$ — is the relevant input to the type 2 level. This "readout" may not be perfect though and could be subject to some readout noise $\sigma_2$ and a multiplicative bias $\varphi_2$. Thus, the evidence at the type 2 stage — type 2 evidence $z_2$ — may be different from $z_1$. A model with unreliable readout is referred to as *noisy-readout* model.

By design, the model assumes that observers represent confidence as probability correct, thus computing $c=p(\text{correct}\mid z_2,\hat{\sigma}_1)$. This is sometimes referred to as *Bayesian confidence*. Nevertheless, for this computation, the observer needs an estimate $\hat{\sigma}_1$ of their own type 1 noise $\sigma_1$.

Estimating one's own type 1 noise $\sigma_1$ is far from trivial and may thus plausibly be subject to some noise itself. This is a second form of metacognitive noise that was suggested by [](doi:10.1038/s41562-022-01464-x). Here, this is referred to as *noisy-temperature* model.

Finally, internal representations of confidence need to be communicated to the experiementer. This likewise is a lossy process, for two reasons: 1) the most confidence scales are discrete and hence required confidence criteria $\gamma_1, \dots, \gamma_n$ to map a continuous internal estimate to a finite number of response options; 2) the mapping itself may be noisy. This last form of metacognitive noise is referred to as a *noisy-report* model.

In ReMeta, the researcher must decide for which of these three sources of metacognitive noise they consider most dominant or most interesting, since only one can be modeled at a time. 