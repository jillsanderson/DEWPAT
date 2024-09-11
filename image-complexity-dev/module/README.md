## Gaussian Probability Divergences

To avoid excess detail in the main readme, we detail the specifics of our additional patch-wise complexity measures.

We assume that our image has been separated into patches, and we have fit distributional moments to each one, resulting in an empirical mean $`\widehat{\mu}(p)\in\mathbb{R}^{3}`$ and covariance $`\widehat{\Sigma}(p)\in\mathbb{R}^{3\times 3}`$ for each patch $`p`$.

Following the natural Maximum Entropy assumption [1], we consider each patch to be multivariate Gaussian $`\mathcal{G} = \mathcal{N}(\widehat{\mu}(p),\widehat{\Sigma}(p))`$, and assume that a complex image is likely to contain patches are as "different" from each other as possible.
We therefore measure the expected pairwise distance between patches as a simple complexity measure.
Since trivial vector space metrics (e.g., the $`L_2`$ distance) are well-known to be poor image metrics in the context of biological visual perception [2], we utilize distributional divergences instead, which are more robust to perceptually irrelevant perturbations, such as small translations.

Thus, we consider a family of complexity measures on an image $`I`$ defined by
```math
\mathcal{C}_{\mathfrak{D}}(I) = \frac{1}{|P|^2} \sum_{p_i,p_j\in P} \mathfrak{D}\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right],
```
where $`P`$ is the set of patches, $`\mathcal{G}_i`$ is the Gaussian corresponding to patch $`i`$, and $`\mathfrak{D}`$ is a particular parameterizing choice of information-theoretic divergence, the options for which are detailed below. Note that these divergences (and hence the resulting metrics) are simple to compute and computationally tractable, due to the Gaussian assumption.

Implementation-wise, we consider $`\mathfrak{D} \in \{ \mathfrak{D}_{\mathcal{J}}, \mathfrak{D}_{W_2}, \mathfrak{D}_B, \mathfrak{D}_H, \mathfrak{D}_\text{FMATF} \}`$. 

In the following, let $`\mathcal{P},\mathcal{Q}`$ be probability distributions over $`x\in X`$, and $`f_p,f_q`$ be their respective densities.

### Jeffrey's Divergence

Jeffrey's divergence is simply the symmetric KL-divergence:
```math
\mathfrak{D}_\mathcal{J}\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] = \frac{1}{2} \mathfrak{D}_\text{KL}\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] + \frac{1}{2} \mathfrak{D}_\text{KL}\left[ \mathcal{G}_j \mid\mid \mathcal{G}_i \right],
```
where 
```math
\mathfrak{D}_\text{KL}[\mathcal{P},\mathcal{Q}] = \int_X f_p(x) \log\left( \frac{f_p(x)}{f_q(x)} \right) \, dx.
```

### Wasserstein-2 Metric 

For the specific case of Gaussian distributions, the Wasserstein-2 distance simplifies down to the following form:
```math
\mathfrak{D}_{W_2}\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] = || \widehat{\mu}_i - \widehat{\mu}_j ||_2^2 + \text{tr}\left( \widehat{\Sigma}_i + \widehat{\Sigma}_j - 2[\widehat{\Sigma}_j^{1/2} \widehat{\Sigma}_i \widehat{\Sigma}_j^{1/2}]^{1/2} \right).
```

### Bhattacharyya distance 

For notational brevity, define the Bhattacharyya coefficient as
```math
\mathcal{B}[\mathcal{P},\mathcal{Q}] = \int_X \sqrt{f_p(x) f_q(x)} \, dx.
```
Then the Bhattacharyya distance is simply given by
```math
\mathfrak{D}_B\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] = -\log\mathcal{B}[\mathcal{G}_i, \mathcal{G}_j].
```

### Squared Hellinger distance

Utilizing the notation just above, the squared Hellinger distance is given by 
```math
\mathfrak{D}_H\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] = \sqrt{ 1 - \mathcal{B}[\mathcal{G}_i, \mathcal{G}_j] }.
```

### Forstner-Moonen Abou-Moustafa-Torres-Ferries (FM-ATF) density metric

Folllowing [3,4], we utilize the following metric
```math
\mathfrak{D}_\text{FMATF}\left[ \mathcal{G}_i \mid\mid \mathcal{G}_j \right] = d_\mu(\mathcal{G}_i, \mathcal{G}_j)^{1/2} + d_\sigma(\mathcal{G}_i, \mathcal{G}_j)^{1/2},
```
where the normalized (Mahalanobis) distance between means is given by
```math
d_\mu(\mathcal{G}_i, \mathcal{G}_j) = (\widehat{\mu}_i - \widehat{\mu}_j)^T \widehat{\Sigma}_a^{-1} (\widehat{\mu}_i - \widehat{\mu}_j),
```
with $`\widehat{\Sigma}_a = (\widehat{\Sigma}_i + \widehat{\Sigma}_j) / 2`$,
and the squared FM metric [4] on SPD matrices is written
```math
d_\sigma(\mathcal{G}_i, \mathcal{G}_j) = \sum_\ell \log (\lambda_\ell)^2 
```
with $`\text{diag}(\lambda_1,\ldots,\lambda_K) = \Lambda`$ satisfying the generalized eigenvalue problem $`\widehat{\Sigma}_i V = \Lambda \widehat{\Sigma}_j V`$.

Note that this formulation is also specific to Gaussians.

## References

[1] Jaynes, Edwin T. "Information theory and statistical mechanics." Physical review 106.4 (1957): 620.

[2] Zhang, Richard, et al. "The unreasonable effectiveness of deep features as a perceptual metric." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

[3] Abou–Moustafa, Karim T., Fernando De La Torre, and Frank P. Ferrie. "Designing a metric for the difference between Gaussian densities." Brain, Body and Machine. Springer, Berlin, Heidelberg, 2010. 57-70.

[4] Förstner, Wolfgang, and Boudewijn Moonen. "A metric for covariance matrices." Geodesy-the Challenge of the 3rd Millennium. Springer, Berlin, Heidelberg, 2003. 299-309.

