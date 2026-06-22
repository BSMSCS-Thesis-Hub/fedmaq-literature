FIX: The Gaussian noise equivalence formula in the summary is incorrect. According to Theorem 1 of the paper, the quantization error follows a Gaussian distribution with variance $\sigma^2 n_{i,t}^2$, not $\sigma^2 / n_i^2$. The statement should be corrected to:

"The error from DQ is equivalent to Gaussian noise: $\hat{g}_i^t \sim g_i^t + \mathcal{N}(0, \mathbb{I}_m \sigma^2 n_{i,t}^2)$."

No other mathematical inaccuracies were found, and all other formulas and parameters match the original paper.