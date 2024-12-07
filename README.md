# truncated-pareto-numpyro

This repository provides an implementation of a Truncated Pareto Distribution using the probabilistic programming library [NumPyro](https://github.com/pyro-ppl/numpyro). The Truncated Pareto distribution extends the Pareto distribution by truncating it between specified lower and upper bounds.

## Getting Started
### Prerequisites

- Python 3.10 or later
- JAX and NumPyro installed

### Usage

Here is an example of how to use the TruncatedPareto distribution:

```python
import jax
import numpyro
from numpyro.infer import MCMC, NUTS
from truncated_pareto import TruncatedPareto

# Example parameters
scale = 1.0
alpha = 3.0
low = 1.5
high = 5.0

# Initialize the distribution
truncated_pareto = TruncatedPareto(scale=scale, alpha=alpha, low=low, high=high)

# Sampling
key = jax.random.PRNGKey(0)
samples = truncated_pareto.sample(key, sample_shape=(1000,))

# Log probability
value = 2.0
log_prob = truncated_pareto.log_prob(value)

print(f"Sampled values: {samples}")
print(f"Log probability of {value}: {log_prob}")
```

### Integration with NumPyro Models

You can directly use TruncatedPareto in NumPyro probabilistic models:

```python
import numpyro
import numpyro.distributions as dist

def model():
    scale = 1.0
    alpha = 2.0
    low = 1.5
    high = 4.0
    numpyro.sample("x", TruncatedPareto(scale, alpha, low, high))
```

## API Reference
### Constructor

```python
TruncatedPareto(scale, alpha, low=0.0, high=float("inf"))
```

- **scale** (`float` or `array`): Scale parameter of the Pareto distribution.
- **alpha** (`float` or `array`): Shape parameter of the Pareto distribution.
- **low** (`float` or `array`): Lower bound of the truncation (default: 0.0).
- **high** (`float` or `array`): Upper bound of the truncation (default: inf).

### Methods
- `log_prob(value)`: Returns the log-probability of a given `value`.
- `sample(key, sample_shape=())`: Draws samples from the truncated Pareto distribution.
- `logcdf(x, x_m, alpha)`: Computes the logarithm of the cumulative distribution function.
- `logicdf(x, x_m, alpha)`: Inverse CDF for sampling.
- `logpdf(x, x_m, alpha)`: Computes the logarithm of the probability density function.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

