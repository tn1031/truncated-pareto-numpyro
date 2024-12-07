import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax._src.numpy.util import promote_args_inexact
from jax.scipy.stats import pareto
from numpyro.distributions.util import promote_shapes


class TruncatedPareto(dist.Distribution):
    arg_constraints = {
        "scale": dist.constraints.positive,
        "alpha": dist.constraints.positive,
        "low": dist.constraints.positive,
        "high": dist.constraints.greater_than(0),
    }

    def __init__(self, scale, alpha, low=0.0, high=float("inf"), validate_args=None):
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(scale),
            jnp.shape(alpha),
            jnp.shape(low),
            jnp.shape(high),
        )
        self.scale, self.alpha, self.low, self.high = promote_shapes(scale, alpha, low, high)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        log_m = self.logcdf(self.high, self.scale, self.alpha) - self.logcdf(self.low, self.scale, self.alpha)
        log_p = self.logpdf(value, self.scale, self.alpha)
        return jnp.where((self.low < value) * (value < self.high), log_p - log_m, -jnp.inf)

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        minval = lax.exp(self.logcdf(self.low, self.scale, self.alpha))
        maxval = lax.exp(self.logcdf(self.high, self.scale, self.alpha))
        u = jax.random.uniform(key, shape, minval=minval, maxval=maxval)
        return lax.exp(self.logicdf(u, self.scale, self.alpha))

    def logcdf(self, x, x_m, alpha):
        x, x_m, alpha = promote_args_inexact("truncated_pareto.logcdf", x, x_m, alpha)
        h = lax.exp(lax.mul(alpha, lax.log(lax.div(x_m, x))))
        return lax.log1p(lax.neg(h))

    def logicdf(self, x, x_m, alpha):
        x, x_m, alpha = promote_args_inexact("truncated_pareto.logicdf", x, x_m, alpha)
        return lax.sub(lax.log(x_m), lax.div(lax.log1p(lax.neg(x)), alpha))

    def logpdf(self, x, x_m, alpha):
        return pareto.logpdf(x, b=alpha, scale=x_m)

    @dist.constraints.dependent_property
    def support(self):
        return dist.constraints.interval(self.low, self.high)
