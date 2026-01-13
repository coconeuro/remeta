import numpy as np
from scipy.stats import lognorm, uniform, gumbel_r
from scipy.special import ndtr, gammainc as gammainc_spec, gamma as gamma_spec, log_ndtr, ndtri_exp
from scipy.integrate import quad


def _params_lognorm_mode_std(mode, stddev):
    """
    Compute scipy lognorm's shape and scale to presere mode and SD.
    The analytical formula is exact and was computed with WolframAlpha.

    Parameters
    ----------
    mode : float or array-like
        Mode of the distribution.
    stddev : float or array-like
        Standard deviation of the distribution.

    Returns
    ----------
    shape : float
        Scipy lognorm shape parameter.
    scale : float
        Scipy lognorm scale parameter.
    """
    mode = np.maximum(1e-5, mode)
    a = stddev ** 2 / mode ** 2
    x = 1 / 4 * np.sqrt(np.maximum(1e-300, -(16 * (2 / 3) ** (1 / 3) * a) / (
                np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) +
                                   2 * (2 / 3) ** (2 / 3) * (
                                               np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (
                                               1 / 3) + 1)) + \
        1 / 2 * np.sqrt(
        (4 * (2 / 3) ** (1 / 3) * a) / (np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) -
        (np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) / (2 ** (1 / 3) * 3 ** (2 / 3)) +
        1 / (2 * np.sqrt(np.maximum(1e-300, -(16 * (2 / 3) ** (1 / 3) * a) / (
                    np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (1 / 3) +
                                    2 * (2 / 3) ** (2 / 3) * (
                                                np.sqrt(3) * np.sqrt(256 * a ** 3 + 27 * a ** 2) - 9 * a) ** (
                                                1 / 3) + 1))) + 1 / 2) + \
        1 / 4
    shape = np.sqrt(np.log(x))
    scale = mode * x
    return shape, scale

def _params_truncnorm_mode_std(low, high, x, noise):
    noise = np.maximum(noise, 1e-12)
    u = np.log(noise)
    log_var_target = np.log(np.maximum(noise ** 2, 1e-12))
    # for _ in range(1):  # increase Newton steps for more precision
    # We do only a single update step and thus skip the for loop
    sigma = np.exp(u)
    a, b = (low - x) / sigma, (high - x) / sigma
    Z = np.maximum(ndtr(b) - ndtr(a), 1e-12)
    pa = np.exp(-a**2 / 2) / np.sqrt(2 * np.pi)
    if np.isinf(high):
        term1, term2 = 1 + (a * pa) / Z, pa / Z
    else:
        pb = np.exp(-b**2 / 2) / np.sqrt(2 * np.pi)
        term1, term2 = 1 + (a * pa - b * pb) / Z, (pa - pb) / Z

    var = (sigma * sigma) * np.maximum(term1 - term2 * term2, 1e-12)
    sigma_p = np.exp(u + 1e-3)
    a_p, b_p = (low - x) / sigma_p, (high - x) / sigma_p
    Z_p = np.maximum(ndtr(b_p) - ndtr(a_p), 1e-12)
    pa_p = np.exp(-a_p**2 / 2) / np.sqrt(2 * np.pi)
    if np.isinf(high):
        term1_p, term2_p = 1.0 + (a_p * pa_p) / Z_p, pa_p / Z_p
    else:
        pb_p = np.exp(-b_p**2 / 2) / np.sqrt(2 * np.pi)
        term1_p, term2_p = 1.0 + (a_p * pa_p - b_p * pb_p) / Z_p, (pa_p - pb_p) / Z_p

    var_p = (sigma_p * sigma_p) * np.maximum(term1_p - term2_p**2, 1e-12)
    dlogv_du = (np.log(var_p) - np.log(var)) / 1e-3
    step = np.clip((np.log(var) - log_var_target) /
                np.where(np.abs(dlogv_du)>1e-12, dlogv_du, 1e-12*(np.sign(dlogv_du)+1)),-1.5,1.5)
    u = u - step
    # End of Newton for loop
    sigma = np.exp(u)
    return sigma


class TruncatedLognormalMode:  # noqa
    """
    Implementation of the truncated lognormal distribution.
    Only the upper truncation bound is supported as the lognormal distribution is naturally lower-bounded at zero.

    Parameters
    ----------
    mode : float or array-like
        Original value
    noise : float or array-like
        Noise parameter, corresponds to scipy lognorm s parameter.
    b : float or array-like
        Upper truncation bound.
    """
    def __init__(self, mode, noise, b):
        self.loc = mode
        self.scale = np.maximum(1e-5, mode) * np.exp(noise ** 2)
        self.s = noise
        self.b = b
        self.dist = lognorm(loc=0, scale=self.scale, s=self.s)
        self.lncdf_b = self.dist.cdf(self.b)

    def pdf(self, x):
        pdens = (x <= self.b) * self.dist.pdf(x) / self.lncdf_b
        return pdens

    def cdf(self, x):
        cdens = (x > self.b) + (x <= self.b) * self.dist.cdf(x) / self.lncdf_b
        return cdens

    def rvs(self, size=None):
        if size is None:
            if hasattr(self.scale, '__len__'):
                size = self.scale.shape
            else:
                size = 1
        # cdens = uniform(loc=0, scale=self.b).rvs(size)
        cdens = uniform(loc=0, scale=1).rvs(size)
        x = self.cdf_inv(cdens)
        return x

    def cdf_inv(self, cdens):
        x = (cdens >= 1) * self.b + (cdens < 1) * self.dist.ppf(cdens * self.lncdf_b)
        return x

    def mean(self):
        scale = np.asarray(self.scale, dtype=np.float64)
        s = np.asarray(self.s, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)

        finfo = np.finfo(np.float64)
        tiny = finfo.tiny
        maxlog = np.log(finfo.max)

        s = np.maximum(s, tiny)
        scale = np.maximum(scale, tiny)

        u = b

        out_shape = np.broadcast(scale, s, u).shape
        u_b, mu_b, sig_b = np.broadcast_arrays(u, np.log(scale), s)

        mean = np.full(out_shape, np.nan, dtype=np.float64)

        m = u_b > 0.0
        if np.any(m):
            um = u_b[m]
            mu = mu_b[m]
            sig = sig_b[m]

            logu = np.log(np.maximum(um, tiny))

            z0 = (logu - mu) / sig
            z1 = (logu - mu - sig * sig) / sig

            logZ  = log_ndtr(z0)
            logF1 = log_ndtr(z1)

            log_mean = mu + 0.5 * sig * sig + (logF1 - logZ)
            mean[m] = np.exp(np.clip(log_mean, -np.inf, maxlog))

        return mean

    def std(self):
        scale = np.asarray(self.scale, dtype=np.float64)
        s = np.asarray(self.s, dtype=np.float64)
        b = np.asarray(self.b, dtype=np.float64)

        finfo = np.finfo(np.float64)
        tiny = finfo.tiny
        maxlog = np.log(finfo.max)

        s = np.maximum(s, tiny)
        scale = np.maximum(scale, tiny)

        u = b  # no shift; dist.loc == 0

        out_shape = np.broadcast(scale, s, u).shape
        u_b, mu_b, sig_b = np.broadcast_arrays(u, np.log(scale), s)

        std = np.zeros(out_shape, dtype=np.float64)

        m = u_b > 0.0
        if np.any(m):
            um = u_b[m]
            mu = mu_b[m]
            sig = sig_b[m]

            logu = np.log(np.maximum(um, tiny))

            z0 = (logu - mu) / sig
            z1 = (logu - mu - sig * sig) / sig
            z2 = (logu - mu - 2.0 * sig * sig) / sig

            logZ  = log_ndtr(z0)
            logF1 = log_ndtr(z1)
            logF2 = log_ndtr(z2)

            log_mean_y = mu + 0.5 * sig * sig + (logF1 - logZ)
            log_m2_y   = 2.0 * mu + 2.0 * sig * sig + (logF2 - logZ)

            t = 2.0 * log_mean_y - log_m2_y
            t = np.minimum(t, 0.0)

            m2 = np.exp(np.clip(log_m2_y, -np.inf, maxlog))
            var = m2 * (-np.expm1(t))
            var = np.maximum(var, 0.0)

            std[m] = np.sqrt(var)

        return std


class TruncatedGumbelModeSD:
    """
    Truncated Gumbel-R distribution on [0, upper] with:
      - mode preserved (exact after truncation)
      - noise matched to the *truncated* standard deviation (accurate; fixed-cost Newton)

    Supports vectorized parameters (NumPy arrays) and implements:
      pdf, logpdf, cdf, ppf, rvs
    """

    def __init__(self, mode, noise, upper=np.inf, n_newton=6, eps=1e-12):
        self.eps = float(eps)
        self.n_newton = int(n_newton)

        self.mode = np.asarray(mode, dtype=np.float64)
        self.noise = np.asarray(noise, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

        self.low = 0.0
        self.finite_up = np.isfinite(self.upper)

        # Truncated-mode preservation: clamp requested mode into support
        loc = np.maximum(self.mode, self.low)
        loc = np.where(self.finite_up, np.minimum(loc, self.upper), loc)
        self.loc = loc

        # Maximum achievable truncated std on [0, upper] is that of Uniform(0, upper)
        # (this distribution approaches Uniform as scale -> infinity)
        sd_cap = np.where(self.finite_up, (self.upper - self.low) / np.sqrt(12.0), np.inf)

        # Effective noise is capped at sd_cap (tiny slack avoids chasing the asymptote)
        self.noise_eff = np.minimum(self.noise, sd_cap * (1.0 - 1e-12))
        self.noise_eff = np.maximum(self.noise_eff, 0.0)

        # Solve scale to match truncated std ~= noise
        # self.scale = self._solve_scale_for_trunc_std(self.loc, self.noise, self.upper)
        self.scale = self._solve_scale_for_trunc_std(self.loc, self.noise_eff, self.upper)

        # Truncation normalization constants
        self.F0 = self._gumbel_cdf(self.low, loc=self.loc, scale=self.scale)
        self.Fu = np.where(
            self.finite_up,
            self._gumbel_cdf(self.upper, loc=self.loc, scale=self.scale),
            1.0,
        )
        self.Z = np.maximum(self.Fu - self.F0, self.eps)

    # -----------------------
    # Public distribution API
    # -----------------------

    def pdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_b, loc_b, scale_b, up_b, Z_b = np.broadcast_arrays(
            x, self.loc, self.scale, self.upper, self.Z
        )
        mask = (x_b >= self.low) & (x_b <= up_b)
        out = np.zeros_like(x_b, dtype=np.float64)
        out[mask] = gumbel_r.pdf(x_b[mask], loc=loc_b[mask], scale=scale_b[mask]) / Z_b[mask]
        return out

    def logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_b, loc_b, scale_b, up_b, Z_b = np.broadcast_arrays(
            x, self.loc, self.scale, self.upper, self.Z
        )
        mask = (x_b >= self.low) & (x_b <= up_b)
        out = np.full_like(x_b, -np.inf, dtype=np.float64)
        out[mask] = gumbel_r.logpdf(x_b[mask], loc=loc_b[mask], scale=scale_b[mask]) - np.log(Z_b[mask])
        return out

    def cdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_b, loc_b, scale_b, F0_b, Z_b = np.broadcast_arrays(
            x, self.loc, self.scale, self.F0, self.Z
        )
        Fx = self._gumbel_cdf(x_b, loc=loc_b, scale=scale_b)
        return np.clip((Fx - F0_b) / Z_b, 0.0, 1.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=np.float64)
        q = np.clip(q, 0.0, 1.0)
        q_b, loc_b, scale_b, F0_b, Z_b = np.broadcast_arrays(
            q, self.loc, self.scale, self.F0, self.Z
        )
        u = F0_b + q_b * Z_b
        return self._gumbel_ppf(u, loc=loc_b, scale=scale_b, eps=self.eps)

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)

        # Draw U ~ Uniform(F0, Fu) and invert via base PPF
        if size is None:
            # Use parameter broadcast shape
            shape = np.broadcast(self.loc, self.scale, self.F0, self.Fu).shape
            u = rng.uniform(self.F0, self.Fu, size=shape)
            return self._gumbel_ppf(u, loc=self.loc, scale=self.scale, eps=self.eps)

        u = rng.uniform(self.F0, self.Fu, size=size)
        return self._gumbel_ppf(u, loc=self.loc, scale=self.scale, eps=self.eps)

    # -----------------------
    # Internal helpers
    # -----------------------

    @staticmethod
    def _lower_gamma(s, x):
        # γ(s, x) = gammainc(s, x) * Γ(s)
        return gammainc_spec(s, x) * gamma_spec(s)

    @staticmethod
    def _exp_no_overflow(z):
        """
        Safe exp(z) that never overflows:
        - Computes exp only on entries that are <= log(max_float)
        - Returns +inf where exp would overflow
        """
        z = np.asarray(z, dtype=np.float64)
        log_max = np.log(np.finfo(np.float64).max)  # ~709.782712893384

        out = np.empty_like(z, dtype=np.float64)
        mask = z <= log_max

        # Compute exp only where it's safe
        out[mask] = np.exp(z[mask])
        out[~mask] = np.inf
        return out

    @staticmethod
    def _exp_masked(z):
        """exp(z) evaluated only where it won't overflow; returns inf otherwise."""
        z = np.asarray(z, dtype=np.float64)
        log_max = np.log(np.finfo(np.float64).max)
        out = np.empty_like(z, dtype=np.float64)
        m = z <= log_max
        out[m] = np.exp(z[m])
        out[~m] = np.inf
        return out

    @classmethod
    def _gumbel_logcdf(cls, x, loc, scale):
        """
        log CDF of Gumbel_R(loc, scale): log F = -exp(-(x-loc)/scale)
        Computed stably without overflow.
        """
        x = np.asarray(x, dtype=np.float64)
        loc = np.asarray(loc, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)

        t = -(x - loc) / scale  # can be huge
        et = cls._exp_masked(t)  # may be inf
        return -et  # logcdf = -exp(t); if et=inf => -inf

    @classmethod
    def _gumbel_cdf(cls, x, loc, scale):
        """CDF computed as exp(logcdf) safely."""
        return np.exp(cls._gumbel_logcdf(x, loc, scale))

    @staticmethod
    def _gumbel_ppf(p, loc, scale, eps=1e-15):
        """
        PPF: x = loc - scale * log(-log(p))
        with clipping to avoid log(0).
        """
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, eps, 1.0 - eps)
        return loc - scale * np.log(-np.log(p))

    def _trunc_std_gumbel_standard(self, a, b, h=1e-4):
        """
        Std of Y ~ Gumbel_R(0,1) truncated to [a, b] in y-space (b may be +inf).
        Vectorized. Uses t = exp(-y) => Exp(1) truncated to [exp(-b), exp(-a)].
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # t-limits: l = exp(-b), u = exp(-a). If b=+inf => l=0
        # b is typically >= 0 given your loc clamping, but keep this robust anyway.
        l = np.where(np.isfinite(b), np.exp(-b), 0.0)

        # SAFE: u = exp(-a) can overflow when a is very negative.
        # In that regime, u should be treated as +inf.
        u = np.where(np.isfinite(a), self._exp_no_overflow(-a), np.inf)

        # Stable normalizer for Exp(1) truncated to [l, u]:
        # Z = e^{-l} - e^{-u}.
        # - If u = +inf, e^{-u}=0 => Z = e^{-l}.
        # - If u ~ l, use expm1 to avoid catastrophic cancellation.
        exp_neg_l = np.exp(-l)
        exp_neg_u = np.exp(-u)  # exp(-inf)=0 is fine

        finite_u = np.isfinite(u)
        Z_stable_finite = exp_neg_l * (-np.expm1(-(u - l)))  # e^{-l}*(1 - e^{-(u-l)})
        Z = np.where(finite_u, Z_stable_finite, exp_neg_l)
        Z = np.maximum(Z, self.eps)

        def dG(s):
            # ΔG(s) = ∫_l^u t^{s-1} e^{-t} dt = γ(s,u) - γ(s,l)
            # Works with u=+inf because gammainc(s, inf)=1 => γ(s,inf)=Γ(s).
            return self._lower_gamma(s, u) - self._lower_gamma(s, l)

        dG1 = dG(1.0)
        dGp = dG(1.0 + h)
        dGm = dG(1.0 - h)

        # K1 = ∫ ln(t) e^{-t} dt = d/ds ΔG(s)|_{s=1}
        K1 = (dGp - dGm) / (2.0 * h)
        # K2 = ∫ (ln t)^2 e^{-t} dt = d^2/ds^2 ΔG(s)|_{s=1}
        K2 = (dGp - 2.0 * dG1 + dGm) / (h * h)

        # For Y = -ln t:
        I0 = Z
        I1 = -K1
        I2 = K2

        m1 = I1 / I0
        v = np.maximum(I2 / I0 - m1 * m1, self.eps)
        return np.sqrt(v)

    def _solve_scale_for_trunc_std(self, loc, noise, upper):
        """
        Fixed-cost Newton updates in u=log(scale) to match truncated std to 'noise'.
        Fully vectorized; no bracketing / adaptive iteration count.
        """
        noise = np.maximum(np.asarray(noise, dtype=np.float64), 0.0)
        loc = np.asarray(loc, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        finite_up = np.isfinite(upper)

        # Initial guess from untruncated std: std = scale * pi/sqrt(6)
        c = np.pi / np.sqrt(6.0)
        scale0 = np.maximum(noise / c, self.eps) * 1.25
        u = np.log(scale0)

        log_target = np.log(np.maximum(noise, self.eps))

        for _ in range(self.n_newton):
            scale = np.exp(u)

            a = (self.low - loc) / scale
            b = np.where(finite_up, (upper - loc) / scale, np.inf)

            std_y = self._trunc_std_gumbel_standard(a, b)
            std_x = scale * std_y

            f = np.log(np.maximum(std_x, self.eps)) - log_target

            # Finite-diff derivative of log(std_x) wrt u=log(scale)
            du = 1e-3
            scale_p = np.exp(u + du)
            a_p = (self.low - loc) / scale_p
            b_p = np.where(finite_up, (upper - loc) / scale_p, np.inf)
            std_y_p = self._trunc_std_gumbel_standard(a_p, b_p)
            std_x_p = scale_p * std_y_p

            dlogstd_du = (
                np.log(np.maximum(std_x_p, self.eps)) - np.log(np.maximum(std_x, self.eps))
            ) / du
            dlogstd_du = np.where(
                np.abs(dlogstd_du) > self.eps,
                dlogstd_du,
                np.sign(dlogstd_du) * self.eps + self.eps,
            )

            step = f / dlogstd_du
            step = np.clip(step, -1.5, 1.5)  # damping
            u = u - step

        return np.exp(u)

    def _trunc_mean_gumbel_standard(self, a, b, h=1e-4):
        """
        Mean of Y ~ Gumbel_R(0,1) truncated to [a, b] in y-space (b may be +inf).
        Vectorized. Uses t = exp(-y) => Exp(1) truncated to [exp(-b), exp(-a)].
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # t-limits: l = exp(-b), u = exp(-a). If b=+inf => l=0
        l = np.where(np.isfinite(b), np.exp(-b), 0.0)

        # u = exp(-a) can overflow when a is very negative -> treat as +inf
        u = np.where(np.isfinite(a), self._exp_no_overflow(-a), np.inf)

        # Stable normalizer for Exp(1) truncated to [l, u]:
        # Z = e^{-l} - e^{-u}
        exp_neg_l = np.exp(-l)
        exp_neg_u = np.exp(-u)  # exp(-inf)=0 is fine

        finite_u = np.isfinite(u)
        Z_stable_finite = exp_neg_l * (-np.expm1(-(u - l)))  # e^{-l}*(1 - e^{-(u-l)})
        Z = np.where(finite_u, Z_stable_finite, exp_neg_l)
        Z = np.maximum(Z, self.eps)

        def dG(s):
            # ΔG(s) = ∫_l^u t^{s-1} e^{-t} dt = γ(s,u) - γ(s,l)
            return self._lower_gamma(s, u) - self._lower_gamma(s, l)

        dGp = dG(1.0 + h)
        dGm = dG(1.0 - h)

        # K1 = ∫_l^u ln(t) e^{-t} dt = d/ds ΔG(s)|_{s=1}
        K1 = (dGp - dGm) / (2.0 * h)

        # For Y = -ln t, numerator = ∫ (-ln t) e^{-t} dt = -K1
        mean_y = (-K1) / Z
        return mean_y

    def mean(self):
        """
        Returns the (truncated) mean E[X] over support [0, upper].

        Uses an analytic transform to Exp(1) (no quadrature), fully vectorized.
        """
        loc = np.asarray(self.loc, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)

        finite_up = np.isfinite(upper)

        # y-space truncation bounds for Y = (X - loc)/scale, with X in [0, upper]
        a = (self.low - loc) / scale
        b = np.where(finite_up, (upper - loc) / scale, np.inf)

        mean_y = self._trunc_mean_gumbel_standard(a, b)
        mean_x = loc + scale * mean_y

        # Broadcast to the same shape as other moment methods
        shape = np.broadcast(loc, scale, upper).shape
        return np.broadcast_to(mean_x, shape)

    def std(self):
        """
        Returns the (truncated) standard deviation.

        By construction, this distribution matches the truncated std to `noise`
        (up to Newton/eps tolerances), so we can return it directly.
        """
        # return np.broadcast_to(self.noise, np.broadcast(self.loc, self.scale, self.upper).shape)
        shape = np.broadcast(self.loc, self.scale, self.upper).shape
        return np.broadcast_to(self.noise_eff, shape)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **quad_kw):
        """
        SciPy-style expectation.

        Parameters
        ----------
        func : callable or None
            Function f(x) whose expectation is computed. If None, uses f(x)=x.
        lb, ub : float or array-like or None
            Integration bounds. If None, defaults to the distribution's support bounds:
            lb -> self.low, ub -> self.upper.
            Bounds are intersected with [self.low, self.upper] elementwise.
        conditional : bool
            If False (default), return ∫ f(x) dF(x) over [lb, ub] (unnormalized).
            If True, return E[f(X) | lb <= X <= ub].
        **quad_kw :
            Passed to scipy.integrate.quad (e.g. epsabs, epsrel, limit).

        Returns
        -------
        out : ndarray
            Expectation value(s), broadcasted over parameter shapes and bound shapes.
        """
        if func is None:
            func = lambda x: x  # noqa: E731

        # Defaults and broadcasting for bounds
        if lb is None:
            lb = self.low
        if ub is None:
            ub = self.upper

        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        # Broadcast parameters and bounds to a common shape
        loc_b, scale_b, up_b, Z_b, lb_b, ub_b = np.broadcast_arrays(
            self.loc, self.scale, self.upper, self.Z, lb, ub
        )

        out = np.empty_like(loc_b, dtype=np.float64)

        # Reasonable quad defaults (can be overridden by caller)
        quad_defaults = dict(epsabs=1e-10, epsrel=1e-10, limit=200)
        quad_defaults.update(quad_kw)

        it = np.nditer(
            [loc_b, scale_b, up_b, Z_b, lb_b, ub_b, out],
            flags=["multi_index", "refs_ok"],
            op_flags=[["readonly"]]*6 + [["writeonly"]],
        )

        for l, s, u, z, a_req, b_req, out_cell in it:
            # Intersect requested bounds with the support [self.low, u]
            a = float(max(self.low, a_req))
            b = float(min(u, b_req))

            # Empty / invalid interval -> integral is 0; conditional -> nan
            if not np.isfinite(a) or not np.isfinite(b) or (b <= a):
                out_cell[...] = np.nan if conditional else 0.0
                continue

            # Build a scalar pdf for this parameter point (avoids broadcasting inside quad)
            # We compute truncated pdf directly to ensure correct normalization.
            dist = gumbel_r(loc=float(l), scale=float(s))

            def pdf_trunc(x):
                # x is scalar float from quad
                if x < self.low or x > u:
                    return 0.0
                return dist.pdf(x) / float(z)

            def integrand(x):
                return func(x) * pdf_trunc(x)

            val, _err = quad(integrand, a, b, **quad_defaults)

            if not conditional:
                out_cell[...] = val
            else:
                # Conditional expectation divides by prob mass in [a,b]
                p = float(self.cdf(b) - self.cdf(a))
                if p <= self.eps:
                    out_cell[...] = np.nan
                else:
                    out_cell[...] = val / p

        return out


class TruncatedGumbelMode:
    """
    Fast truncated Gumbel-R on [0, upper] (upper in {1, +inf}) with:
      - mode preserved (after truncation)
      - noise mapped directly to underlying scale (not moment-matched)

    Implements: pdf, logpdf, cdf, ppf, rvs
    """

    @staticmethod
    def _exp_masked(z):
        """
        exp(z) computed only where it cannot overflow; returns +inf otherwise.
        """
        z = np.asarray(z, dtype=np.float64)
        log_max = np.log(np.finfo(np.float64).max)  # ~709.78
        out = np.empty_like(z, dtype=np.float64)
        m = z <= log_max
        out[m] = np.exp(z[m])
        out[~m] = np.inf
        return out

    @classmethod
    def _gumbel_logcdf(cls, x, loc, scale):
        """
        log CDF of Gumbel_R(loc, scale):
          F(x) = exp(-exp(-(x-loc)/scale))
          logF = -exp(-(x-loc)/scale)
        Computed without overflow.
        """
        x = np.asarray(x, dtype=np.float64)
        loc = np.asarray(loc, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        t = -(x - loc) / scale                      # can be huge
        et = cls._exp_masked(t)                     # may be +inf
        return -et                                  # if et=inf => -inf

    @classmethod
    def _gumbel_cdf(cls, x, loc, scale):
        return np.exp(cls._gumbel_logcdf(x, loc, scale))

    @staticmethod
    def _gumbel_ppf(p, loc, scale, eps):
        """
        PPF of Gumbel_R(loc, scale):
          x = loc - scale * log(-log(p))
        Clip p to avoid log(0) and log(negative).
        """
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, eps, 1.0 - eps)
        return loc - scale * np.log(-np.log(p))

    def __init__(self, mode, noise, upper=np.inf, eps=1e-12):
        self.eps = float(eps)

        self.mode = np.asarray(mode, dtype=np.float64)
        self.noise = np.asarray(noise, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

        self.low = 0.0
        self.finite_up = np.isfinite(self.upper)

        # Mode preservation for truncated distribution: clamp into support
        loc = np.maximum(self.mode, self.low)
        loc = np.where(self.finite_up, np.minimum(loc, self.upper), loc)
        self.loc = loc

        scale = np.maximum(self.noise, self.eps)
        self.scale = np.maximum(scale, self.eps)

        # Truncation constants (SAFE; avoids SciPy overflow warnings)
        self.F0 = self._gumbel_cdf(self.low, loc=self.loc, scale=self.scale)
        self.Fu = np.where(
            self.finite_up,
            self._gumbel_cdf(self.upper, loc=self.loc, scale=self.scale),
            1.0,
        )
        self.Z = np.maximum(self.Fu - self.F0, self.eps)

    def pdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_b, loc_b, scale_b, up_b, Z_b = np.broadcast_arrays(
            x, self.loc, self.scale, self.upper, self.Z
        )
        mask = (x_b >= self.low) & (x_b <= up_b)
        out = np.zeros_like(x_b, dtype=np.float64)
        out[mask] = gumbel_r.pdf(x_b[mask], loc=loc_b[mask], scale=scale_b[mask]) / Z_b[mask]
        return out

    def logpdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_b, loc_b, scale_b, up_b, Z_b = np.broadcast_arrays(
            x, self.loc, self.scale, self.upper, self.Z
        )
        mask = (x_b >= self.low) & (x_b <= up_b)
        out = np.full_like(x_b, -np.inf, dtype=np.float64)
        out[mask] = (
            gumbel_r.logpdf(x_b[mask], loc=loc_b[mask], scale=scale_b[mask])
            - np.log(Z_b[mask])
        )
        return out

    def cdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_b, loc_b, scale_b, F0_b, Z_b = np.broadcast_arrays(
            x, self.loc, self.scale, self.F0, self.Z
        )
        Fx = self._gumbel_cdf(x_b, loc=loc_b, scale=scale_b)  # SAFE
        return np.clip((Fx - F0_b) / Z_b, 0.0, 1.0)

    def ppf(self, q):
        q = np.asarray(q, dtype=np.float64)
        q = np.clip(q, 0.0, 1.0)
        q_b, loc_b, scale_b, F0_b, Z_b = np.broadcast_arrays(
            q, self.loc, self.scale, self.F0, self.Z
        )
        u = F0_b + q_b * Z_b
        return self._gumbel_ppf(u, loc=loc_b, scale=scale_b, eps=self.eps)  # SAFE

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        if size is None:
            shape = np.broadcast(self.loc, self.scale, self.F0, self.Fu).shape
            u = rng.uniform(self.F0, self.Fu, size=shape)
            return self._gumbel_ppf(u, loc=self.loc, scale=self.scale, eps=self.eps)  # SAFE
        u = rng.uniform(self.F0, self.Fu, size=size)
        return self._gumbel_ppf(u, loc=self.loc, scale=self.scale, eps=self.eps)      # SAFE

    def stats(self):
        """
        Returns (mean, var) of the truncated distribution.
        Computed via exact expectation under truncation.
        """
        loc = self.loc
        scale = self.scale
        low = self.low
        up = self.upper
        Z = self.Z

        # Broadcast all parameters
        loc_b, scale_b, up_b, Z_b = np.broadcast_arrays(
            loc, scale, up, Z
        )

        mean = np.empty_like(loc_b, dtype=np.float64)
        var = np.empty_like(loc_b, dtype=np.float64)

        it = np.nditer(
            [loc_b, scale_b, up_b, Z_b, mean, var],
            flags=["multi_index", "refs_ok"],
            op_flags=[["readonly"]]*4 + [["writeonly"]]*2,
        )

        for l, s, u, z, m_out, v_out in it:
            dist = gumbel_r(loc=float(l), scale=float(s))

            if np.isfinite(u):
                m1 = dist.expect(lambda x: x, lb=low, ub=float(u))
                m2 = dist.expect(lambda x: x * x, lb=low, ub=float(u))
            else:
                m1 = dist.expect(lambda x: x, lb=low)
                m2 = dist.expect(lambda x: x * x, lb=low)

            m = m1 / z
            v = m2 / z - m * m

            m_out[...] = m
            v_out[...] = np.maximum(v, 0.0)

        return mean, var

    def std(self):
        """
        Returns the standard deviation of the truncated distribution.
        """
        _, var = self.stats()
        return np.sqrt(var)

    @staticmethod
    def _lower_gamma(s, x):
        # γ(s, x) = gammainc(s, x) * Γ(s)
        return gammainc_spec(s, x) * gamma_spec(s)

    @staticmethod
    def _exp_no_overflow(z):
        """
        Safe exp(z) that never overflows:
        - Computes exp only on entries <= log(max_float)
        - Returns +inf where exp would overflow
        """
        z = np.asarray(z, dtype=np.float64)
        log_max = np.log(np.finfo(np.float64).max)  # ~709.78

        out = np.empty_like(z, dtype=np.float64)
        m = z <= log_max
        out[m] = np.exp(z[m])
        out[~m] = np.inf
        return out

    def _trunc_mean_gumbel_standard(self, a, b, h=1e-4):
        """
        Mean of Y ~ Gumbel_R(0,1) truncated to [a, b] in y-space (b may be +inf).

        Uses transform t = exp(-y) so that T ~ Exp(1), truncated to [l, u] with:
          l = exp(-b) (or 0 if b=+inf), u = exp(-a) (may be +inf).

        Then Y = -ln T and:
          E[Y] = (∫_l^u (-ln t) e^{-t} dt) / (∫_l^u e^{-t} dt).

        The numerator is computed as a finite-difference derivative of
          ΔG(s) = ∫_l^u t^{s-1} e^{-t} dt = γ(s,u) - γ(s,l)
        at s=1.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # t-limits: l = exp(-b), u = exp(-a). If b=+inf => l=0
        l = np.where(np.isfinite(b), np.exp(-b), 0.0)
        u = np.where(np.isfinite(a), self._exp_no_overflow(-a), np.inf)

        # Normalizer for Exp(1) truncated to [l, u]:
        # Z = e^{-l} - e^{-u}, computed stably.
        exp_neg_l = np.exp(-l)
        exp_neg_u = np.exp(-u)  # exp(-inf)=0

        finite_u = np.isfinite(u)
        Z_finite = exp_neg_l * (-np.expm1(-(u - l)))  # e^{-l}*(1 - e^{-(u-l)})
        Z = np.where(finite_u, Z_finite, exp_neg_l)
        Z = np.maximum(Z, self.eps)

        def dG(s):
            # ΔG(s) = γ(s,u) - γ(s,l) ; works with u=+inf because γ(s,inf)=Γ(s)
            return self._lower_gamma(s, u) - self._lower_gamma(s, l)

        dGp = dG(1.0 + h)
        dGm = dG(1.0 - h)

        # K1 = ∫_l^u ln(t) e^{-t} dt = d/ds ΔG(s)|_{s=1}
        K1 = (dGp - dGm) / (2.0 * h)

        # E[Y] = (-K1) / Z
        return (-K1) / Z

    def mean(self):
        """
        Returns the (truncated) mean E[X] over support [0, upper].

        Analytic, vectorized, fixed-cost; avoids quad/stats().
        """
        loc = np.asarray(self.loc, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)

        finite_up = np.isfinite(upper)

        # Truncation bounds in standard y-space: Y = (X - loc)/scale
        a = (self.low - loc) / scale
        b = np.where(finite_up, (upper - loc) / scale, np.inf)

        mean_y = self._trunc_mean_gumbel_standard(a, b)
        mean_x = loc + scale * mean_y

        # Ensure consistent broadcasting with other methods
        shape = np.broadcast(loc, scale, upper).shape
        return np.broadcast_to(mean_x, shape)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **quad_kw):
        """
        SciPy-style expectation.

        Parameters
        ----------
        func : callable or None
            Function f(x) whose expectation is computed. If None, uses f(x)=x.
        lb, ub : float or array-like or None
            Integration bounds. If None, defaults to the distribution's support bounds:
            lb -> self.low, ub -> self.upper.
            Bounds are intersected with [self.low, self.upper] elementwise.
        conditional : bool
            If False (default), return ∫ f(x) dF(x) over [lb, ub] (unnormalized).
            If True, return E[f(X) | lb <= X <= ub].
        **quad_kw :
            Passed to scipy.integrate.quad (e.g. epsabs, epsrel, limit).

        Returns
        -------
        out : ndarray
            Expectation value(s), broadcasted over parameter shapes and bound shapes.
        """
        # Fast path: unconditional mean over full support
        if func is None and lb is None and ub is None and not conditional:
            mean, _ = self.stats()
            return mean

        if func is None:
            func = lambda x: x  # noqa: E731

        if lb is None:
            lb = self.low
        if ub is None:
            ub = self.upper

        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        # Broadcast parameters and bounds
        loc_b, scale_b, up_b, Z_b, lb_b, ub_b = np.broadcast_arrays(
            self.loc, self.scale, self.upper, self.Z, lb, ub
        )

        out = np.empty_like(loc_b, dtype=np.float64)

        # quad defaults (caller can override)
        quad_defaults = dict(epsabs=1e-10, epsrel=1e-10, limit=200)
        quad_defaults.update(quad_kw)

        it = np.nditer(
            [loc_b, scale_b, up_b, Z_b, lb_b, ub_b, out],
            flags=["multi_index", "refs_ok"],
            op_flags=[["readonly"]]*6 + [["writeonly"]],
        )

        for l, s, u, z, a_req, b_req, out_cell in it:
            # Intersect requested bounds with the support [low, u]
            a = float(max(self.low, a_req))
            b = float(min(u, b_req))

            # Empty / invalid interval
            if not np.isfinite(a) or not np.isfinite(b) or (b <= a):
                out_cell[...] = np.nan if conditional else 0.0
                continue

            dist = gumbel_r(loc=float(l), scale=float(s))
            z = float(z)

            def pdf_trunc(x):
                if x < self.low or x > u:
                    return 0.0
                return dist.pdf(x) / z

            def integrand(x):
                return func(x) * pdf_trunc(x)

            val, _err = quad(integrand, a, b, **quad_defaults)

            if not conditional:
                out_cell[...] = val
            else:
                # Probability mass in [a,b] under the truncated law
                # Using the class CDF ensures consistent truncation/normalization.
                p = float(self.cdf(b) - self.cdf(a))
                if p <= self.eps:
                    out_cell[...] = np.nan
                else:
                    out_cell[...] = val / p

        return out


class TruncatedLognormalModeSD:
    """
    Truncated lognormal distribution with:

      - lower bound fixed at 0
      - upper bound = 1 or +inf
      - mode  = mode of the *truncated* distribution
      - noise = standard deviation of the *truncated* distribution

    Fast, vectorized, broadcasting-safe.
    """

    _LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)

    def __init__(self, mode, noise, upper=np.inf, n_newton=3):
        self.mode = np.asarray(mode, float)
        self.noise = np.asarray(noise, float)
        self.upper = float(upper)

        if not (np.isinf(self.upper) or np.isclose(self.upper, 1.0)):
            raise ValueError("upper must be 1.0 or np.inf")

        self.newton_iters = int(n_newton)

        # --- parameter solve ---
        if np.isinf(self.upper):
            self._mu, self._s = self._mode_std_underlying(self.mode, self.noise)
        else:
            self._mu, self._s = self._mode_std_truncated(
                self.mode, self.noise, self.upper, self.newton_iters
            )

        # --- normalization constant (ALWAYS defined) ---
        if np.isinf(self.upper):
            self._logZ = np.zeros_like(self._mu)
        else:
            s = np.maximum(self._s, np.finfo(float).tiny)
            self._logZ = log_ndtr((-self._mu) / s)

    # ------------------------------------------------------------------
    # Parameter solvers
    # ------------------------------------------------------------------

    @staticmethod
    def _trunc_moments_logspace(mu, s, upper_is_finite=True):
        """
        For X ~ LogNormal(mu, s) truncated to (0, 1], return (mean, m2, var)
        computed stably using log-space. Assumes upper=1 when upper_is_finite=True.
        """
        finfo = np.finfo(np.float64)
        tiny = finfo.tiny
        maxlog = np.log(finfo.max)  # ~709.78

        s = np.maximum(s, tiny)

        if not upper_is_finite:
            # Untruncated:
            log_mean = mu + 0.5 * s * s
            log_m2   = 2.0 * mu + 2.0 * s * s

            mean = np.exp(np.clip(log_mean, -np.inf, maxlog))
            m2   = np.exp(np.clip(log_m2,   -np.inf, maxlog))

            # var = m2 - mean^2, stable because 2*log_mean - log_m2 = -s^2 <= 0
            var = m2 * (1.0 - np.exp(-s * s))
            var = np.maximum(var, 0.0)
            return mean, m2, var

        # Truncated to X <= 1:
        # u=1 => logu=0, so z0 = (0 - mu)/s = -mu/s
        z0 = (-mu) / s
        z1 = (-mu - s * s) / s
        z2 = (-mu - 2.0 * s * s) / s

        logZ  = log_ndtr(z0)
        logF1 = log_ndtr(z1)
        logF2 = log_ndtr(z2)

        log_r1 = logF1 - logZ
        log_r2 = logF2 - logZ

        log_mean = mu + 0.5 * s * s + log_r1
        log_m2   = 2.0 * mu + 2.0 * s * s + log_r2

        mean = np.exp(np.clip(log_mean, -np.inf, maxlog))
        m2   = np.exp(np.clip(log_m2,   -np.inf, maxlog))

        # var = exp(log_m2) * (1 - exp(2*log_mean - log_m2)) stably
        t = 2.0 * log_mean - log_m2
        t = np.minimum(t, 0.0)           # clamp tiny positive roundoff
        var = m2 * (-np.expm1(t))         # = m2*(1-exp(t)) stable when t~0
        var = np.maximum(var, 0.0)

        return mean, m2, var

    @staticmethod
    def _mode_std_underlying(mode, std, iters=7):
        """Underlying (untruncated) lognormal mode/std → mu, sigma."""
        mode = np.maximum(mode, np.finfo(float).tiny)
        std = np.maximum(std, 0.0)

        r = (std / mode) ** 2
        t = np.where(r < 1.0, 1.0 + r, r ** 0.25)
        t = np.maximum(t, 1.0 + 1e-12)

        for _ in range(iters):
            f = t**4 - t**3 - r
            fp = t**2 * (4 * t - 3)
            t -= f / np.maximum(fp, 1e-12)
            t = np.maximum(t, 1.0 + 1e-12)

        s2 = np.log(t)
        s = np.sqrt(np.maximum(s2, 0.0))
        mu = np.log(mode) + s2
        return mu, s

    @staticmethod
    def _mode_std_truncated(mode, target_std, upper, iters):
        """
        Solve for mu, s such that:
          - truncated mode = mode
          - truncated std  = target_std
        """

        if np.isfinite(upper):
            sd_cap = upper / np.sqrt(12.0)
            target_std = np.minimum(target_std, sd_cap * (1.0 - 1e-12))

        mode = np.maximum(mode, np.finfo(float).tiny)
        target_std = np.maximum(target_std, 0.0)

        # initial guess: underlying solution
        mu, s = TruncatedLognormalModeSD._mode_std_underlying(mode, target_std)

        for _ in range(iters):
            mu = np.log(mode) + s * s

            z0 = (-mu) / s
            z1 = (-mu - s * s) / s
            z2 = (-mu - 2 * s * s) / s

            logZ = log_ndtr(z0)
            r1 = np.exp(log_ndtr(z1) - logZ)
            r2 = np.exp(log_ndtr(z2) - logZ)

            _, _, var = TruncatedLognormalModeSD._trunc_moments_logspace(mu, s, upper_is_finite=True)
            cur_std = np.sqrt(np.maximum(var, 1e-300))

            # Finite-difference derivative (also stable)
            eps = 1e-4
            sp = s * (1 + eps)
            mup = np.log(mode) + sp * sp

            _, _, varp = TruncatedLognormalModeSD._trunc_moments_logspace(mup, sp, upper_is_finite=True)
            stdp = np.sqrt(np.maximum(varp, 1e-300))

            dstd_ds = (stdp - cur_std) / (sp - s)
            s -= (cur_std - target_std) / np.maximum(dstd_ds, 1e-12)
            s = np.maximum(s, 1e-6)

        mu = np.log(mode) + s * s
        return mu, s

    # ------------------------------------------------------------------
    # Broadcasting helpers
    # ------------------------------------------------------------------

    def _broadcast(self, x):
        x = np.asarray(x, float)
        xb, mub, sb = np.broadcast_arrays(x, self._mu, self._s)
        sb = np.maximum(sb, np.finfo(float).tiny)
        return xb, mub, sb

    @staticmethod
    def _log_pos(x):
        """
        Safe log(x) that does not warn on x<=0:
        returns log(x) for x>0, -inf otherwise.
        """
        x = np.asarray(x, dtype=np.float64)
        out = np.full_like(x, -np.inf, dtype=np.float64)
        m = x > 0
        out[m] = np.log(x[m])
        return out

    # ------------------------------------------------------------------
    # Base lognormal (untruncated)
    # ------------------------------------------------------------------

    def _base_logpdf(self, x):
        xb, mu, s = self._broadcast(x)
        logx = self._log_pos(xb)
        z = (logx - mu) / s
        out = -0.5 * z * z - logx - np.log(s) - self._LOG_SQRT2PI
        return np.where(xb > 0, out, -np.inf)

    def _base_logcdf(self, x):
        xb, mu, s = self._broadcast(x)
        logx = self._log_pos(xb)
        return log_ndtr((logx - mu) / s)

    def _base_ppf_from_logp(self, logp):
        logp, mu, s = np.broadcast_arrays(logp, self._mu, self._s)
        z = ndtri_exp(logp)
        return np.exp(mu + s * z)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def logpdf(self, x):
        base = self._base_logpdf(x)
        out = base - self._logZ
        return np.where((x >= 0) & (x <= self.upper), out, -np.inf)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        x = np.asarray(x, float)
        base = self._base_logcdf(np.minimum(x, self.upper))
        out = base - self._logZ
        out = np.where(x <= 0, -np.inf, out)
        out = np.where(x >= self.upper, 0.0, out)
        return out

    def cdf(self, x):
        return np.exp(self.logcdf(x))

    def ppf(self, q):
        q = np.clip(q, 0.0, 1.0)
        logq = np.full_like(q, -np.inf, dtype=np.float64)
        m = q > 0
        logq[m] = np.log(q[m])
        return self._base_ppf_from_logp(logq + self._logZ)

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.random(size))

    def stats(self):
        s = np.maximum(self._s, np.finfo(float).tiny)
        mu = self._mu

        if np.isinf(self.upper):
            mean, _, var = self._trunc_moments_logspace(mu, s, upper_is_finite=False)
            return mean, var

        mean, _, var = self._trunc_moments_logspace(mu, s, upper_is_finite=True)
        return mean, var

    def mean(self):
        """
        Returns the mean of the (possibly truncated) lognormal distribution.
        Broadcasting-safe and numerically stable (log-space moments).
        """
        s = np.maximum(self._s, np.finfo(float).tiny)
        mu = self._mu

        if np.isinf(self.upper):
            mean, _, _ = self._trunc_moments_logspace(mu, s, upper_is_finite=False)
            return mean

        mean, _, _ = self._trunc_moments_logspace(mu, s, upper_is_finite=True)
        return mean

    def std(self):
        """
        Returns the standard deviation of the truncated distribution.
        """
        _, var = self.stats()
        return np.sqrt(var)


class TruncatedLognormal:
    """
    Simple, fast truncated lognormal distribution.

    Parameterization:
      - location : exp(mu), i.e. median of the underlying lognormal
      - noise    : sigma of log(X) (NOT the std of X)

    Lower bound = 0
    Upper bound = 1 or +inf

    No iteration, no mode preservation, no moment matching.
    Fully vectorized and broadcasting-safe.
    """

    _LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)

    def __init__(self, median_untrunc, noise, upper=1):
        self.location = np.asarray(median_untrunc, float)
        self.noise = np.asarray(noise, float)
        self.upper = float(upper)

        if not (np.isinf(self.upper) or np.isclose(self.upper, 1.0)):
            raise ValueError("upper must be 1.0 or np.inf")

        self._mu = np.log(np.maximum(self.location, np.finfo(float).tiny))
        self._s = np.maximum(self.noise, np.finfo(float).tiny)

        # Normalization constant (always defined)
        if np.isinf(self.upper):
            self._logZ = np.zeros_like(self._mu)
        else:
            self._logZ = log_ndtr((-self._mu) / self._s)

    # ------------------------------------------------------------------
    # Broadcasting helper
    # ------------------------------------------------------------------

    def _broadcast(self, x):
        x = np.asarray(x, float)
        xb, mu, s = np.broadcast_arrays(x, self._mu, self._s)
        s = np.maximum(s, np.finfo(float).tiny)
        return xb, mu, s

    @staticmethod
    def _log_pos(x):
        """
        Safe log(x) that does not warn on x<=0:
        returns log(x) for x>0, -inf otherwise.
        """
        x = np.asarray(x, dtype=np.float64)
        out = np.full_like(x, -np.inf, dtype=np.float64)
        m = x > 0
        out[m] = np.log(x[m])
        return out

    # ------------------------------------------------------------------
    # Base (untruncated) lognormal
    # ------------------------------------------------------------------

    def _base_logpdf(self, x):
        xb, mu, s = self._broadcast(x)
        logx = self._log_pos(xb)
        z = (logx - mu) / s
        out = -0.5 * z * z - logx - np.log(s) - self._LOG_SQRT2PI
        return np.where(xb > 0.0, out, -np.inf)

    def _base_logcdf(self, x):
        xb, mu, s = self._broadcast(x)
        logx = self._log_pos(xb)
        return log_ndtr((logx - mu) / s)

    def _base_ppf_from_logp(self, logp):
        logp, mu, s = np.broadcast_arrays(logp, self._mu, self._s)
        z = ndtri_exp(logp)
        return np.exp(mu + s * z)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def logpdf(self, x):
        base = self._base_logpdf(x)
        out = base - self._logZ
        return np.where((x >= 0.0) & (x <= self.upper), out, -np.inf)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def logcdf(self, x):
        x = np.asarray(x, float)
        base = self._base_logcdf(np.minimum(x, self.upper))
        out = base - self._logZ
        out = np.where(x <= 0.0, -np.inf, out)
        out = np.where(x >= self.upper, 0.0, out)
        return out

    def cdf(self, x):
        return np.exp(self.logcdf(x))

    def ppf(self, q):
        q = np.clip(q, 0.0, 1.0)
        logq = np.full_like(q, -np.inf, dtype=np.float64)
        m = q > 0
        logq[m] = np.log(q[m])
        return self._base_ppf_from_logp(logq + self._logZ)

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.random(size))

    def stats(self):
        """
        Returns (mean, var) of the truncated distribution.
        Numerically stable for large s via log-domain evaluation.
        """
        mu = self._mu
        s = self._s

        # Untruncated moments (also do in log-space to be safe)
        if np.isinf(self.upper):
            log_mean = mu + 0.5 * s * s
            log_m2   = 2.0 * mu + 2.0 * s * s

            # exp(log_m2) - exp(2*log_mean) in a stable way
            # var = exp(log_m2) * (1 - exp(2*log_mean - log_m2))
            # Note: 2*log_mean - log_m2 = -s^2 <= 0
            mean = np.exp(np.clip(log_mean, -745.0, 709.0))
            var = np.exp(np.clip(log_m2, -745.0, 709.0)) * (1.0 - np.exp(-s * s))
            var = np.maximum(var, 0.0)
            return mean, var

        # Truncated to [0,1]
        z0 = (-mu) / s
        z1 = (-mu - s * s) / s
        z2 = (-mu - 2.0 * s * s) / s

        logZ  = log_ndtr(z0)
        logF1 = log_ndtr(z1)
        logF2 = log_ndtr(z2)

        # log r1 = logF1 - logZ, log r2 = logF2 - logZ
        log_r1 = logF1 - logZ
        log_r2 = logF2 - logZ

        # log(mean) = mu + 0.5*s^2 + log_r1
        log_mean = mu + 0.5 * s * s + log_r1

        # log(m2) = 2*mu + 2*s^2 + log_r2
        log_m2 = 2.0 * mu + 2.0 * s * s + log_r2

        # mean and m2 in floating space (clipped to avoid exp overflow)
        mean = np.exp(np.clip(log_mean, -745.0, 709.0))
        m2   = np.exp(np.clip(log_m2,   -745.0, 709.0))

        # var = m2 - mean^2; do it in a way that avoids catastrophic cancellation
        # var = exp(log_m2) * (1 - exp(2*log_mean - log_m2))
        t = 2.0 * log_mean - log_m2  # can be very negative; <= 0 when m2 >= mean^2
        # If due to rounding t>0 slightly, clamp to 0 so (1-exp(t)) doesn't go negative.
        t = np.minimum(t, 0.0)

        var = m2 * (-np.expm1(t))  # 1 - exp(t) computed stably
        var = np.maximum(var, 0.0)
        return mean, var

    def std(self):
        """
        Returns the standard deviation of the truncated distribution.
        """
        _, var = self.stats()
        return np.sqrt(var)

    def mean(self):
        """
        Returns the mean of the truncated distribution.

        Uses the same numerically stable log-domain evaluation as `stats()`,
        but computes only the first moment (faster than calling stats()).
        """
        mu = self._mu
        s = self._s

        # Untruncated lognormal mean: exp(mu + 0.5*s^2)
        if np.isinf(self.upper):
            log_mean = mu + 0.5 * s * s
            return np.exp(np.clip(log_mean, -745.0, 709.0))

        # Truncated to [0, 1]:
        # E[X | X<=1] = exp(mu + 0.5*s^2) * Phi((-mu - s^2)/s) / Phi((-mu)/s)
        z0 = (-mu) / s
        z1 = (-mu - s * s) / s

        logZ  = log_ndtr(z0)
        logF1 = log_ndtr(z1)

        log_mean = mu + 0.5 * s * s + (logF1 - logZ)
        return np.exp(np.clip(log_mean, -745.0, 709.0))


class TruncatedLognormalMean:
    """
    Fast truncated lognormal with cheap affine mean correction.

    Guarantees:
      - location ≈ mean of the TRUNCATED distribution
      - noise = sigma of log(X)
      - lower bound = 0
      - upper bound = 1 or +inf

    No iteration, fully vectorized, broadcasting-safe.
    """

    _LOG_SQRT2PI = 0.5 * np.log(2.0 * np.pi)

    def __init__(self, mean, noise, upper=np.inf):
        self.location = np.asarray(mean, float)
        self.noise = np.asarray(noise, float)
        self.upper = float(upper)

        if not (np.isinf(self.upper) or np.isclose(self.upper, 1.0)):
            raise ValueError("upper must be 1.0 or np.inf")

        s = np.maximum(self.noise, np.finfo(float).tiny)
        loc = np.maximum(self.location, np.finfo(float).tiny)

        # --- base mu (untruncated mean would be location) ---
        mu = np.log(loc) - 0.5 * s * s

        if np.isinf(self.upper):
            # No truncation → exact
            self._mu = mu
            self._s = s
            self._logZ = np.zeros_like(mu)
        else:
            # --- cheap affine correction ---
            z0 = (-mu) / s
            z1 = (-mu - s * s) / s

            logZ = log_ndtr(z0)
            logR = log_ndtr(z1) - logZ

            self._mu = mu - logR
            self._s = s
            self._logZ = log_ndtr((-self._mu) / self._s)

    # ------------------------------------------------------------------
    # Broadcasting helper
    # ------------------------------------------------------------------

    def _broadcast(self, x):
        x = np.asarray(x, float)
        xb, mu, s = np.broadcast_arrays(x, self._mu, self._s)
        s = np.maximum(s, np.finfo(float).tiny)
        return xb, mu, s

    @staticmethod
    def _log_pos(x):
        """
        Safe log(x) that does not warn on x<=0:
        returns log(x) for x>0, -inf otherwise.
        """
        x = np.asarray(x, dtype=np.float64)
        out = np.full_like(x, -np.inf, dtype=np.float64)
        m = x > 0
        out[m] = np.log(x[m])
        return out

    # ------------------------------------------------------------------
    # Base (untruncated) lognormal
    # ------------------------------------------------------------------

    def _base_logpdf(self, x):
        xb, mu, s = self._broadcast(x)
        logx = self._log_pos(xb)
        z = (logx - mu) / s
        out = -0.5 * z * z - logx - np.log(s) - self._LOG_SQRT2PI
        return np.where(xb > 0.0, out, -np.inf)

    def _base_logcdf(self, x):
        xb, mu, s = self._broadcast(x)
        logx = self._log_pos(xb)
        return log_ndtr((logx - mu) / s)

    def _base_ppf_from_logp(self, logp):
        logp, mu, s = np.broadcast_arrays(logp, self._mu, self._s)
        z = ndtri_exp(logp)
        return np.exp(mu + s * z)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def logpdf(self, x):
        base = self._base_logpdf(x)
        out = base - self._logZ
        return np.where((x >= 0.0) & (x <= self.upper), out, -np.inf)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def pdf(self, x):
        logp = self.logpdf(x)
        return np.exp(np.clip(logp, -745.0, 709.0))

    def logcdf(self, x):
        x = np.asarray(x, float)
        base = self._base_logcdf(np.minimum(x, self.upper))
        out = base - self._logZ
        out = np.where(x <= 0.0, -np.inf, out)
        out = np.where(x >= self.upper, 0.0, out)
        return out

    def cdf(self, x):
        return np.exp(self.logcdf(x))

    def ppf(self, q):
        q = np.clip(q, 0.0, 1.0)
        logq = np.full_like(q, -np.inf, dtype=np.float64)
        m = q > 0
        logq[m] = np.log(q[m])
        return self._base_ppf_from_logp(logq + self._logZ)

    def rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        return self.ppf(rng.random(size))

    def stats(self):
        """
        Returns (mean, var) of the (possibly truncated) distribution.
        Numerically stable for large s using log-domain evaluation.
        """
        s = np.maximum(self._s, np.finfo(np.float64).tiny)
        mu = self._mu

        finfo = np.finfo(np.float64)
        maxlog = np.log(finfo.max)  # ~709.78

        if np.isinf(self.upper):
            # log moments
            log_mean = mu + 0.5 * s * s
            log_m2   = 2.0 * mu + 2.0 * s * s

            mean = np.exp(np.clip(log_mean, -np.inf, maxlog))

            # var = exp(log_m2) - exp(2*log_mean)
            # but 2*log_mean - log_m2 = -s^2 <= 0, so stable:
            m2 = np.exp(np.clip(log_m2, -np.inf, maxlog))
            var = m2 * (1.0 - np.exp(-s * s))
            var = np.maximum(var, 0.0)
            return mean, var

        # Truncated to [0, 1]
        z0 = (-mu) / s
        z1 = (-mu - s * s) / s
        z2 = (-mu - 2.0 * s * s) / s

        logZ  = log_ndtr(z0)
        logF1 = log_ndtr(z1)
        logF2 = log_ndtr(z2)

        # log r1, log r2
        log_r1 = logF1 - logZ
        log_r2 = logF2 - logZ

        # log moments under truncation
        log_mean = mu + 0.5 * s * s + log_r1
        log_m2   = 2.0 * mu + 2.0 * s * s + log_r2

        mean = np.exp(np.clip(log_mean, -np.inf, maxlog))
        m2   = np.exp(np.clip(log_m2,   -np.inf, maxlog))

        # var = m2 - mean^2, computed stably:
        # var = exp(log_m2) * (1 - exp(2*log_mean - log_m2))
        t = 2.0 * log_mean - log_m2
        t = np.minimum(t, 0.0)          # clamp tiny positive roundoff
        var = m2 * (-np.expm1(t))        # = m2*(1-exp(t)) stably
        var = np.maximum(var, 0.0)

        return mean, var

    def std(self):
        """
        Returns the standard deviation of the truncated distribution.
        """
        _, var = self.stats()
        return np.sqrt(var)

    def mean(self):
        """
        Returns the mean of the (possibly truncated) distribution.

        Numerically stable for large s via log-domain evaluation.
        Computes only the first moment (cheaper than stats()).
        """
        s = np.maximum(self._s, np.finfo(np.float64).tiny)
        mu = self._mu

        finfo = np.finfo(np.float64)
        maxlog = np.log(finfo.max)  # ~709.78

        if np.isinf(self.upper):
            # Untruncated mean: exp(mu + 0.5*s^2)
            log_mean = mu + 0.5 * s * s
            return np.exp(np.clip(log_mean, -np.inf, maxlog))

        # Truncated to [0, 1]:
        # E[X | X<=1] = exp(mu + 0.5*s^2) * Phi((-mu - s^2)/s) / Phi((-mu)/s)
        z0 = (-mu) / s
        z1 = (-mu - s * s) / s

        logZ  = log_ndtr(z0)
        logF1 = log_ndtr(z1)

        log_mean = mu + 0.5 * s * s + (logF1 - logZ)
        return np.exp(np.clip(log_mean, -np.inf, maxlog))