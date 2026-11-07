# How to Add Student-t Errors to a Stochastic Volatility Model

*Code: [`guide.m`](guide.m), with [`ssm.ksc_ar1_mean`](../../core/+ssm/ksc_ar1_mean.m),
[`ssm.mode_newton`](../../core/+ssm/mode_newton.m),
[`ssm.inefficiency_factor`](../../core/+ssm/inefficiency_factor.m) and
[`ssm.armh`](../../core/+ssm/armh.m). Method:
[Chan and Grant (2016)](../../CITING.md#chan-and-grant-2016b),
[Kim, Shephard and Chib (1998)](../../CITING.md#kim-shephard-and-chib-1998) and
[Chan (2027)](../../CITING.md#chan-2027).*

In the stochastic volatility (SV) model of the first two guides, the errors are normal, so a large
observation can only come from a high volatility. With Student-$`t`$ errors, it can also come from
the tail of the error distribution. We add Student-$`t`$ errors to the sampler of
[How to Estimate a Stochastic Volatility Model by Gibbs Sampling](../sv_gibbs/). The new model,
SV-$`t`$, needs two new blocks: a latent scale for each period and the degrees of freedom. On data
with Student-$`t`$ errors, SV-$`t`$ estimates the volatility more accurately than the SV model.

To draw the figures first, run from the root of the repository:

```matlab
run guides/sv_t/guide.m
```

It takes about 30 seconds on the machine that produced this page.

## The SV-t Model

The model is

```math
y_t = \mu + \sqrt{\lambda_t}\,\mathrm{e}^{h_t/2}\varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, 1), \qquad \lambda_t \sim \mathcal{IG}(\nu/2, \nu/2),
```

with the log-volatility $`h_t`$ of the first guide. Given $`h_t`$, the error
$`(y_t - \mu)\mathrm{e}^{-h_t/2}`$ is a normal variable whose variance $`\lambda_t`$ is drawn from
$`\mathcal{IG}(\nu/2, \nu/2)`$. Its distribution is therefore the Student-$`t`$ distribution
$`\mathcal{T}_\nu`$ with $`\nu`$ degrees of freedom. A small $`\nu`$ gives heavy tails, and as
$`\nu`$ grows the model approaches the SV model. Writing the errors as a scale mixture of normals
keeps the model Gaussian given $`\boldsymbol{\lambda} = (\lambda_1, \ldots, \lambda_T)'`$.

The priors of $`\mu`$, $`\mu_h`$, $`\phi`$ and $`\sigma_h^2`$ are those of the second guide. As in
Chan and Grant (2016), the prior of $`\nu`$ is uniform, $`\nu \sim \mathcal{U}(2, 100)`$. We
generate the data from the log-volatilities and the errors $`\varepsilon_t`$ of the first guide,
with $`\lambda_t`$ drawn for $`\nu = 5`$. The volatility path is therefore the same as in the first
two guides, and only the tails of the errors differ.

## Step 1: Add Two Blocks

The sampler of the second guide cycles through five blocks. SV-$`t`$ keeps them and adds the
scales $`\boldsymbol{\lambda}`$ and the degrees of freedom $`\nu`$:

1. $`(\mu \mid \mathbf{y}, \mathbf{h}, \boldsymbol{\lambda})`$, a normal distribution;
2. the indicators $`\mathbf{s}`$ and then $`\mathbf{h}`$, by the mixture step of the first guide;
3. $`(\boldsymbol{\lambda} \mid \mathbf{y}, \mu, \mathbf{h}, \nu)`$, independent inverse-gamma
   distributions;
4. $`(\nu \mid \boldsymbol{\lambda})`$, by a Metropolis-Hastings step;
5. $`\sigma_h^2`$, $`\phi`$ and $`\mu_h`$ given $`\mathbf{h}`$, as in the second guide.

The parameters of the log-volatility depend only on $`\mathbf{h}`$, so their blocks do not change.

## Step 2: Derive the New Conditional Distributions

**The mean and the log-volatilities.** Given $`\boldsymbol{\lambda}`$, the variance of $`y_t`$
given $`h_t`$ is $`\lambda_t\mathrm{e}^{h_t}`$. The mean is again a regression on a constant with
known variances:

```math
(\mu \mid \mathbf{y}, \mathbf{h}, \boldsymbol{\lambda}) \sim \mathcal{N}\big(K_\mu^{-1}c_\mu,\; K_\mu^{-1}\big), \qquad K_\mu = \frac{1}{V_\mu} + \sum_{t=1}^{T}\frac{1}{\lambda_t\mathrm{e}^{h_t}}, \qquad c_\mu = \frac{\mu_0}{V_\mu} + \sum_{t=1}^{T}\frac{y_t}{\lambda_t\mathrm{e}^{h_t}}.
```

The rescaled observation $`(y_t - \mu)/\sqrt{\lambda_t} = \mathrm{e}^{h_t/2}\varepsilon_t`$ follows
the SV model of the first guide. The mixture step therefore applies to
$`y_t^* = \log\big((y_t - \mu)^2/\lambda_t + c\big)`$.

**The scales.** Given $`y_t`$, $`\mu`$, $`h_t`$ and $`\nu`$, each $`\lambda_t`$ depends on its own
period only, and the inverse-gamma prior is conjugate:

```math
(\lambda_t \mid y_t, \mu, h_t, \nu) \sim \mathcal{IG}\Big(\frac{\nu + 1}{2},\; \frac{\nu + (y_t - \mu)^2\mathrm{e}^{-h_t}}{2}\Big).
```

A large squared error relative to $`\mathrm{e}^{h_t}`$ gives a large $`\lambda_t`$.

**The degrees of freedom.** Given $`\boldsymbol{\lambda}`$, the log density of $`\nu`$ is

```math
\log p(\nu \mid \boldsymbol{\lambda}) = T\Big(\frac{\nu}{2}\log\frac{\nu}{2} - \log\Gamma\Big(\frac{\nu}{2}\Big)\Big) - \Big(\frac{\nu}{2} + 1\Big)\sum_{t=1}^{T}\log\lambda_t - \frac{\nu}{2}\sum_{t=1}^{T}\frac{1}{\lambda_t} + \text{const}
```

on $`(2, 100)`$. It has no standard form. A Metropolis-Hastings step proposes
$`\nu^* \sim \mathcal{N}(\hat{\nu}, K_\nu^{-1})`$, where $`\hat{\nu}`$ is the mode and $`K_\nu`$ the
negative second derivative there. Its gradient and negative second derivative are

```math
\frac{T}{2}\Big(\log\frac{\nu}{2} + 1 - \psi\Big(\frac{\nu}{2}\Big)\Big) - \frac{1}{2}\sum_{t=1}^{T}\Big(\log\lambda_t + \frac{1}{\lambda_t}\Big) \qquad\text{and}\qquad \frac{T}{4}\psi'\Big(\frac{\nu}{2}\Big) - \frac{T}{2\nu},
```

where $`\psi`$ is the digamma function. [`ssm.mode_newton`](../../core/+ssm/mode_newton.m) finds
the mode by Newton's method. The candidate is accepted with probability

```math
\min\Big\{1,\; \frac{p(\nu^* \mid \boldsymbol{\lambda})\, f_{\mathcal{N}}(\nu;\, \hat{\nu}, K_\nu^{-1})}{p(\nu \mid \boldsymbol{\lambda})\, f_{\mathcal{N}}(\nu^*;\, \hat{\nu}, K_\nu^{-1})}\Big\},
```

and rejected outside $`(2, 100)`$.

## Step 3: Write the Loop

```matlab
mu0 = 0; Vmu = 10;                                  % mu ~ N(mu0, Vmu)
muh0 = 1; Vmuh = 10;                                % mu_h ~ N(muh0, Vmuh)
phi0 = .97; Vphi = .1^2;                            % phi ~ TN(phi0, Vphi) on (-1, 1)
nuh = 5; Sh = .16;                                  % sigma_h^2 ~ IG(nuh, Sh)
nuub = 100;                                         % nu ~ U(2, nuub)
offset = 1e-4;
nsim = 20000; burnin = 1000;
muh = log(var(y)); phi = .98; sig2h = .04; nu = 5;  % starting values
h = muh*ones(T,1); lam = ones(T,1);
store_theta = zeros(nsim, 5); nacc = 0;
store_h = zeros(T, nsim/10); store_lam = zeros(T,1);
for isim = 1:burnin + nsim
    w = 1./(lam.*exp(h));                           % 1/(lambda_t e^{h_t})
    Kmu = 1/Vmu + sum(w);                           % K_mu = 1/V_mu + sum_t 1/(lambda_t e^{h_t})
    mu = (mu0/Vmu + w'*y)/Kmu + randn/sqrt(Kmu);

    ystar = log((y - mu).^2./lam + offset);  % y*_t = log((y_t - mu)^2/lambda_t + c)
    h = ssm.ksc_ar1_mean(ystar, h, muh, phi, sig2h);

    e2 = (y - mu).^2.*exp(-h);  % lambda_t ~ IG((nu + 1)/2, (nu + e2_t)/2)
    lam = 1./gamrnd((nu + 1)/2, 2./(nu + e2));

    s1 = sum(log(lam)); s2 = sum(1./lam);  % nu: log p(lambda | nu), its mode and curvature
    fnu = @(x) T*(x/2*log(x/2) - gammaln(x/2)) - (x/2 + 1)*s1 - x/2*s2;
    gradK = @(x) deal(T/2*(log(x/2) + 1 - psi(x/2)) - (s1 + s2)/2, T/4*psi(1, x/2) - T/(2*x));
    [nuhat, Knu] = ssm.mode_newton(nu, gradK);
    nuc = nuhat + randn/sqrt(Knu);         % the proposal N(nuhat, 1/K_nu)
    if nuc > 2 && nuc < nuub && fnu(nuc) - fnu(nu) ...
            + .5*Knu*((nuc - nuhat)^2 - (nu - nuhat)^2) > log(rand)
        nu = nuc;
        nacc = nacc + (isim > burnin);
    end

    hc = h - muh;  % sigma_h^2, phi and mu_h: guides/sv_gibbs
    u = [sqrt(1 - phi^2)*hc(1); hc(2:T) - phi*hc(1:T-1)];
    sig2h = 1/gamrnd(nuh + T/2, 1/(Sh + u'*u/2));

    Kphi = 1/Vphi + hc(1:T-1)'*hc(1:T-1)/sig2h;
    phihat = (phi0/Vphi + hc(1:T-1)'*hc(2:T)/sig2h)/Kphi;
    phic = 2;
    while abs(phic) >= 1, phic = phihat + randn/sqrt(Kphi); end
    g = @(x) .5*log(1 - x^2) - .5*(1 - x^2)*hc(1)^2/sig2h;
    if g(phic) - g(phi) > log(rand), phi = phic; end

    Kmuh = 1/Vmuh + (1 - phi^2 + (T-1)*(1 - phi)^2)/sig2h;
    cmuh = muh0/Vmuh + ((1 - phi^2)*h(1) + (1 - phi)*sum(h(2:T) - phi*h(1:T-1)))/sig2h;
    muh = cmuh/Kmuh + randn/sqrt(Kmuh);

    if isim > burnin
        store_theta(isim - burnin, :) = [mu muh phi sig2h nu];
        store_lam = store_lam + lam/nsim;
        if mod(isim - burnin, 10) == 0, store_h(:, (isim - burnin)/10) = h; end
    end
end
```

The blocks of $`\sigma_h^2`$, $`\phi`$ and $`\mu_h`$ are those of the second guide. The chain
starts from $`\boldsymbol{\lambda} = \mathbf{1}`$ and $`\nu = 5`$, and from the starting values of
the second guide for the other blocks.

## Step 4: See What the Student-t Errors Change

The table compares SV-$`t`$ with the SV model of the second guide on the same data.

| | True value | SV-$`t`$: posterior mean | 90% interval | Inefficiency factor | SV: posterior mean | 90% interval |
|---|---|---|---|---|---|---|
| $`\mu`$ | 0.1 | 0.1200 | (0.0708, 0.1698) | 1.4 | 0.1149 | (0.0646, 0.1648) |
| $`\mu_h`$ | 0 | -0.0250 | (-0.5230, 0.4857) | 4.4 | 0.2759 | (-0.0692, 0.6331) |
| $`\phi`$ | 0.97 | 0.9736 | (0.9545, 0.9895) | 32.7 | 0.9397 | (0.9030, 0.9698) |
| $`\sigma_h^2`$ | 0.04 | 0.0472 | (0.0276, 0.0757) | 79.3 | 0.1375 | (0.0739, 0.2188) |
| $`\nu`$ | 5 | 5.61 | (3.96, 8.10) | 74.1 | | |

We note three results. First, the posterior of $`\nu`$ covers its true value of 5: its median is
5.33 and its 90 percent interval is (3.96, 8.10). Second, under the SV model the outliers raise
the estimated volatility. Its posterior mean of $`\sigma_h^2`$ is 3.4 times the true value, and
the true value lies below its 90 percent interval. Its posterior mean of $`\phi`$ falls to 0.94.
Under SV-$`t`$ both true values lie inside the intervals. Third, the volatility estimated by
SV-$`t`$ is closer to the truth (Figure 1). The root mean squared error of the posterior mean of
$`\mathrm{e}^{h_t/2}`$ is 0.247, against 0.428 under SV. The 90 percent intervals of SV-$`t`$
contain the true volatility in 929 of the 1,000 periods.

The five periods with the largest $`\lambda_t`$ show the mechanism:

| Period | True $`\lambda_t`$ | Posterior mean of $`\lambda_t`$ | True $`\lvert\varepsilon_t\rvert`$ | True $`\mathrm{e}^{h_t/2}`$ | SV-$`t`$ | SV |
|---|---|---|---|---|---|---|
| 396 | 18.08 | 13.30 | 1.75 | 1.50 | 1.75 | 3.70 |
| 664 | 16.00 | 1.32 | 0.21 | 0.54 | 0.71 | 0.79 |
| 668 | 11.52 | 1.25 | 0.03 | 1.09 | 0.77 | 0.99 |
| 158 | 10.89 | 7.18 | 1.69 | 0.86 | 1.06 | 1.94 |
| 118 | 10.74 | 2.12 | 0.58 | 0.88 | 0.91 | 1.22 |

In period 396, the error $`\sqrt{\lambda_t}\varepsilon_t`$ is 7.5 times the volatility. SV puts
the volatility at 3.70, about 2.5 times its true value of 1.50, while SV-$`t`$ puts it at 1.75 and
gives $`\lambda_t`$ a posterior mean of 13.30. A large $`\lambda_t`$ produces a large observation
only when $`\varepsilon_t`$ is not small. In periods 664 and 668, $`|\varepsilon_t|`$ is 0.21 and
0.03, so the data carry little information about $`\lambda_t`$, and its posterior mean stays near
1.

![The data and the volatility under SV-t and SV](fig_vol.png)

*Figure 1: SV-$`t`$ and SV on data generated with Student-$`t`$ errors, $`\nu = 5`$, and the
log-volatilities of the first guide. Top, the data. Bottom, the true volatility
$`\mathrm{e}^{h_t/2}`$, its posterior means under SV-$`t`$ and SV, and the 90 percent pointwise
credible intervals of SV-$`t`$.*

![The draws of nu and their histogram](fig_nu.png)

*Figure 2: The 20,000 draws of $`\nu`$ kept after the burn-in, and their histogram.*

The draws of $`\phi`$, $`\sigma_h^2`$ and $`\nu`$ are the most autocorrelated. With Bartlett
weights up to lag 400, their inefficiency factors are 32.7, 79.3 and 74.1. The 20,000 draws thus
carry the information of about 250 to 610 independent draws. Geweke's Z with $`L = 400`$ is 1.48,
1.04, $`-1.32`$, 1.51 and $`-0.23`$ for the five parameters, all inside $`\pm 1.96`$. The
candidate $`\nu^*`$ is accepted in 98 percent of the iterations.

## How We Check It

<details>
<summary>The posterior from a second sampler, which draws the log-volatilities from their exact
conditional distribution</summary>

The sampler of Chan and Grant (2016) for SV-$`t`$ has the same blocks, except for $`\mathbf{h}`$.
It draws $`\mathbf{h}`$ from its exact conditional distribution given $`\boldsymbol{\lambda}`$, by
the accept-reject Metropolis-Hastings step of
[How to Draw the Log-Volatilities of a Stochastic Volatility Model](../sv_states/). `guide.m` runs
it on the same data with the same priors, for 10,000 draws after a burn-in of 1,000, starting the
log-volatilities at the mode. Its Metropolis-Hastings step takes the candidate path in 62.7
percent of the iterations.

The posterior means of the two samplers differ by $`-0.9`$, $`-1.4`$, $`-1.5`$, 1.1 and $`-0.7`$
Monte Carlo standard errors for $`\mu`$, $`\mu_h`$, $`\phi`$, $`\sigma_h^2`$ and $`\nu`$.

</details>

## Adapt It Yourself

Student-$`t`$ errors give every period a scale of its own. Some models let only a few periods
have outliers: $`\lambda_t = \kappa`$ with probability $`p`$ and $`\lambda_t = 1`$ otherwise, with a
known $`\kappa > 1`$ and the prior $`p \sim \mathcal{B}(a_p, b_p)`$. Which blocks change?

<details>
<summary>Answer</summary>

The block of $`\nu`$ goes, $`p`$ gets a block of its own, and the scales become two-point draws. Let
$`e_t^2 = (y_t - \mu)^2\mathrm{e}^{-h_t}`$. Given $`y_t`$, $`\mu`$, $`h_t`$ and $`p`$, the
probability that period $`t`$ has an outlier is

$`\displaystyle \Pr(\lambda_t = \kappa \mid y_t, \mu, h_t, p) = \frac{p\,\kappa^{-1/2}\exp(-e_t^2/(2\kappa))}{p\,\kappa^{-1/2}\exp(-e_t^2/(2\kappa)) + (1 - p)\exp(-e_t^2/2)},`$

the ratio of the two normal densities of $`y_t`$, with variances $`\kappa\mathrm{e}^{h_t}`$ and
$`\mathrm{e}^{h_t}`$. Given the scales, the number of outliers $`n_{\text{out}}`$ is a binomial
draw, so the beta prior is conjugate:
$`(p \mid \boldsymbol{\lambda}) \sim \mathcal{B}(a_p + n_{\text{out}}, b_p + T - n_{\text{out}})`$.
With the settings

```matlab
kap = 10;         % lambda_t = kap with probability p, else 1
ap = 1; bp = 19;  % p ~ B(ap, bp)
```

these lines replace the blocks of $`\boldsymbol{\lambda}`$ and $`\nu`$ in the loop:

```matlab
    e2 = (y - mu).^2.*exp(-h);              % Pr(lambda_t = kap) given y_t, mu, h_t, p
    a1 = p/sqrt(kap)*exp(-e2/(2*kap)); a0 = (1 - p)*exp(-e2/2);
    lam = 1 + (kap - 1)*(rand(T,1) < a1./(a1 + a0));
    nout = nnz(lam > 1);
    p = betarnd(ap + nout, bp + T - nout);  % p ~ B(ap + n_out, bp + T - n_out)
```

`guide.m` runs this sampler on the data of this guide, with $`\kappa = 10`$ and
$`p \sim \mathcal{B}(1, 19)`$. It checks the outlier probability against the two normal densities
it comes from, and the two agree to within 4e-16. The posterior mean of $`p`$ is 0.059, with 90
percent interval (0.025, 0.103). Eighteen periods have an outlier probability above 0.5, and seven
of them are among the ten largest true errors $`\sqrt{\lambda_t}\varepsilon_t`$. The posterior
mean of $`\sigma_h^2`$ is 0.0578, between 0.0472 under SV-$`t`$ and 0.1375 under SV. With
$`\kappa`$ fixed, only the largest errors enter the outlier component, and the others raise the
estimated volatility.

</details>

## Next

[How to Compare Stochastic Volatility Models by the Marginal Likelihood](../sv_compare/) compares a
constant variance, SV and SV-$`t`$ by their marginal likelihoods. Among the examples,
[ex08](../../examples/ex08_sv_ma_t.m) adds an MA(1) term to SV-$`t`$ and estimates it on daily
silver returns.

## Using This Method in Your Research

The model and its priors are those of Chan and Grant (2016). The sampler draws the log-volatilities
by the auxiliary mixture sampler of Kim, Shephard and Chib (1998), and Chan (2027) gives the
textbook treatment of the model. [`CITING.md`](../../CITING.md) lists the paper behind each
function of the toolkit and how to cite the toolkit itself.

## References

Chan, J. C. C. (2027). *Bayesian Macroeconometrics: Methods and Applications*. Chapman &
Hall/CRC. [Code repository](https://github.com/joshuaccchan/bayesian-macroeconometrics)

Chan, J. C. C. and Grant, A. L. (2016). Modeling Energy Price Dynamics: GARCH versus Stochastic
Volatility. *Energy Economics*, 54: 182-189.
[doi:10.1016/j.eneco.2015.12.003](https://doi.org/10.1016/j.eneco.2015.12.003)

Kim, S., Shephard, N. and Chib, S. (1998). Stochastic Volatility: Likelihood Inference and
Comparison with ARCH Models. *Review of Economic Studies*, 65(3): 361-393.
[doi:10.1111/1467-937X.00050](https://doi.org/10.1111/1467-937X.00050)
