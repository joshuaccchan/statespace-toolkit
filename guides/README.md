# How-To Guides

Each guide shows how to use the toolkit's core functions on one small example with generated data.
A guide is a page, `README.md`, and a script, `guide.m`, that prints the numbers and draws the
figures on the page. The guides come in series, and within a series each guide builds on the one
before.

## Getting Started with the Precision Sampler

The three guides use the local level model on the same generated data.

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Draw the States of an Unobserved Components Model](uc_states/) | Draw the whole path of the states at once with the precision sampler | `ssm.simulate_states`, `ssm.diffmat` |
| [How to Estimate a State Space Model by Gibbs Sampling](gibbs_sampler/) | Draw the parameters and the states in turn, and check the chain | `ssm.simulate_states`, `ssm.diffmat`, `ssm.inefficiency_factor` |
| [How to Compute the Integrated Likelihood of a State Space Model](integrated_likelihood/) | Integrate the states out of the likelihood, and draw the parameters without them | `ssm.intlike`, `ssm.simulate_states`, `ssm.inefficiency_factor` |

## Time-Varying Parameters

The three guides use a regression whose coefficients follow random walks, on generated data.

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Draw Time-Varying Regression Coefficients](tvp_states/) | Stack the regression over time and draw the whole path of the coefficients at once | `ssm.simulate_states`, `ssm.surform`, `ssm.diffmat` |
| [How to Estimate a TVP Regression by Gibbs Sampling](tvp_gibbs/) | Draw the variances and the initial coefficients along with the coefficients, and check the chain | `ssm.simulate_states`, `ssm.surform`, `ssm.inefficiency_factor` |
| [How to Tell Whether a Coefficient Varies over Time](tvp_compare/) | Compare models whose coefficients vary or stay constant by their marginal likelihoods | `ssm.intlike`, `ssm.surform`, `ssm.diffmat` |

## Stochastic Volatility

The four guides use a stochastic volatility model whose log-volatility follows a stationary AR(1)
process, on generated data in percent.

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Draw the Log-Volatilities of a Stochastic Volatility Model](sv_states/) | Draw the whole path of the log-volatilities by the auxiliary mixture sampler, and from their exact conditional distribution | `ssm.ksc_ar1_mean`, `ssm.simulate_states`, `ssm.armh` |
| [How to Estimate a Stochastic Volatility Model by Gibbs Sampling](sv_gibbs/) | Draw the mean and the parameters of the log-volatility along with it, and check the chain | `ssm.ksc_ar1_mean`, `ssm.inefficiency_factor`, `ssm.geweke` |
| [How to Add Student-t Errors to a Stochastic Volatility Model](sv_t/) | Give the errors heavy tails, so that outliers do not pass into the volatility | `ssm.ksc_ar1_mean`, `ssm.mode_newton` |
| [How to Compare Stochastic Volatility Models by the Marginal Likelihood](sv_compare/) | Compare constant variance, stochastic volatility and Student-t errors by their marginal likelihoods | `ssm.intlike_sv`, `ssm.intlike_sv_t` |

The [examples](../examples/) run the core functions on additional models, with real and generated
data, and the [tutorials](../tutorials/) answer empirical questions with the models of published
papers.
