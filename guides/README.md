# How-To Guides

Each guide shows how to use the toolkit's core functions on one small example with generated data.
A guide is a page, `README.md`, and a script, `guide.m`, that prints the numbers and draws the
figures on the page. The guides come in series, and within a series each guide builds on the one
before.

## Getting Started with the Precision Sampler

The three guides use the local level model on the same generated data.

| Guide | What it shows | Core functions |
|---|---|---|
| [How to Draw the States of an Unobserved Components Model](uc_states/) | Stack the equations of the model, read the precision matrix of the states off the log density, and draw the whole path of the states at once | `ssm.simulate_states`, `ssm.diffmat` |
| [How to Estimate a State Space Model by Gibbs Sampling](gibbs_sampler/) | Draw the parameters and the states in turn, and check the chain | `ssm.simulate_states`, `ssm.diffmat`, `ssm.inefficiency_factor` |
| [How to Compute the Integrated Likelihood of a State Space Model](integrated_likelihood/) | Integrate the states out of the likelihood, and draw the parameters without them | `ssm.intlike`, `ssm.simulate_states`, `ssm.inefficiency_factor` |

The [examples](../examples/) run the core functions on further models, with real and generated
data, and the [tutorials](../tutorials/) answer empirical questions with the models of published
papers.
