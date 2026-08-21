# References

## Papers
- **Kristiadi, Hein, Hennig (2020)** — *Being Bayesian, Even Just a Bit, Fixes
  Overconfidence in ReLU Networks*, ICML. The paper reimplemented here.
  <https://arxiv.org/abs/2002.10118> · local: `literature/Kristiadi et al - 2020 - Being
  Bayesian, Even Just a Bit, Fixed Overconfidence in ReLU Networks.pdf`
- **Ritter, Botev, Barber (2018)** — *A Scalable Laplace Approximation for Neural
  Networks*, ICLR. Source of the prior-precision tuning idea — [Q1](open-questions.md#q1).
  <https://openreview.net/forum?id=Skdvd2xAZ> · local: `literature/Ritter et al - 2018 -
  A Scalable Laplace Approximation for Neural Networks.pdf`
- **MacKay (1992)** — *The Evidence Framework Applied to Classification Networks*.
  Origin of the probit approximation `sigmoid(m/sqrt(1+πv/8))` used in
  `Model.predict` — [I5](gotchas/hessian-scaling.md).
  <https://doi.org/10.1162/neco.1992.4.5.720>
- **Daxberger et al. (2021)** — *Laplace Redux — Effortless Bayesian Deep Learning*,
  NeurIPS. The `laplace-torch` paper; survey of Laplace variants — [Q6](open-questions.md#q6).
  <https://arxiv.org/abs/2106.14806>

> `literature/` holds PDFs copied from the UQHM vault (`06_Literature`). It is
> **gitignored** — paths are local-only, the files are not in the repo.

## Code
- `laplace-torch` (recommended over this repo, per its README):
  <https://github.com/AlexImmer/Laplace>
- BackPack (used by the paper's code for curvature): <https://github.com/f-dangel/backpack>

## API docs
- Pyro distributions (`Normal(loc, scale)` — scale is a **std**):
  <https://docs.pyro.ai/en/stable/distributions.html>
- Pyro autoguides + `init_to_value`:
  <https://docs.pyro.ai/en/stable/infer.autoguide.html>
- Pyro SVI / `Trace_ELBO`: <https://docs.pyro.ai/en/stable/inference_algos.html>
- Pyro MCMC / NUTS: <https://docs.pyro.ai/en/stable/mcmc.html>
- Pyro `Predictive`: <https://docs.pyro.ai/en/stable/inference_algos.html#pyro.infer.predictive.Predictive>
- PyTorch `BatchNorm1d` (train/eval semantics): <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html>
- PyTorch `nn.Linear` default init (fan-in rule): <https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html>
