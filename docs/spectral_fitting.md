# Composable likelihood fitting: conditional linearity, VarPro, and a module registry

This note describes the architecture behind `lusee.fitting` (`lusee/Fitting.py`),
its first concrete model `lusee.SpectralHealpixSky` (`lusee/SpectralSky.py`), and
the demo `notebooks/spectral_fit_demo.py` / `notebooks/spectral_fit_recovery.ipynb`.

The goal is a fitting system for LuSEE-Night where the forward model is built
from **interchangeable modules** — sky, beam, instrument — each contributing its
own free parameters and priors, and where a single set of solvers/samplers works
for any combination. We want to both **optimise** and **HMC-sample** a
`-χ²/2` likelihood that is fully `jax`-differentiable. The two failure modes to
avoid are *over*-engineering (a magic auto-composing DSL nobody can debug) and
*under*-engineering (a pile of bespoke scripts that drift apart).

The design rests on one structural fact.

## 1. The organising principle: conditional linearity

Split the parameters into non-linear `η` and linear `λ`. For LuSEE the forward
model is **linear in `λ` at fixed `η`**:

```
m(η, λ) = A(η) λ
```

Examples of `λ`: sky `a_lm`, source amplitudes, eigenmode coefficients. Examples
of `η`: spectral indices, regolith depth, antenna impedance, gains. This single
fact decides everything:

1. **Linear block → exact inner solve.** For any `η`, the optimal `λ` is the
   Wiener solution
   ```
   λ̂(η) = (Aᵀ N⁻¹ A + S⁻¹)⁻¹ Aᵀ N⁻¹ d.
   ```
   No iteration over `λ` is needed beyond one linear solve.

2. **Optimisation → variable projection (VarPro).** Profile `λ` out and optimise
   only the small non-linear block:
   ```
   χ²_prof(η) = χ²(η, λ̂(η)),
   ```
   minimised over `η` with L-BFGS / Adam. The outer problem has the dimension of
   `η` (tens–hundreds), not `λ` (thousands).

3. **Sampling → Rao–Blackwellised HMC.** The Gaussian-prior linear block can be
   marginalised analytically (a Gaussian integral), leaving a low-dimensional,
   better-conditioned posterior over `η` for NUTS. The marginal adds a
   `−½ log det(Aᵀ N⁻¹ A + S⁻¹)` term; the model and forward code are unchanged.
   *(Roadmap — not yet implemented; the optimiser ships first.)*

Throwing the whole problem at a generic PPL (numpyro/blackjax) would treat every
`a_lm` as a sampling dimension and discard this structure — the single biggest
computational win. We use a sampler library for the `η` block only and keep the
linear solve custom.

### The envelope theorem makes the VarPro gradient cheap and exact

Let `F(η, λ) = χ²_data(η, λ) + λᵀ S⁻¹ λ + prior(η)` and `λ̂(η) = argmin_λ F`.
Because `∂F/∂λ = 0` at the inner optimum,

```
dF_prof/dη = ∂F/∂η |_{λ̂}.
```

So we do **not** differentiate through the linear solve. In code: compute
`λ̂` (CG), `stop_gradient` it, evaluate `F(η, stop_gradient(λ̂))`, and let `jax`
differentiate the explicit `η` dependence. The `λᵀ S⁻¹ λ` term then contributes
zero gradient (it has no explicit `η`), exactly as the envelope theorem requires.

### Bilinear parameters

If a sky parameter `A` and a beam/instrument parameter `B` enter as a product
`A·B` (e.g. data `~ gain · flux`), they are *jointly* non-linear and cannot both
sit in `λ`. The clean resolution: **put one factor in `λ`, the other in `η`**.
Then for fixed `η` the model is linear in `λ` again and the whole machinery
applies. The spectral demo's optional `inst.gain` is exactly this — `gain ∈ η`,
`flux ∈ λ`, and `predict = gain · (A₀ λ)` stays conditionally linear. (A
broadband gain is degenerate with the flux amplitude, so it carries an
informative Gaussian prior to be identifiable.)

## 2. The three layers

### Layer 1 — parameter/prior registry (`Param`, `ParamSet`)

The one genuinely new abstraction, and the reusable core. Each parameter block
is a `Param(name, kind, …)` with `kind ∈ {linear, nonlinear}`:

* **linear** carries a `reparam` (a real reparameterisation object) and a
  `ClPrior`. For a real band-limited sky map the reparam is `RealAlmBlock`, which
  packs `θ = [Re(a_lm); Im(a_{l,m>0})]` (eliminating the `Im(a_{l,0})` null
  space) and builds `S⁻¹ = 1/C_l`. The messy real/imag convention lives *in the
  block that owns it*, never in the solver.
* **nonlinear** carries `init`, optional box `bounds` (handled directly by
  L-BFGS-B), an optional `prior` callable (`value → log p`; flat by default), and
  a `bijector` slot reserved for unconstrained HMC.

`ParamSet` collects blocks from all modules and exposes a uniform API the
solvers are written against: `Sinv_linear()`, `pack/unpack_linear`,
`nonlinear_init / nonlinear_bounds / pack/unpack_nonlinear`, and
`logprior_nonlinear`.

### Layer 2 — module ports

A *module* is one physical component. It declares its parameters with namespaced
names (`"sky.flux"`, `"inst.gain"`) via `params()` and implements a **port** that
the assembly chains. Ports receive the merged parameter dict and select their own
keys. Modules with no free parameters still implement their port (`params()`
empty), so the assembly is uniform.

| Module | Port | Parameters |
| --- | --- | --- |
| `SpectralSkyModule` | `sky(p) -> SpectralHealpixSky` | `sky.flux` (linear, `C_l`), `sky.beta` (nonlinear, box) |
| `BeamModule` | `beam(p) -> beam pytree or None` | none (baseline); subclass for regolith/impedance |
| `InstrumentModule` | `apply(p, vis) -> vis` | optional `inst.gain` (nonlinear, Gaussian prior) |

The ports are the natural seams already in luseepy: `get_alm` on skies, the
`beam=` pytree on `CroSimulator.simulate`, and a data-space map on the
instrument.

### Layer 3 — explicit assembly + shared drivers

`Experiment` wires the modules and the simulator into one conditionally-linear
forward model:

```python
def predict(self, linear, nonlinear):
    p   = {**linear, **nonlinear}
    sky  = self.sky_module.sky(p)
    beam = self.beam_module.beam(p)
    vis  = self.sim.simulate(sky=sky, beam=beam).ravel()
    return self.inst_module.apply(p, vis)
```

This assembly is **explicit Python by design**. The registry generalises
*parameters and solving*; but `sky → beam → instrument` is real physics wiring,
not a generic graph, and trying to auto-compose it is the over-engineering trap.
Each fiducial model is a short, readable assembly — and they all share the same
drivers:

* `linear_solve_cg` — matrix-free preconditioned CG for the Wiener step. The
  normal-equation operator is applied via `jax.vjp` (reverse-mode only), so it
  works through `s2fft`'s `custom_vjp` transforms, which forbid forward-mode.
  A diagonal `M = diag(C_l)` preconditioner equalises the orders of magnitude in
  `S⁻¹`.
* `profile_optimize` — the generic VarPro driver. It knows nothing about skies
  or betas: it splits linear/nonlinear via the `ParamSet`, runs the inner solve
  over the assembled linear block, and L-BFGS-Bs the non-linear vector with the
  envelope gradient.

Adding a module never touches the solvers.

## 3. The spectral example end to end

Model: `T(θ, f) = flux(θ) · (f/f_fid)^β(θ) = flux(θ) · exp(β(θ) · ln(f/f_fid))`.

* `flux` lives in `a_lm` — natural for the `C_l` prior — and is **linear**.
* `β` lives in real space at a coarse `beta_nside` — natural for the per-pixel
  box prior `[-4, -1.5]` and the per-pixel power law — and is **non-linear**.

**Why the product is done in real space.** `exp(β·ln f)` has no closed harmonic
form, so `β` must be evaluated in pixel space regardless. Once there, the product
`flux · exp(…)` is an elementwise multiply costing `2 + N_freq` spherical-harmonic
transforms total. The same product as a harmonic convolution would be an
`O(l⁵)` Gaunt contraction *and* still need the transform for the exponential —
strictly more expensive. So `flux` stays in `a_lm`, `β` in real space, and
`SpectralHealpixSky.get_alm` does:

```
flux_alm --SHT--> flux_map ; β_pix --(upsample)--> β_map
per freq:  flux_map · exp(β_map · ln(f/f_fid)) --SHT--> sky_alm(f)
```

All transforms use `s2fft` (JAX backend), so `get_alm` is differentiable in both
`flux_alm` (linear) and `β_pix` (non-linear). The whole `SpectralHealpixSky` is a
registered pytree whose two children are exactly the free-parameter arrays, so it
drops into `CroSimulator.simulate(sky=…)`.

**Crucial self-consistency note.** `s2fft` healpix sampling requires
`lmax + 1 ≥ 2·Nside`, and its inverse→forward round trip is not exact (healpix has
no sampling theorem). That approximation lives *identically* inside the forward
operator used to both **generate** and **fit** the data, so it is part of `A` and
introduces no bias in the recovery.

The entire model assembly reduces to a declaration:

```python
exp = Experiment(
    sim,
    sky=SpectralSkyModule(lmax=…, Nside=…, freq=…, f_fid=…,
                          beta_nside=…, cl_flux=…),
    beam=BeamModule(),                       # no free params
    instrument=InstrumentModule(gain=False), # flip to True for the bilinear demo
    data=data, N_inv=N_inv)
result = exp.optimize(maxiter=…)
```

## 4. A second instance: the separable template model

To check that the architecture generalises without special-casing, a second
model is built on the *same* registry and drivers (`lusee/SeparableSky.py`,
`SeparableSkyModule`, `notebooks/separable_fit_demo.py`):

    T(theta, f) = sum_i  flux_i(theta) * shape_i(f)      (n_templates = 2)

with template maps `flux_i` (alm) as the **linear** block and spectral shapes
`shape_i(f)` as the **non-linear** block. This is the bilinear product the design
anticipated — `flux_i` in `lambda`, `shape_i` in `eta` — and it exercised three
things the first model did not:

* **Multiple linear blocks.** `SeparableSkyModule.params()` declares
  `sep.flux.0` and `sep.flux.1` as two linear `Param`s; `ParamSet` concatenates
  their `S^{-1}` and the CG Wiener step solves them jointly. No solver change.
* **A gauge.** The decomposition has a per-template scaling (and template-mixing)
  freedom. The scaling is fixed in the *model* by anchoring each shape at a
  reference frequency (`shape_i(f_ref)=1`), implemented inside the module — not a
  framework kludge. Gauge-invariant quantities (the total flux at any frequency)
  are what is compared; the residual mixing is broken at the MAP by the
  per-template `C_l` priors.
* **No spherical-harmonic transform.** Because both factors meet in alm space,
  `get_alm` is a single `einsum` (`a_lm(f) = sum_i shape_i(f) flux_i_alm`), so the
  forward is cheaper than the power-law model.

The data are the real ULSA maps (the rank-2 model is mildly mis-specified). The
recovery is reported as gauge-invariant total flux at low/mid/high frequency plus
the two recovered shapes. Reuse outcome: **only the sky module is new** — the
registry, `Experiment`, `profile_optimize`, `linear_solve_cg`, `BeamModule` and
`InstrumentModule` are imported unchanged.

## 5. Performance: CPU baseline and GPU path

The original fitting path was faster on CPU than GPU. On an identical
lmax=31 / Nside=16 / 3-template separable fit:

| backend | compile | per evaluation | full fit (117 evals) |
| --- | --- | --- | --- |
| CPU | ~95 s | ~2.7 s | **310 s** |
| GPU (RTX 4090) | ~17 min | ~480 s | ~15 h (extrapolated) |

The workload was **latency/dispatch-bound, not throughput-bound**. The old
`CroSimulator` path rebuilt fixed beam/frame rotations inside each forward and
called the croissant convolution separately for each real/imag beam channel. The
inner CG solve then re-ran that forward many times per optimiser evaluation. At
the low resolutions used in the demos, this produced a very large XLA graph made
from many tiny kernels; GPU compile time and command-buffer pressure dominated.

The current path adds two GPU-oriented changes while preserving the old fallbacks:

* `CroSimulator.simulate()` now defaults to a cached batched MEPA plan for fixed
  beams. The observation phases, galactic→MEPA rotation coefficients, and
  topocentric→MEPA beam alms are cached once; all real/imag output channels are
  stacked and convolved with one batched `einsum`. The old loop path is still
  used for plotting and dynamic `beam=` pytrees.
* `profile_optimize()` has `inner_method={auto,cg,dense}`. `auto` chooses a
  dense Cholesky solve when the linear block is small (`n_linear <=
  dense_threshold`, default 512) and CG otherwise. For the common
  `lmax=16` spectral run, `n_linear=289`, so the inner solve is dense by default.
  This avoids thousands of matrix-free VJP/CG forwards on the GPU.

`linear_fisher()` has the same idea: for small blocks it builds the dense Fisher
matrix by batching all design-matrix columns with `jax.vmap`, then computes the
covariance by Cholesky. The old Python column loop remains available via
`--fisher-method loop`.

Short validation probe on an RTX 4090, using
`--truth ulsa --lmax 16 --nside 8 --beta-nside 8 --dt-hours 1 --maxiter 1
--inner-maxiter 20`:

| path | result |
| --- | --- |
| old GPU path | `jit_loss` compile >2m40s, then CUDA graph command-buffer OOM |
| new GPU path | selected `inner solve: dense (n_linear=289)`, completed 3 objective evals in ~12 s |
| new GPU path + `--fisher` | Fisher completed in ~2 s after the short fit |

This does not mean every full run is GPU-optimal. Dense solves are intended for
small linear blocks; for larger `lmax`, leave `--inner-method auto` or force
`--inner-method cg`. When sharing a GPU between jobs, JAX may still preallocate a
large fraction of the card; cap it with `XLA_PYTHON_CLIENT_MEM_FRACTION` if
needed.

## 6. Roadmap

* **Sampling.** The current `sample_posterior()` still samples the joint
  `(linear, nonlinear)` posterior, but it now exposes BlackJAX engine selection:
  `engine="nuts"` (default) or `engine="hmc"` with `hmc_num_integration_steps`.
  A future Rao–Blackwellised sampler could marginalise the linear block and add
  the dense-system `−½ log det(A^T N^{-1} A + S^{-1})` term, but this should be
  benchmarked before replacing the joint sampler.
* **Parameterised beam.** Subclass `BeamModule` to return a `CachedBeam` pytree
  built from regolith/impedance parameters; it slots into the same `ParamSet`.
* **Vectorise the forward** (see §5) — the single highest-value performance fix,
  and the prerequisite for the GPU ever being worthwhile here.

## Running the demos (CLI)

Both drivers are command-line programs (`--help` lists every option and its
default) that write a single `.npz`; the recovery notebooks read *everything*
from that `.npz` (set `RESULT` / the loaded path to point at it). Run with
`JAX_ENABLE_X64=1` and `LUSEE_DRIVE_DIR` set (beam + ULSA data). Use
`JAX_PLATFORMS=cpu` for CPU-only runs; omit it to let JAX use a visible GPU.

```bash
# Spectral (power-law) — MAP, default Nside=16/lmax=31, real ULSA data:
python notebooks/spectral_fit_demo.py --truth ulsa
# add Fisher (Wiener-weighted) recovery + HMC posterior, custom output:
python notebooks/spectral_fit_demo.py --truth ulsa --fisher --hmc -o out/spec.npz
# GPU-oriented low-resolution run; auto selects the dense inner solve at lmax=16:
python notebooks/spectral_fit_demo.py --truth ulsa --lmax 16 --nside 8 \
    --beta-nside 8 --dt-hours 1 --fisher --hmc
# self-consistent (theory) data instead of ULSA, low-res quick run:
python notebooks/spectral_fit_demo.py --truth powerlaw --lmax 15 --nside 8 \
    --beta-nside 4 --dt-hours 8 --hmc

# Separable (templates) — 3 templates, HMC, data-PCA shape init:
python notebooks/separable_fit_demo.py --n-templates 3 --hmc -o out/sep.npz
```

Key options (shared): `--lmax`, `--nside`, `--freq f1 f2 …`, `--dt-hours`,
`--target-snr`, `--maxiter`, `--inner-maxiter`, `--inner-method {auto,cg,dense}`,
`--dense-threshold`, `--hmc`, `--hmc-engine {nuts,hmc}`, `--hmc-steps`
(+`--num-samples` / `--num-warmup`), `--seed`, `-o/--output`. Spectral-only:
`--truth {powerlaw,ulsa}`, `--beta-nside`, `--fisher`,
`--fisher-method {auto,dense,loop}`, `--fit-gain`. Separable-only:
`--n-templates`, `--ref-freq`, `--init-from {data,truth}`. The same names are
keyword arguments of each module's `run()`.

## File map

| File | Role |
| --- | --- |
| `lusee/SpectralSky.py` | `SpectralHealpixSky` — the power-law model (pytree, differentiable `get_alm`) |
| `lusee/SeparableSky.py` | `SeparableHealpixSky` — the separable template model (no SHT in forward) |
| `lusee/Fitting.py` | Layer 1 (`Param`/`ParamSet`/`RealAlmBlock`), Layer 2 modules, Layer 3 `Experiment`, drivers (`linear_solve_cg`, `linear_solve_dense`, `profile_optimize`) |
| `notebooks/spectral_fit_demo.py` | Power-law driver (`truth='powerlaw'` self-consistent, or `'ulsa'` real maps) |
| `notebooks/spectral_fit_recovery.ipynb` | Input vs recovered flux/β maps and `ρ_ℓ` |
| `notebooks/separable_fit_demo.py` | Separable-model driver (ULSA data, rank-2 PCA truth) |
| `notebooks/separable_fit_recovery.ipynb` | Total flux at 3 freqs (ULSA vs recovered) and the 2 shapes |
| `docs/wirtinger_cg.md` | Real-vs-complex CG parameterisation (companion note) |
