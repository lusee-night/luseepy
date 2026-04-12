# Differentiable simulation in luseepy

## What works now (branch `an/simulator_ng_jax`)

Beam, BeamGauss, and all SkyModels are registered as JAX pytrees.
`jax.grad` returns gradient objects with the same type and structure as
the input — the likelihood never touches raw arrays.

Two gradient paths coexist:

```python
import jax, jax.numpy as jnp, lusee

obs  = lusee.Observation("2025-03-01 00:00:00 to 2025-03-01 06:00:00",
                         deltaT_sec=3600.0)
beam = lusee.BeamGauss(alt_deg=90, az_deg=0, sigma_deg=20, id="b")
sky  = lusee.sky.HarmonicPointSourceSky(lmax=8, freq=[10.0],
                                         l_deg=0, b_deg=0)

# ── Approach A: precomputed (fast inner loop) ──────────────────
# Build once, differentiate only through simulate().
sim = lusee.CroSimulator(obs, [beam], sky, Tground=0,
                         combinations=[(0,0)], freq=[10.0], lmax=8)

grad_sky = jax.grad(lambda s: jnp.sum(sim.simulate(sky=s) ** 2))(sky)
# grad_sky is a HarmonicPointSourceSky — same type, gradient values
grad_sky._T   # d(loss)/d(source amplitude)

# ── Approach B: through-constructor (full differentiability) ───
# Differentiates through beam ALM computation, sky rotation, everything.
def loss_fn(sky):
    sim = lusee.CroSimulator(obs, [beam], sky, Tground=0,
                             combinations=[(0,0)], freq=[10.0], lmax=8)
    return jnp.sum(sim.simulate() ** 2)

grad_sky = jax.grad(loss_fn)(sky)  # also works, but slower per call
```

Cached-call performance (lmax=64): **A ~ 7 ms/iter, B ~ 660 ms/iter**.
Use A for iterative solves (CG mapmaking), B for one-shot sensitivity
analysis or when differentiating through raw beam Etheta/Ephi.

### Pytree leaf summary

| Class                    | Leaves (differentiated)            | Aux (static)                    |
|--------------------------|------------------------------------|---------------------------------|
| `Beam` / `BeamGauss`     | `Etheta`, `Ephi`, `gain_conv`, ... | grid params, version            |
| `ConstSky`               | `mapalm`, `_T`                     | Nside, frame, freq              |
| `HealpixSky`             | `mapalm`                           | Nside, frame, freq              |
| `HarmonicPointSourceSky` | `_alm`, `_T`                       | lmax, frame, freq               |

### How `sky=` works

`CroSimulator.simulate(sky=...)` calls `sky.get_alm()` inside the MEPA
pipeline. JAX traces through the pytree leaves to the convolution:

```
sky.get_alm()  →  hp_to_2d  →  gal2mepa  →  convolve(beam, sky, phases)
     ↑                                              ↓
  pytree leaves                               waterfall
  (_T, _alm, ...)                         (Ntimes, Nch, Nfreq)
```

The constructor precomputes beam rotation, Wigner-d matrices, and time
phases once. Only the sky path is re-traced per gradient call.

### How `beam=` works

`CroSimulator.simulate(beam=...)` reads `.efbeams` from the beam pytree.
The base class `lusee.CachedBeam` handles all the boilerplate — efbeams
storage, pytree flatten/unflatten, stop_gradient on fixed patterns.
Subclasses only define their free parameters and a `transform_beam`:

```python
import jax, jax.numpy as jnp, lusee

jax.config.update("jax_enable_x64", True)

# ── setup ──────────────────────────────────────────────────────
obs  = lusee.Observation("2025-03-01 00:00:00 to 2025-03-01 06:00:00",
                         deltaT_sec=3600.0)
beam = lusee.BeamGauss(alt_deg=90, az_deg=0, sigma_deg=20, id="b")
sky  = lusee.sky.HarmonicPointSourceSky(lmax=8, freq=[10.0],
                                         l_deg=0, b_deg=0)

sim = lusee.CroSimulator(obs, [beam], sky, Tground=0,
                         combinations=[(0,0)], freq=[10.0], lmax=8)

# ── define a beam pytree ───────────────────────────────────────
# Subclass CachedBeam: override transform_beam + param plumbing.
# The base class handles efbeams storage, stop_gradient, and the
# pytree flatten/unflatten boilerplate.

@jax.tree_util.register_pytree_node_class
class ScaledBeam(lusee.CachedBeam):
    def __init__(self, amplitude, base_efbeams):
        super().__init__(base_efbeams)
        self.amplitude = jnp.asarray(float(amplitude))

    def transform_beam(self, beamreal, groundpower):
        a = self.amplitude
        return a * beamreal, 1.0 - a * (1.0 - groundpower)

    def _param_leaves(self):
        return (self.amplitude,)

    @classmethod
    def _from_param_leaves(cls, params, base_efbeams):
        obj = cls.__new__(cls)
        lusee.CachedBeam.__init__(obj, base_efbeams)
        (obj.amplitude,) = params
        return obj

# ── take the gradient ──────────────────────────────────────────
sb = ScaledBeam(1.0, sim.efbeams)

def loss(b):
    wf = sim.simulate(beam=b)
    return jnp.sum(wf.real ** 2)

grad_beam = jax.grad(loss)(sb)

isinstance(grad_beam, ScaledBeam)  # True
grad_beam.amplitude                # d(loss)/d(amplitude) — scalar
```

The same base class works for richer parameterizations:

- **`InterpolatedBeam(CachedBeam)`**: leaf is `dielectric_params`,
  `transform_beam` calls `BeamInterpolator.interpolate()` (already
  pure JAX) to produce alms from the parameter vector.
- **`TemplateSky`** (sky-side analogue): `gsm_amplitude *
  cached_gsm_alms + sum(flux_i * cached_source_alms_i)`, leaves are
  the amplitudes/fluxes.

## What each simulator supports

| Feature                         | CroSimulator | JaxSimulator | DefaultSimulator |
|---------------------------------|:------------:|:------------:|:----------------:|
| `sky=` kwarg (approach A)       | yes          | not yet      | not yet          |
| `beam=` kwarg (approach A)      | yes          | not yet      | not yet          |
| Through-constructor grad (B)    | yes          | yes          | no               |
| Engine                          | croissant    | s2fft/JAX    | healpy/numpy     |
| Beam alm via JAX recurrence     | no           | yes          | no               |

## Next step: mapmaking via CG

The forward model is the linear operator **A**: sky -> waterfall.
With `sky=`, the CG iteration is:

```python
sim = CroSimulator(obs, beams, sky, ...)

def A(sky):
    return sim.simulate(sky=sky)

def AT(v, sky_template):
    return jax.grad(lambda s: jnp.sum(sim.simulate(sky=s) * v))(sky_template)

def cg_matvec(sky):
    return AT(N_inv * A(sky), sky)

rhs = AT(N_inv * data, sky_init)
sky_hat = jax.scipy.sparse.linalg.cg(cg_matvec, rhs)[0]
```

The adjoint returns a **sky object** whose leaves are gradients.
Both `A` and `AT` are JIT-able. The entire CG solve can run on GPU.

### Open questions

- **Noise model**: What is N? Diagonal (per time-freq)? Block-diagonal?
- **Regularization**: Condition number of A^T A may need preconditioning.
- **`sky=` for JaxSimulator**: Same pattern applies, just needs the kwarg added.

## Environment

- Python 3.12, venv at `.venv/`
- `pip install -e ".[croissant]"` then `pip install jax[cuda12]`
- `JAX_ENABLE_X64=1` recommended for precision
