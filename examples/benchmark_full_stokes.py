"""Small reproducible benchmark for the full-Stokes harmonic hot path."""

import json
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np


def timed(call):
    start = perf_counter()
    result = call()
    jax.tree_util.tree_map(
        lambda value: value.block_until_ready()
        if hasattr(value, "block_until_ready")
        else value,
        result,
    )
    return result, perf_counter() - start


def main():
    jax.config.update("jax_enable_x64", True)
    import croissant as cro

    from lusee.frequencies import FrequencyMap
    from lusee.ReceiverImpedance import loading_matrix
    from lusee.SyntheticResponse import synthetic_four_port_response

    native_freq = np.linspace(10.0, 24.0, 8)
    response = synthetic_four_port_response(
        native_freq,
        angular_step_deg=10.0,
    )
    target = np.linspace(10.1, 23.9, 64)
    target[[7, 31, 53]] = target[3]
    target = target[::-1].copy()
    lmax = 17

    pair_alms, prepare_first = timed(
        lambda: response.pair_stokes_alms(lmax, target)[0]
    )
    count_after_first = response.native_transform_count
    _, prepare_cached = timed(
        lambda: response.pair_stokes_alms(lmax, target)[0]
    )
    count_after_cached = response.native_transform_count

    shape = response._full_sphere_maps(
        response.all_pair_stokes_maps([0])
    ).shape[-2:]
    rng = np.random.default_rng(14)
    sky_maps = rng.normal(size=(native_freq.size, 4) + shape)
    sky = cro.PolarizedSky(
        sky_maps,
        native_freq,
        sampling="mwss",
        coord="mcmf",
        frame="topo",
    )
    sky_native, sky_transform = timed(sky.compute_alm)
    frequency_map = FrequencyMap.build(target, native_freq)
    sky_target = frequency_map.from_native(sky_native)
    phases = jnp.exp(
        -1j
        * jnp.linspace(0.0, 2 * jnp.pi, 128)[:, None]
        * jnp.arange(-lmax, lmax + 1)[None]
    )
    convolve = jax.jit(cro.polarized_convolve)
    result, convolve_first = timed(
        lambda: convolve(pair_alms, sky_target, phases)
    )
    _, convolve_cached = timed(
        lambda: convolve(pair_alms, sky_target, phases)
    )

    ZA = jnp.broadcast_to(
        (30.0 + 4.0j) * jnp.eye(4)[None],
        (target.size, 4, 4),
    )
    ZL = jnp.broadcast_to(
        (2.0 - 20.0j) * jnp.eye(4)[None],
        ZA.shape,
    )
    _, solve_first = timed(lambda: loading_matrix(ZA, ZL))
    _, solve_cached = timed(lambda: loading_matrix(ZA, ZL))

    report = {
        "backend": jax.default_backend(),
        "native_frequencies": int(native_freq.size),
        "target_frequencies": int(target.size),
        "times": int(phases.shape[0]),
        "pairs": int(pair_alms.shape[0]),
        "lmax": lmax,
        "native_transforms_first": count_after_first,
        "native_transforms_after_cached": count_after_cached,
        "result_shape": tuple(int(value) for value in result.shape),
        "seconds": {
            "response_prepare_first": prepare_first,
            "response_prepare_cached": prepare_cached,
            "sky_transform": sky_transform,
            "convolve_first_compile_and_run": convolve_first,
            "convolve_cached": convolve_cached,
            "loading_solve_first": solve_first,
            "loading_solve_cached": solve_cached,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
