"""Small reproducible benchmark for the full-Stokes harmonic hot path."""

import argparse
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


def memory_stats():
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return None
    return {
        str(key): int(value)
        for key, value in stats.items()
        if isinstance(value, (int, np.integer))
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-frequencies", type=int, default=8)
    parser.add_argument("--target-frequencies", type=int, default=64)
    parser.add_argument("--times", type=int, default=128)
    parser.add_argument("--angular-step-deg", type=float, default=10.0)
    parser.add_argument("--lmax", type=int, default=17)
    parser.add_argument("--preparation-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    jax.config.update("jax_enable_x64", True)
    import croissant as cro

    from lusee.frequencies import FrequencyMap
    from lusee.ReceiverImpedance import loading_matrix
    from lusee.SyntheticResponse import synthetic_four_port_response

    native_freq = np.linspace(
        10.0,
        24.0,
        args.native_frequencies,
    )
    response, response_construct = timed(
        lambda: synthetic_four_port_response(
            native_freq,
            angular_step_deg=args.angular_step_deg,
        )
    )
    target = np.linspace(
        10.1,
        23.9,
        args.target_frequencies,
    )
    duplicate_indices = [
        index
        for index in (7, 31, 53)
        if index < target.size
    ]
    if duplicate_indices:
        target[duplicate_indices] = target[min(3, target.size - 1)]
    target = target[::-1].copy()
    lmax = args.lmax

    pair_alms, prepare_first = timed(
        lambda: response.pair_stokes_alms(lmax, target)[0]
    )
    count_after_first = response.native_transform_count
    _, prepare_cached = timed(
        lambda: response.pair_stokes_alms(lmax, target)[0]
    )
    count_after_cached = response.native_transform_count
    report = {
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "native_frequencies": int(native_freq.size),
        "target_frequencies": int(target.size),
        "pairs": int(pair_alms.shape[0]),
        "lmax": lmax,
        "angular_step_deg": args.angular_step_deg,
        "transform_chunk_shape": response._transform_chunk_shape(lmax),
        "transform_memory_budget_bytes": (
            response.transform_memory_budget_bytes
        ),
        "native_transforms_first": count_after_first,
        "native_transforms_after_cached": count_after_cached,
        "pair_alm_bytes": int(pair_alms.nbytes),
        "memory_after_preparation": memory_stats(),
        "seconds": {
            "response_construct_and_validate": response_construct,
            "response_prepare_first": prepare_first,
            "response_prepare_cached": prepare_cached,
        },
    }
    if args.preparation_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

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
    sky_native, sky_transform = timed(lambda: sky.compute_alm(lmax=lmax))
    frequency_map = FrequencyMap.build(target, native_freq)
    sky_target = frequency_map.from_native(sky_native)
    phases = jnp.exp(
        -1j
        * jnp.linspace(0.0, 2 * jnp.pi, args.times)[:, None]
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

    report["times"] = int(phases.shape[0])
    report["result_shape"] = tuple(int(value) for value in result.shape)
    report["memory_after_full_benchmark"] = memory_stats()
    report["seconds"].update(
        {
            "sky_transform": sky_transform,
            "convolve_first_compile_and_run": convolve_first,
            "convolve_cached": convolve_cached,
            "loading_solve_first": solve_first,
            "loading_solve_cached": solve_cached,
        }
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
