"""Benchmark four-port receiver loading implementations.

The production implementation forms

    M = ZL @ inv(ZA + ZL)

with a right-side linear solve.  This benchmark compares that implementation
with two explicitly vmapped JAX kernels:

* the same right-side solve, one 4x4 system at a time;
* a closed-form adjugate/determinant inverse assembled from explicit 3x3
  determinants.

Every timed callable is lowered and compiled before steady-state timings are
recorded.  Synchronization uses ``block_until_ready`` so GPU timings include
device execution rather than only asynchronous dispatch.
"""

import argparse
import json
import platform
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from lusee.ReceiverImpedance import loading_matrix


def _minor_indices():
    rows = []
    columns = []
    for removed_row in range(4):
        kept_rows = [row for row in range(4) if row != removed_row]
        for removed_column in range(4):
            kept_columns = [
                column
                for column in range(4)
                if column != removed_column
            ]
            rows.append(
                np.broadcast_to(
                    np.asarray(kept_rows, dtype=np.int32)[:, None],
                    (3, 3),
                )
            )
            columns.append(
                np.broadcast_to(
                    np.asarray(kept_columns, dtype=np.int32)[None, :],
                    (3, 3),
                )
            )
    return jnp.asarray(np.stack(rows)), jnp.asarray(np.stack(columns))


_MINOR_ROWS, _MINOR_COLUMNS = _minor_indices()
_COFACTOR_SIGNS = jnp.asarray(
    [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]] * 2
)


def _determinant_3x3(matrix):
    """Evaluate explicit 3x3 determinants over arbitrary leading axes."""
    return (
        matrix[..., 0, 0]
        * (
            matrix[..., 1, 1] * matrix[..., 2, 2]
            - matrix[..., 1, 2] * matrix[..., 2, 1]
        )
        - matrix[..., 0, 1]
        * (
            matrix[..., 1, 0] * matrix[..., 2, 2]
            - matrix[..., 1, 2] * matrix[..., 2, 0]
        )
        + matrix[..., 0, 2]
        * (
            matrix[..., 1, 0] * matrix[..., 2, 1]
            - matrix[..., 1, 1] * matrix[..., 2, 0]
        )
    )


def _inverse_4x4_determinant(matrix):
    """Invert one complex 4x4 matrix through its adjugate and determinant."""
    minors = matrix[_MINOR_ROWS, _MINOR_COLUMNS]
    cofactors = (
        _determinant_3x3(minors).reshape((4, 4))
        * _COFACTOR_SIGNS.astype(matrix.real.dtype)
    )
    determinant = jnp.sum(matrix[0] * cofactors[0])
    return jnp.swapaxes(cofactors, 0, 1) / determinant


def _loading_solve_one(ZA, ZL):
    return jnp.swapaxes(
        jnp.linalg.solve(
            jnp.swapaxes(ZA + ZL, 0, 1),
            jnp.swapaxes(ZL, 0, 1),
        ),
        0,
        1,
    )


def _loading_determinant_one(ZA, ZL):
    return ZL @ _inverse_4x4_determinant(ZA + ZL)


def _flatten_vmap(kernel, ZA, ZL):
    ZA = jnp.asarray(ZA)
    ZL = jnp.asarray(ZL)
    if ZA.shape != ZL.shape or ZA.shape[-2:] != (4, 4):
        raise ValueError("ZA and ZL must have matching (..., 4, 4) shapes.")
    batch_shape = ZA.shape[:-2]
    flat_ZA = ZA.reshape((-1, 4, 4))
    flat_ZL = ZL.reshape((-1, 4, 4))
    result = jax.vmap(kernel)(flat_ZA, flat_ZL)
    return result.reshape(batch_shape + (4, 4))


@jax.jit
def loading_matrix_vmap_solve(ZA, ZL):
    """Right-side solve with an explicit vmap over 4x4 systems."""
    return _flatten_vmap(_loading_solve_one, ZA, ZL)


@jax.jit
def loading_matrix_vmap_determinant(ZA, ZL):
    """Closed-form determinant inverse with an explicit vmap."""
    return _flatten_vmap(_loading_determinant_one, ZA, ZL)


def _block(value):
    jax.tree_util.tree_map(
        lambda leaf: (
            leaf.block_until_ready()
            if hasattr(leaf, "block_until_ready")
            else leaf
        ),
        value,
    )
    return value


def _compile_and_time(function, ZA, ZL, repeats):
    started = perf_counter()
    executable = function.lower(ZA, ZL).compile()
    compile_seconds = perf_counter() - started

    started = perf_counter()
    result = _block(executable(ZA, ZL))
    first_execution_seconds = perf_counter() - started

    for _ in range(3):
        _block(executable(ZA, ZL))
    samples = []
    for _ in range(repeats):
        started = perf_counter()
        _block(executable(ZA, ZL))
        samples.append(perf_counter() - started)
    samples = np.asarray(samples)
    return result, {
        "compile_seconds": compile_seconds,
        "first_execution_seconds": first_execution_seconds,
        "steady_median_seconds": float(np.median(samples)),
        "steady_min_seconds": float(np.min(samples)),
        "steady_p90_seconds": float(np.percentile(samples, 90)),
    }


def _relative_residual(result, ZA, ZL):
    residual = np.asarray(result) @ np.asarray(ZA + ZL) - np.asarray(ZL)
    numerator = np.linalg.norm(residual, axis=(-2, -1))
    denominator = np.maximum(
        np.linalg.norm(np.asarray(ZL), axis=(-2, -1)),
        np.finfo(np.asarray(ZL).real.dtype).tiny,
    )
    return float(np.max(numerator / denominator))


def _relative_difference(left, right):
    numerator = np.linalg.norm(
        np.asarray(left) - np.asarray(right),
        axis=(-2, -1),
    )
    denominator = np.maximum(
        np.linalg.norm(np.asarray(right), axis=(-2, -1)),
        np.finfo(np.asarray(right).real.dtype).tiny,
    )
    return float(np.max(numerator / denominator))


def _typical_inputs(batch_size, dtype, seed):
    rng = np.random.default_rng(seed)
    coupling = (
        rng.normal(size=(batch_size, 4, 4))
        + 1j * rng.normal(size=(batch_size, 4, 4))
    ).astype(dtype)
    load_coupling = (
        rng.normal(size=(batch_size, 4, 4))
        + 1j * rng.normal(size=(batch_size, 4, 4))
    ).astype(dtype)
    identity = np.eye(4, dtype=dtype)[None]
    ZA = 0.35 * coupling + (35.0 + 5.0j) * identity
    ZL = 0.15 * load_coupling + (2.0 - 20.0j) * identity
    return (
        jnp.asarray(ZA, dtype=dtype),
        jnp.asarray(ZL, dtype=dtype),
    )


def _gradient_difference(function, reference, ZA, ZL, argument):
    if argument not in {"ZA", "ZL"}:
        raise ValueError("argument must be 'ZA' or 'ZL'.")

    def candidate_loss(matrices):
        value = (
            function(matrices, ZL)
            if argument == "ZA"
            else function(ZA, matrices)
        )
        return jnp.real(jnp.sum(value.conjugate() * value))

    def reference_loss(matrices):
        value = (
            reference(matrices, ZL)
            if argument == "ZA"
            else reference(ZA, matrices)
        )
        return jnp.real(jnp.sum(value.conjugate() * value))

    differentiated_value = ZA if argument == "ZA" else ZL
    candidate_gradient = jax.jit(jax.grad(candidate_loss))(
        differentiated_value
    )
    reference_gradient = jax.jit(jax.grad(reference_loss))(
        differentiated_value
    )
    _block((candidate_gradient, reference_gradient))
    return _relative_difference(candidate_gradient, reference_gradient)


def _stability_sweep(dtype):
    rng = np.random.default_rng(91)
    target_conditions = (
        (1.0, 1e2, 1e4, 1e6)
        if dtype == jnp.complex64
        else (1.0, 1e4, 1e8, 1e12)
    )
    records = []
    for condition in target_conditions:
        left, _ = np.linalg.qr(
            rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        )
        right, _ = np.linalg.qr(
            rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        )
        singular_values = np.geomspace(1.0, 1.0 / condition, 4)
        system = left @ np.diag(singular_values) @ right.conjugate().T
        ZL = (
            rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        )
        ZA = system - ZL
        ZA = jnp.asarray(ZA[None], dtype=dtype)
        ZL = jnp.asarray(ZL[None], dtype=dtype)
        solve = _block(loading_matrix(ZA, ZL))
        determinant = _block(loading_matrix_vmap_determinant(ZA, ZL))
        determinant_finite = bool(
            np.all(np.isfinite(np.asarray(determinant)))
        )
        solve_finite = bool(np.all(np.isfinite(np.asarray(solve))))
        records.append(
            {
                "condition": float(np.linalg.cond(np.asarray(ZA + ZL)[0])),
                "solve_all_finite": solve_finite,
                "determinant_all_finite": determinant_finite,
                "solve_relative_residual": _relative_residual(
                    solve, ZA, ZL
                ),
                "determinant_relative_residual": _relative_residual(
                    determinant, ZA, ZL
                ),
                "determinant_vs_solve": _relative_difference(
                    determinant, solve
                ),
            }
        )
    return records


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-sizes",
        default="1,64,1024,16384",
        help="Comma-separated numbers of independent 4x4 systems.",
    )
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--dtype",
        choices=("complex64", "complex128", "both"),
        default="both",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)
    batch_sizes = tuple(
        int(value) for value in args.batch_sizes.split(",") if value
    )
    dtype_names = (
        ("complex64", "complex128")
        if args.dtype == "both"
        else (args.dtype,)
    )
    methods = {
        "production_batched_solve": loading_matrix,
        "vmap_solve": loading_matrix_vmap_solve,
        "vmap_explicit_determinant": loading_matrix_vmap_determinant,
    }
    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_kinds": [
            getattr(device, "device_kind", "unknown")
            for device in jax.devices()
        ],
        "repeats": args.repeats,
        "results": {},
    }
    for dtype_name in dtype_names:
        dtype = getattr(jnp, dtype_name)
        dtype_results = {}
        for batch_size in batch_sizes:
            ZA, ZL = _typical_inputs(
                batch_size,
                dtype,
                seed=1400 + batch_size,
            )
            condition = np.linalg.cond(np.asarray(ZA + ZL))
            batch_result = {
                "max_condition": float(np.max(condition)),
                "methods": {},
            }
            outputs = {}
            for name, function in methods.items():
                output, timing = _compile_and_time(
                    function,
                    ZA,
                    ZL,
                    args.repeats,
                )
                timing["matrices_per_second"] = (
                    batch_size / timing["steady_median_seconds"]
                )
                timing["max_relative_residual"] = _relative_residual(
                    output,
                    ZA,
                    ZL,
                )
                outputs[name] = output
                batch_result["methods"][name] = timing
            determinant_name = "vmap_explicit_determinant"
            reference_name = "production_batched_solve"
            batch_result["determinant_vs_production"] = (
                _relative_difference(
                    outputs[determinant_name],
                    outputs[reference_name],
                )
            )
            if batch_size == batch_sizes[0]:
                batch_result["determinant_ZA_gradient_vs_production"] = (
                    _gradient_difference(
                        loading_matrix_vmap_determinant,
                        loading_matrix,
                        ZA,
                        ZL,
                        "ZA",
                    )
                )
                batch_result["determinant_ZL_gradient_vs_production"] = (
                    _gradient_difference(
                        loading_matrix_vmap_determinant,
                        loading_matrix,
                        ZA,
                        ZL,
                        "ZL",
                    )
                )
            dtype_results[str(batch_size)] = batch_result
        dtype_results["stability_sweep"] = _stability_sweep(dtype)
        report["results"][dtype_name] = dtype_results
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
