import numpy as np
import jax


def _is_jax_array(value):
    """Return True for JAX array values that should be converted at the boundary."""
    return isinstance(value, jax.Array)


class _ArrayWrapper:
    """Small adapter that converts array values at an object boundary."""

    def __init__(self, wrapped, *, array_predicate, array_converter):
        self._wrapped = wrapped
        self._array_predicate = array_predicate
        self._array_converter = array_converter

    def _unwrap_value(self, value):
        """Pass wrapped objects through to the underlying implementation."""
        if isinstance(value, _ArrayWrapper):
            return value._wrapped
        if isinstance(value, list):
            return [self._unwrap_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._unwrap_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._unwrap_value(item) for key, item in value.items()}
        return value

    def _convert_value(self, value):
        """Recursively convert backend arrays to the target array type."""
        if self._array_predicate(value):
            return self._array_converter(value)
        if isinstance(value, list):
            return [self._convert_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._convert_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._convert_value(item) for key, item in value.items()}
        return value

    def __getattr__(self, name):
        value = getattr(self._wrapped, name)
        if callable(value):
            def wrapped_call(*args, **kwargs):
                args = tuple(self._unwrap_value(arg) for arg in args)
                kwargs = {key: self._unwrap_value(arg) for key, arg in kwargs.items()}
                return self._convert_value(value(*args, **kwargs))
            return wrapped_call
        return self._convert_value(value)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._wrapped, name, self._unwrap_value(value))

    def __repr__(self):
        return f"{type(self).__name__}({self._wrapped!r})"


class NpWrapper(_ArrayWrapper):
    """Expose a JAX-backed object through a NumPy-facing interface."""

    def __init__(self, wrapped):
        super().__init__(wrapped, array_predicate=_is_jax_array, array_converter=np.asarray)
