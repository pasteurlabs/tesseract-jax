# Sharp Bits


## An output is marked as non differentiable in Tesseract API

For simplicity, we will refer to the non-differentiable output as O and the differentiable outputs as y.

### Forward mode differentiation

For JVPs and forward mode differentiation. If JAX explicitly requests the grads of I we obtain NaNs, as defined in this code section in tesseract_compat.py:

```python
 out = []
for path, aval in zip(output_flat, output_avals, strict=False):
    if path in out_data:
        out.append(out_data[path])
    else:
        # Missing paths mean zero gradient
        out.append(jax.numpy.full_like(aval, jax.numpy.nan))
```

The NaN indicates that the gradient is not defined due to the non-differentiable nature of the output or input.

In the case of general forward-mode differentiation over multiple components, the NaN will be propagated through the computation, leading to NaN gradients for any component that depends on the non-differentiable output or input.


### Reverse mode differentiation

In the case of a VJP:

In this case, we simply do not request the gradient wrt. to O, but all other partial derivates are summed up for the output tangent of the VJP. If a user by accident forgets to mark an output as differentiable and sends a cotangent vector that includes O, he will obtain wrong gradients without errors.

If the user sets O as non-differentiable on purpose, he has to wrap the tesseract in a closure to ensure the correct behaviour for a VJP:

```python
def f(inputs):
    res = apply_tesseract(tess, inputs)
    res.pop("O")
    return res

# request the vjp function
(primal, f_vjp) = jax.vjp(f, inputs)

# Note that the primal cannot contain O
vjp = f_vjp(primal)
```

If the user tries to compute the vjp with a cotangent that does contain O, he will get an error message like:

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ValueError: unexpected tree structure of argument to vjp function: got PyTreeDef({'O': *, 'result': *}), but expected to match PyTreeDef({'result': *})
```

In the case of a general reverse-mode differentiation over multiple components, the user can use stop_gradient to ensure that the non-differentiable output does not contribute to the gradients of the differentiable outputs. For example:

```python
def loss_fn(inputs):
    # Merge differentiable and non-differentiable inputs
    pytree_fn: jax.Callable = lambda inputs: apply_tesseract(
        tess,
        inputs=merge_dicts(inputs),
    )

    y = pytree_fn(diffable_inputs)["y"]
    O = pytree_fn(diffable_inputs)["O"]
    O = jax.lax.stop_gradient(O)
    return jnp.sum(y**2 + O.sum())
```

## An input is marked as non differentiable in Tesseract API

For simplicity, we will refer to the non-differentiable input as I and the differentiable inputs as x.


### Forward mode differentiation

TODO

### Reverse mode differentiation

TODO


## Why we cannot use static arguments for non-differentiable inputs

- For JAX, all non positional arguments are primitives that need to be hashed.
- Our bug is cased by arrays that are traced due to JIT, but we do not wish to compute gradients with respect to them.
- Marking them as static does not work, because we cannot remove the Tracer that wraps them. (JAX limitation, not exactly sure why)
- Therefore we need to keep them inside the positional arguments
