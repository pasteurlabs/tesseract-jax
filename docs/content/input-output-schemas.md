# Input/Output Schemas

Every Tesseract defines its interface through Pydantic `BaseModel` classes (`InputSchema` and `OutputSchema`). These schemas describe the structure, shapes, and dtypes of all array fields, and crucially, which fields are **differentiable**.

## The `Differentiable[...]` annotation

Fields wrapped with `Differentiable[...]` participate in automatic differentiation. Fields without it are treated as constants by JAX's AD machinery, even if their values change between calls.

```python
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float32

class InputSchema(BaseModel):
    x: Differentiable[Array[(3,), Float32]]   # differentiable
    label: Array[(1,), Float32]               # non-differentiable

class OutputSchema(BaseModel):
    loss: Differentiable[Array[(), Float32]]  # differentiable
    metadata: Array[(4,), Float32]            # non-differentiable
```

Fields can be arbitrarily nested (dicts, lists, and nested models). `Differentiable[...]` applies per-leaf:

```python
class InputSchema(BaseModel):
    params: dict[str, Differentiable[Array[(None,), Float32]]]  # all leaves differentiable
    config: dict[str, Array[(None,), Float32]]                  # all leaves non-differentiable
```

When `apply_tesseract` is called inside a JAX differentiation context, Tesseract-JAX inspects these annotations to determine which inputs to request tangents/cotangents for, and which outputs to return derivatives of.

## Non-differentiable inputs

When an input is not marked as `Differentiable[...]` in the Tesseract schema, differentiating with respect to it raises a `ValueError` in both forward and reverse mode. If you see this error, it likely means you forgot to annotate an input as `Differentiable[...]`, or you are accidentally including a non-differentiable input in your differentiation.

- **Forward mode** (`jax.jvp`, `jax.jacfwd`): providing a non-symbolic-zero tangent for a non-differentiable input raises a `ValueError`.
- **Reverse mode** (`jax.vjp`, `jax.grad`, `jax.jacrev`): requesting a gradient with respect to a non-differentiable input raises a `ValueError`.

In both modes, use one of these strategies to exclude non-differentiable inputs from gradient computation:

::::{dropdown} Strategies for handling non-differentiable inputs

**Closure:** capture non-differentiable inputs outside the differentiated function:

```python
# "b" is non-differentiable according to the Tesseract schema
def loss_fn(a):
    c = apply_tesseract(tess, {"a": a, "b": b})["c"]  # b captured from outer scope
    return jnp.sum(c**2)

jax.grad(loss_fn)(a)  # ✅ only differentiates w.r.t. "a"
```

**`argnums`:** explicitly select which arguments to differentiate:

```python
def loss_fn(a, b):
    c = apply_tesseract(tess, {"a": a, "b": b})["c"]
    return jnp.sum(c**2)

jax.grad(loss_fn, argnums=0)(a, b)  # ✅ only differentiates w.r.t. "a"
```

**`stop_gradient`:** apply `jax.lax.stop_gradient` to the non-differentiable input inside the function, before passing it to `apply_tesseract`. This converts it to a concrete value, so no tangent reaches the primitive boundary:

```{warning}
`stop_gradient` changes the mathematical result of differentiation. Only use it if you are confident that gradient contributions through that path are genuinely undesirable or negligible. It is not a safe no-op.
```

```python
def loss_fn(a, b):
    b = jax.lax.stop_gradient(b)
    c = apply_tesseract(tess, {"a": a, "b": b})["c"]
    return jnp.sum(c**2)

jax.grad(loss_fn)(a, b)  # ✅ stop_gradient prevents b from being differentiated
```


::::


## Non-differentiable outputs

When an output is not marked as `Differentiable[...]` in the Tesseract schema, Tesseract-JAX makes the problem explicit rather than silently producing wrong gradients:

- **Forward mode** (`jax.jvp`, `jax.jacfwd`): the tangent for the non-differentiable output is `NaN`, which propagates to any downstream computation that depends on it. A `ValueError` is not raised here because the JVP rule is executed before any post-processing (such as `pop` or `stop_gradient`) can discard the output.
- **Reverse mode** (`jax.vjp`, `jax.grad`, `jax.jacrev`): passing any concrete value as the cotangent for a non-differentiable output raises a `ValueError`. Only a *symbolic zero* `jax._src.ad_util.Zero` is accepted. If you see this error, it most likely means you forgot to annotate an output as `Differentiable[...]` in the Tesseract schema.

In both modes, you can use one of these strategies to exclude or insulate the non-differentiable output from gradient computation:

::::{dropdown} Strategies for handling non-differentiable outputs

**Pop:** remove it from the return value before differentiation:

```python
def f(inputs):
    res = apply_tesseract(tess, inputs)
    res.pop("nondiff_res")
    return res
```

**`has_aux`:** return it as an auxiliary value outside the differentiated pytree:

```python
def f(inputs):
    res = apply_tesseract(tess, inputs)
    return res["result"], res["nondiff_res"]  # (differentiable outputs, aux)

primals, f_vjp, nondiff_res = jax.vjp(f, inputs, has_aux=True)
```

**`stop_gradient`:** keep it in the return value but block gradient flow through it. In forward mode this produces a zero tangent instead of `NaN`; in reverse mode it produces a symbolic zero cotangent so no error is raised:

```{warning}
`stop_gradient` changes the mathematical result of differentiation. Only use it if you are confident that gradient contributions through that path are genuinely undesirable or negligible. It is not a safe no-op.
```

```python
def f(inputs):
    res = apply_tesseract(tess, inputs)
    res["nondiff_res"] = jax.lax.stop_gradient(res["nondiff_res"])
    return res
```

```{tip}
For complex pytrees with many mixed differentiable and non-differentiable leaves, [`equinox.partition`](https://docs.kidger.site/equinox/api/manipulation/#equinox.partition) provides a convenient way to split and recombine them.
```

::::

Note that the cotangent/tangent pytree structure must always match the function's output structure. If you exclude outputs via `pop` or `has_aux`, including them in the cotangent raises a `ValueError`:

```
ValueError: unexpected tree structure of argument to vjp function:
  got PyTreeDef({'nondiff_res': *, 'result': *}), but expected PyTreeDef({'result': *})
```
