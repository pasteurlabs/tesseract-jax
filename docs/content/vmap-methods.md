# Batching strategies for `jax.vmap`

When you wrap an `apply_tesseract` call with `jax.vmap`, the `vmap_method` parameter controls how the batch dimension is handled. Each method has different trade-offs for performance, compatibility, and what it requires from the Tesseract. Options here are ordered from safe and slow to faster but more reliant on guarantees of your Tesseract internals.

- **Start with `"sequential"`** if you are unsure. It should always work.
- **Use `"broadcast_all"`** when the Tesseract requires uniform shapes across all inputs.
- **Use `"expand_dims"`** when you know the Tesseract accepts a leading batch dimension on all inputs and handles broadcasting internally.
- **Use `"auto_experimental"`** for Tesseracts with `Array[..., dtype]` schemas when you only ever vmap over differentiable inputs. It auto-detects when vectorization is safe and falls back to sequential otherwise.

`jax.jacfwd` and `jax.jacrev` batch over (co)tangents with `jax.vmap` internally. If the Tesseract has a `jacobian` endpoint, they call it once instead, whatever `vmap_method` is set to, so they need no `vmap_method`. See [Interaction with autodiff](#interaction-with-autodiff) and [`materialize_jacobian`](#materialize-jacobian).

## Quick reference

| Method                | Shape of unbatched args          | Inspects InputSchema | Tesseract requirement                                    |
| --------------------- | -------------------------------- | -------------------- | -------------------------------------------------------- |
| `"sequential"`        | Unchanged (one call per element) | None                 | Any Tesseract                                            |
| `"broadcast_all"`     | `(batch_size, ...)`              | None                 | Must accept a leading batch dim                          |
| `"expand_dims"`       | `(1, ...)`                       | None                 | Must accept a leading batch dim                          |
| `"auto_experimental"` | `(1, ...)`                       | Yes                  | `Array[..., dtype]` on all batched differentiable inputs |

## Methods in detail

### `None` (default)

```python
apply_tesseract(tess, inputs)
```

No vmap support. Raises `NotImplementedError` if `jax.vmap` is applied. This also affects `jax.jacfwd` and `jax.jacrev`, which use `jax.vmap` internally, unless the Tesseract has a `jacobian` endpoint they can call instead (see [`materialize_jacobian`](#materialize-jacobian)). All other JAX transforms (`jit`, `grad`, `jvp`) work normally.

### `"sequential"`

```python
apply_tesseract(tess, inputs, vmap_method="sequential")
```

Calls the Tesseract once per batch element using `jax.lax.map`. This is safe for any Tesseract regardless of its schema, but may be slow for large batches since each call is a separate request.

### `"expand_dims"`

```python
apply_tesseract(tess, inputs, vmap_method="expand_dims")
```

Adds a leading `(1,)` dimension to every unbatched array arg, then sends a single batched call. The Tesseract must handle broadcasting `(1, ...)` against `(batch, ...)` internally.

This is a lightweight vectorization method -- no data is duplicated. Use it when you know the Tesseract accepts a leading batch dimension on all inputs and handles broadcasting.

### `"broadcast_all"`

```python
apply_tesseract(tess, inputs, vmap_method="broadcast_all")
```

Broadcasts every unbatched array arg to `(batch, ...)` so all array args have an identical leading batch dimension. This will results in redundant data being transferred to the Tesseract and may increase overhead. Use this when the Tesseract requires all inputs to have matching shapes (e.g. because it checks shape consistency rather than broadcasting).

### `"auto_experimental"`

```python
apply_tesseract(tess, inputs, vmap_method="auto_experimental")
```

Inspects the Tesseract's InputSchema at JAX trace time. If all batched differentiable inputs use ellipsis shapes (`Array[..., dtype]`), adds a leading `(1,)` dimension to unbatched args and sends a single batched call. This is equivalent to `"expand_dims"` but only when the schema confirms it is safe, with a fallback to `"sequential"` otherwise. This method is considered experimental due to only supporting differentiable inputs (`Differentiable[...]`). Non-differentiable array inputs are not considered and will cause a fallback to sequential even if they have ellipsis shapes.

Falls back to `"sequential"` when:

- Any batched differentiable input has a fixed number of dimensions (e.g. `Array[(None,), Float32]`)
- A batched input is non-differentiable (shape info not yet available in the schema)

## How static vs array inputs are handled

The batching method only affects **array args** -- values that are JAX tracers at trace time. Values that are not tracers are treated as **static** and are never transformed by any method.

| Input type         | Example                       | Traced?     | Batched by vmap methods? |
| ------------------ | ----------------------------- | ----------- | ------------------------ |
| Python scalar      | `float`, `int`, `bool`        | No (static) | Never                    |
| Scalar array (0-d) | `jnp.float32(1.0)`, `Float64` | Yes         | Yes                      |
| Array              | `jnp.ones((3,))`              | Yes         | Yes                      |
| String / other     | `"hello"`                     | No (static) | Never                    |

Scalar arrays (0-d) are treated as regular array args. Under `"expand_dims"`, a scalar `()` becomes `(1,)`. Under `"broadcast_all"`, it becomes `(batch,)`. If the Tesseract's schema expects a scalar (`Array[(), Float64]`), these methods may cause a shape mismatch -- use `"auto_experimental"` or `"sequential"` in that case.

(interaction-with-autodiff)=

## Interaction with autodiff

`jax.jacfwd`, `jax.jacrev`, and `jax.vmap` of `jax.grad`, `jax.jvp`, `jax.vjp` or `jax.linearize` all batch a derivative call to the Tesseract. What runs depends on whether the primals (the point the derivative is taken at), the (co)tangents, or both are batched.

| Primals   | (Co)tangents | Example                                          | Tesseract calls                                                                                                                                                                                                                               |
| --------- | ------------ | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unbatched | Batched      | `jax.jacfwd(f)(x)`, `jax.jacrev(f)(x)`           | One `jacobian` call, if the Tesseract has the endpoint and `materialize_jacobian` is not `False`. This holds whatever `vmap_method` is, `None` included.                                                                                      |
| Batched   | Unbatched    | `jax.vmap(jax.grad(f))(xs)`                      | Batched according to `vmap_method`.                                                                                                                                                                                                           |
| Any       | Batched      | `jax.jacfwd(f)(x)` without a `jacobian` endpoint | If the first row does not apply, JVP calls are batched according to `vmap_method`, and VJP calls run once per cotangent (`"sequential"`) for every `vmap_method` except `None`, since Tesseract VJP endpoints do not take batched cotangents. |

With `vmap_method=None`, every case not handled by a `jacobian` call raises `NotImplementedError`.

When a vectorized method batches a derivative call, it broadcasts arguments so that every batched argument carries the batch dimension:

- **Forward mode** (`jax.jvp`/`jax.linearize`/`jax.jacfwd`): Tesseracts require the shape of each tangent to match its input, so all methods except `"sequential"` broadcast tangents and their inputs to a common batch dimension.
- **Reverse mode** (`jax.vjp`/`jax.grad`): with batched primals and unbatched cotangents, as in `jax.vmap(jax.grad(f))` below, the cotangents are broadcast to `(batch, ...)`, along with every unbatched input that the gradient is taken with respect to, so that the Tesseract returns one gradient per batch element.

Nested transforms are batched one level at a time, innermost first. In `jax.vmap(jax.jacrev(f))`, `jax.jacrev` batches the cotangents at each primal, which becomes a `jacobian` call by the first row of the table. The outer `jax.vmap` then batches the primals of that call. A batched `jacobian` call always runs once per batch element, whatever `vmap_method` is, because the endpoint has no batched form. The forward `apply` call is batched by the outer `jax.vmap` too, so it still needs a `vmap_method`.

(materialize-jacobian)=

### Computing Jacobians: `materialize_jacobian`

```python
apply_tesseract(tess, inputs, materialize_jacobian=None)
```

`materialize_jacobian` decides how a derivative call with unbatched primals and batched (co)tangents is computed (the first row of the table above):

- **`None`** (default): call the `jacobian` endpoint if the Tesseract has one, and batch the JVP / VJP calls otherwise.
- **`True`**: always call the `jacobian` endpoint. Raises `RuntimeError` if the Tesseract has none.
- **`False`**: never call the `jacobian` endpoint. JVP calls are batched according to `vmap_method`, and VJP calls run once per cotangent.

The `jacobian` endpoint returns the full Jacobian, of size (output size × input size), whatever the number of (co)tangents. JAX then contracts it with the batch of (co)tangents. Besides `jax.jacfwd` and `jax.jacrev`, this covers any `vmap` over (co)tangents at a fixed point, such as `jax.vmap(jax.linearize(f, x)[1])` or `jax.vmap(jax.vjp(f, x)[1])`.

Calling the endpoint is usually the faster choice when you need the whole Jacobian, as `jax.jacfwd` and `jax.jacrev` do, or when there are many (co)tangents. `False` can be faster when the Jacobian is large and you only apply it to a few (co)tangents, for example in low-rank approximations or coloring methods.

The Jacobian has the dtype that JAX's own `jax.jacfwd` and `jax.jacrev` would give, which is the output dtype in forward mode and the input dtype in reverse mode. A `jacobian` endpoint that returns another dtype is cast to it. That is always the case in one of the two modes for a Tesseract whose inputs and outputs have different dtypes. When GPU arrays stay on the device, that cast goes through host memory (see [GPU arrays](gpu-arrays.md#dtypes)).

### Example: `vmap(grad(f))` — per-element gradients

This results in a single Tesseract call with no broadcasting.

```python
def f(x, y):
    return apply_tesseract(tess, {"x": x, "y": y}, vmap_method="auto_experimental")["result"]

# x_batch has shape (n, 3), y_batch has shape (n, 3)
jax.vmap(jax.grad(f))(x_batch, y_batch)
```

`vmap` batches the entire forward + backward pass. The Tesseract's derivative endpoints receive the batched shapes:

- **apply** — Tesseract sees `x=(n, 3)`, `y=(n, 3)` → returns `result=(n, 3)`
- **vector_jacobian_product** — Tesseract sees primals `x=(n, 3)`, `y=(n, 3)` and cotangent `ct=(n, 3)` → returns `dx=(n, 3)`, `dy=(n, 3)`

The Tesseract's derivative endpoints must handle these batched inputs correctly; this is automatic when the tesseract AD endpoints are based on the template generated by the `tesseract init --recipe jax` [command](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/creating-tesseracts/create.html).

### Example: `jacfwd(f)` and `jacrev(f)` — through the `jacobian` endpoint

If the Tesseract has a `jacobian` endpoint, both compute the Jacobian with a single call to it, and need no `vmap_method`:

```python
def f(x):
    return apply_tesseract(tess, {"x": x, "y": y0})["result"]

jax.jacfwd(f)(x0)  # x0 has shape (3,), result has shape (3, 3)
jax.jacrev(f)(x0)
```

The Tesseract sees the primals `x=(3,)`, `y=(3,)` and returns the `(3, 3)` Jacobian of `result` with respect to `x`. Both transforms then contract it with the identity basis they batch over, which reproduces the Jacobian.

### Example: `jacfwd(f)` with `materialize_jacobian=False` — batched tangents

This results in a single Tesseract call with broadcasting of primals to match tangent dimensions.

```python
def f(x):
    return apply_tesseract(
        tess,
        {"x": x, "y": y0},
        vmap_method="auto_experimental",
        materialize_jacobian=False,
    )["result"]

jax.jacfwd(f)(x0)  # x0 has shape (3,), result has shape (3, 3)
```

`jacfwd` computes a full Jacobian by using `vmap` over tangent vectors. The primal `x0` has shape `(3,)` and is unbatched, while the tangents are the columns of a `(3, 3)` identity matrix — so each tangent has shape `(3,)` and there are 3 of them (the vmap batch dimension).

The batching rule detects that the primal is unbatched but the tangent is batched, and broadcasts the primal to match. The Tesseract sees primals `x=(3, 3)` (three identical vectors broadcast from `(3,)`) and tangents `tx=(3, 3)` (the identity matrix), and returns `jvp_result=(3, 3)` — the full Jacobian.

### Example: `jacrev(f)` with `materialize_jacobian=False` — batched cotangents

This results in multiple Tesseract calls (one for each cotangent).

```python
def f(x):
    return apply_tesseract(
        tess,
        {"x": x, "y": y0},
        vmap_method="auto_experimental",
        materialize_jacobian=False,
    )["result"]

jax.jacrev(f)(x0)  # x0 has shape (3,), result has shape (3, 3)
```

`jacrev` computes the Jacobian using reverse-mode AD. It calls `vjp` once to get the backward function, then uses `vmap` over cotangent vectors (columns of a `(3, 3)` identity matrix) to compute each row of the Jacobian. Since Tesseract VJP endpoints do not take batched cotangents, the batching rule falls back to `"sequential"` for the VJP call, and each cotangent basis vector is processed one at a time.
