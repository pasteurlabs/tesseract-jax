# Batching strategies for `jax.vmap`

When you wrap an `apply_tesseract` call with `jax.vmap`, the `vmap_method` parameter controls how the batch dimension is handled. Each method has different trade-offs for performance, compatibility, and what it requires from the Tesseract. Options here are ordered from safe and slow to faster but more reliant on guarantees of your Tesseract internals.

- **Start with `"sequential"`** if you are unsure. It should always work.
- **Use `"broadcast_all"`** when the Tesseract requires uniform shapes across all inputs.
- **Use `"expand_dims"`** when you know the Tesseract accepts a leading batch dimension on all inputs and handles broadcasting internally.
- **Use `"auto_experimental"`** for Tesseracts with `Array[..., dtype]` schemas when you only ever vmap over differentiable inputs. It auto-detects when vectorization is safe and falls back to sequential otherwise.


## Quick reference

| Method | Shape of unbatched args | Inspects InputSchema | Tesseract requirement |
|---|---|---|---|
| `"sequential"` | Unchanged (one call per element) | None | Any Tesseract |
| `"broadcast_all"` | `(batch_size, ...)` | None | Must accept a leading batch dim |
| `"expand_dims"` | `(1, ...)` | None | Must accept a leading batch dim |
| `"auto_experimental"` | `(1, ...)` | Yes | `Array[..., dtype]` on all batched differentiable inputs |

## Methods in detail

### `None` (default)

```python
apply_tesseract(tess, inputs)
```

No vmap support. Raises `NotImplementedError` if `jax.vmap` is applied. This also affects `jax.jacfwd` and `jax.jacrev`, which use `jax.vmap` internally. You must set `vmap_method` explicitly to use any of these. All other JAX transforms (`jit`, `grad`, `jvp`) work normally.

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

| Input type | Example | Traced? | Batched by vmap methods? |
|---|---|---|---|
| Python scalar | `float`, `int`, `bool` | No (static) | Never |
| Scalar array (0-d) | `jnp.float32(1.0)`, `Float64` | Yes | Yes |
| Array | `jnp.ones((3,))` | Yes | Yes |
| String / other | `"hello"` | No (static) | Never |

Scalar arrays (0-d) are treated as regular array args. Under `"expand_dims"`, a scalar `()` becomes `(1,)`. Under `"broadcast_all"`, it becomes `(batch,)`. If the Tesseract's schema expects a scalar (`Array[(), Float64]`), these methods may cause a shape mismatch -- use `"auto_experimental"` or `"sequential"` in that case.

## Interaction with autodiff

All methods enable `jax.vmap` to be fully compatible with `jax.grad`, `jax.jvp`, and `jax.vjp` but sometimes involve additional broadcasting or sequential calls to the Tesseract.

- **Forward-mode** (`jax.jvp`/`jax.linearize`/`jax.jacfwd`): Tesseracts require dimensions of tangent vectors to match their corresponding inputs. Therefore, all methods (except `"sequential"`) perform broadcasting to ensure batch dimensions of tangent vectors and their corresponding inputs match.
- **Reverse-mode** (`jax.vjp`/`jax.grad`/`jax.jacrev`): If cotangent vectors are directly batched (e.g. by `jax.jacrev`), all methods fall back to `"sequential"` for that VJP call, since Tesseract VJP endpoints do not support batched cotangent vectors.

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

### Example: `jacfwd(f)` — batched tangents, unbatched primals

This results in a single Tesseract call with broadcasting of primals to match tangent dimensions.

```python
def f(x):
    return apply_tesseract(tess, {"x": x, "y": y0}, vmap_method="auto_experimental")["result"]

jax.jacfwd(f)(x0)  # x0 has shape (3,), result has shape (3,3)
```

`jacfwd` computes a full Jacobian by using `vmap` over tangent vectors. The primal `x0` has shape `(3,)` and is unbatched, while the tangents are the columns of a `(3, 3)` identity matrix — so each tangent has shape `(3,)` and there are 3 of them (the vmap batch dimension).

The batching rule detects that the primal is unbatched but the tangent is batched, and broadcasts the primal to match. The Tesseract sees primals `x=(3, 3)` (three identical vectors broadcast from `(3,)`) and tangents `tx=(3, 3)` (the identity matrix), and returns `jvp_result=(3, 3)` — the full Jacobian.

### Example: `jacrev(f)` — batched cotangents, unbatched primals

This results in multiple Tesseract calls (one for each cotangent),

```python
def f(x):
    return apply_tesseract(tess, {"x": x, "y": y0}, vmap_method="auto_experimental")["result"]

jax.jacrev(f)(x0)  # x0 has shape (3,), result has shape (3,3)
```

`jacrev` computes the Jacobian using reverse-mode AD. It calls `vjp` once to get the backward function, then uses `vmap` over cotangent vectors (columns of a `(3, 3)` identity matrix) to compute each row of the Jacobian. Since independently batched cotangents are not supported, the batching rule falls back to `"sequential"` for the VJP call — each cotangent basis vector is processed one at a time.
