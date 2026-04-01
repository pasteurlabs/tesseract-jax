# Batching strategies for `jax.vmap`

When you wrap an `apply_tesseract` call with `jax.vmap`, the `vmap_method` parameter controls how the batch dimension is handled. Each method has different trade-offs for performance, compatibility, and what it requires from the Tesseract.

## Quick reference

| Method | Batch dim on unbatched args | Schema check | Tesseract requirement |
|---|---|---|---|
| `"sequential"` | N/A (one call per element) | None | Any Tesseract |
| `"auto"` | Unchanged (do nothing) | Yes | `Array[..., dtype]` on all batched differentiable inputs |
| `"expand_dims"` | `(1, ...)` | None | Must accept a leading batch dim |
| `"broadcast_all"` | `(batch, ...)` | None | Must accept a leading batch dim |

## Methods in detail

### `"sequential"` (default)

```python
apply_tesseract(tess, inputs)  # or equivalently:
apply_tesseract(tess, inputs, vmap_method="sequential")
```

Calls the Tesseract once per batch element using `jax.lax.map`. This is safe for any Tesseract regardless of its schema, but may be slow for large batches since each call is a separate request.

### `"auto"`

```python
apply_tesseract(tess, inputs, vmap_method="auto")
```

Inspects the Tesseract's differentiable input schema at JAX trace time. If all batched differentiable inputs use ellipsis shapes (`Array[..., dtype]`), sends a single call with the full batch dimension. Unbatched args are passed through unchanged, the Tesseract handles broadcasting via standard NumPy rules.

Falls back to `"sequential"` when:
- Any batched differentiable input has a fixed shape (e.g. `Array[(None,), Float32]`)
- A batched input is non-differentiable (shape info not yet available in the schema)

### `"expand_dims"`

```python
apply_tesseract(tess, inputs, vmap_method="expand_dims")
```

Adds a leading `(1,)` dimension to every unbatched array arg, then sends a single batched call. The Tesseract must handle broadcasting `(1, ...)` against `(batch, ...)` internally (most NumPy-based implementations do this naturally).

This is a lightweight vectorization method -- no data is duplicated. Use it when you know the Tesseract accepts a leading batch dimension on all inputs and handles broadcasting.

### `"broadcast_all"`

```python
apply_tesseract(tess, inputs, vmap_method="broadcast_all")
```

Broadcasts every unbatched array arg to `(batch, ...)` so all array args have an identical leading batch dimension. This will results in redundant data being transferred to the Tesseract and may increase overhead. Use this when the Tesseract requires all inputs to have matching shapes (e.g. because it checks shape consistency rather than broadcasting).

## How static vs array inputs are handled

The batching method only affects **array args** -- values that are JAX tracers at trace time. Values that are not tracers are treated as **static** and are never transformed by any method.

| Input type | Example | Traced? | Batched by vmap methods? |
|---|---|---|---|
| Python scalar | `float`, `int`, `bool` | No (static) | Never |
| Scalar array (0-d) | `jnp.float32(1.0)`, `Float64` | Yes | Yes |
| Array | `jnp.ones((3,))` | Yes | Yes |
| String / other | `"hello"` | No (static) | Never |

Scalar arrays (0-d) are treated as regular array args. Under `"expand_dims"`, a scalar `()` becomes `(1,)`. Under `"broadcast_all"`, it becomes `(batch,)`. If the Tesseract's schema expects a scalar (`Array[(), Float64]`), these methods may cause a shape mismatch -- use `"auto"` or `"sequential"` in that case.

## Interaction with autodiff

All methods work with `jax.grad`, `jax.jvp`, and `jax.vjp`, however independent batching of (co)tangent vectors is not yet supported by Tesseracts so the vmap methods automatically manage this:

- **JVP / `jax.jacfwd`**: If a primal is batched but its tangent is not (or vice versa), the unbatched one is broadcast to match.
- **VJP / `jax.grad`**: If cotangent vectors are independently batched (e.g. by an outer `jax.vmap`), all methods fall back to `"sequential"` for that VJP call, since Tesseract VJP endpoints do not support batched cotangent vectors.

## Choosing a method

- **Start with `"sequential"`** if you are unsure. It always works.
- **Use `"auto"`** for Tesseracts with `Array[..., dtype]` schemas that handle mismatched batch dimensions.
- **Use `"expand_dims"`** when you know the Tesseract handles broadcasting internally.
- **Use `"broadcast_all"`** when the Tesseract requires uniform shapes across all inputs.
