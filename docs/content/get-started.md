# Get started

## Quick start

```{note}
Before proceeding, make sure you have a [working installation of Docker](https://docs.docker.com/engine/install/) and a modern Python installation (Python 3.10+).
```

```{seealso}
For more detailed installation instructions, please refer to the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/installation.html).
```

1. Install Tesseract-JAX:

   ```bash
   $ pip install tesseract-jax
   ```

2. Build an example Tesseract:

   ```bash
   $ git clone https://github.com/pasteurlabs/tesseract-jax
   $ tesseract build tesseract-jax/examples/simple/vectoradd_jax
   ```

3. Use it as part of a JAX program:

   ```python
   import jax
   import jax.numpy as jnp
   from tesseract_core import Tesseract
   from tesseract_jax import apply_tesseract

   # Load the Tesseract
   t = Tesseract.from_image("vectoradd_jax")
   t.serve()

   # Run it with JAX
   x = jnp.ones((1000,))
   y = jnp.ones((1000,))

   def vector_sum(x, y):
       res = apply_tesseract(t, {"a": {"v": x}, "b": {"v": y}}, vmap_method="sequential")
       return res["vector_add"]["result"].sum()

   vector_sum(x, y) # success!

   # You can also use it with JAX transformations like JIT and grad
   vector_sum_jit = jax.jit(vector_sum)
   vector_sum_jit(x, y)

   vector_sum_grad = jax.grad(vector_sum)
   vector_sum_grad(x, y)

   # vmap requires an explicit vmap_method — "sequential" is safe but slow
   # while "auto_experimental" or "expand_dims" is more efficient for Tesseracts that support batching.
   vector_sum_vmap = jax.vmap(vector_sum)
   vector_sum_vmap(x.reshape(10, 100), y.reshape(10, 100))
   ```

```{seealso}
See [Batching strategies for jax.vmap](vmap-methods.md) for a guide on selecting the appropriate `vmap_method`.
```

```{tip}
Now you're ready to jump into our [examples](https://github.com/pasteurlabs/tesseract-jax/tree/main/examples) for ways to use Tesseract-JAX.
```

## Sharp edges

- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined when used in conjunction with automatic differentiation and JAX transformations. This is because JAX, in these cases, mandates abstract evaluation of all operations before they are executed. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

```{tip}
When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.
```

- **Non-differentiable inputs/outputs**: Differentiating through inputs or outputs not marked as `Differentiable[...]` in the Tesseract schema can raise a `ValueError` or produce `NaN` tangents. See the [Handling Differentiability](handling-differentiability.md) page for details and workarounds.
