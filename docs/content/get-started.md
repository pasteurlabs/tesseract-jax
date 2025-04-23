# Get started

`tesseract-jax` executes [Tesseracts](https://github.com/pasteurlabs/tesseract-core) as part of JAX programs, with full support for function transformations like JIT, `grad`, `jvp`, and more.

The API of Tesseract-JAX consists of a single function, [`apply_tesseract(tesseract_client, inputs)`](tesseract_jax.apply_tesseract), which is fully traceable by JAX. This enables end-to-end autodifferentiation and JIT compilation of Tesseract-based pipelines.

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
   $ tesseract build examples/simple/vectoradd_jax
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
       res = apply_tesseract(t, {"a": {"v": x}, "b": {"v": y}})
       return res["vector_add"]["result"].sum()

   vector_sum(x, y) # success!

   # You can also use it with JAX transformations like JIT and grad
   vector_sum_jit = jax.jit(vector_sum)
   vector_sum_jit(x, y)

   vector_sum_grad = jax.grad(vector_sum)
   vector_sum_grad(x, y)
   ```

```{tip}
Now you're ready to jump into our [examples](https://github.com/pasteurlabs/tesseract-jax/tree/main/examples) for ways to use Tesseract-JAX.
```

## Sharp edges

- **Arrays vs. array-like objects**: Tesseract-JAX ist stricter than Tesseract Core in that all array inputs to Tesseracts must be JAX or NumPy arrays, not just any array-like (such as Python floats or lists). As a result, you may need to convert your inputs to JAX arrays before passing them to Tesseract-JAX, including scalar values.

  ```python
  from tesseract_core import Tesseract
  from tesseract_jax import apply_tesseract

  tess = Tesseract.from_image("vectoradd")
  apply_tesseract(tess, {"a": 1.0, "b": 2.0})  # ❌ raises an error
  apply_tesseract(tess, {"a": jnp.array(1.0), "b": jnp.array(2.0)})  # ✅ works
  ```
- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined for all operations. This is because JAX mandates abstract evaluation of all operations before they are executed. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

```{tip}
When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.
```
