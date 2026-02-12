<img src="docs/static/logo-transparent.png" width="128" align="right">

### Tesseract-JAX

Tesseract-JAX is a lightweight extension to [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) that makes Tesseracts look and feel like regular [JAX](https://github.com/jax-ml/jax) primitives, and makes them jittable, differentiable, and composable.

[Read the docs](https://docs.pasteurlabs.ai/projects/tesseract-jax/latest/) |
[Explore the examples](https://github.com/pasteurlabs/tesseract-jax/tree/main/examples) |
[Report an issue](https://github.com/pasteurlabs/tesseract-jax/issues) |
[Talk to the community](https://si-tesseract.discourse.group/) |
[Contribute](CONTRIBUTING.md)

---

The API of Tesseract-JAX consists of a single function, [`apply_tesseract(tesseract_client, inputs)`](https://docs.pasteurlabs.ai/projects/tesseract-jax/latest/content/api.html#tesseract_jax.apply_tesseract), which is fully traceable by JAX. This enables end-to-end autodifferentiation and JIT compilation of Tesseract-based pipelines:

```python
@jax.jit
def vector_sum(x, y):
    res = apply_tesseract(vectoradd_tesseract, {"a": {"v": x}, "b": {"v": y}})
    return res["vector_add"]["result"].sum()

jax.grad(vector_sum)(x, y) # 🎉
```

## Quick start

> [!NOTE]
> Before proceeding, make sure you have a [working installation of Docker](https://docs.docker.com/engine/install/) and a modern Python installation (Python 3.10+).

> [!IMPORTANT]
> For more detailed installation instructions, please refer to the [Tesseract Core documentation](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/introduction/installation.html).

1. Install Tesseract-JAX:

   ```bash
   $ pip install tesseract-jax
   ```

2. Build an example Tesseract:

   ```bash
   $ git clone https://github.com/pasteurlabs/tesseract-jax
   $ tesseract build tesseract-jax/examples/simple/vectoradd_jax
   ```

3. Use it as part of a JAX program via the JAX-native `apply_tesseract` function:

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

> [!TIP]
> Now you're ready to jump into our [examples](https://github.com/pasteurlabs/tesseract-jax/tree/main/examples) for more ways to use Tesseract-JAX.

## Sharp edges

- **Arrays vs. array-like objects**: Tesseract-JAX is stricter than Tesseract Core in that all array inputs to Tesseracts must be JAX or NumPy arrays, not just any array-like (such as Python floats or lists). As a result, you may need to convert your inputs to JAX arrays before passing them to Tesseract-JAX, including scalar values.

  ```python
  from tesseract_core import Tesseract
  from tesseract_jax import apply_tesseract

  tess = Tesseract.from_image("vectoradd_jax")
  with Tesseract.from_image("vectoradd_jax") as tess:
      apply_tesseract(tess, {"a": {"v": [1.0]}, "b": {"v": [2.0]}})  # ❌ raises an error
      apply_tesseract(tess, {"a": {"v": jnp.array([1.0])}, "b": {"v": jnp.array([2.0])}})  # ✅ works
  ```
- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined when used in conjunction with automatic differentiation and JAX transformations. This is because JAX in these cases, mandates abstract evaluation of all operations before they are executed. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

> [!TIP]
> When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.

## License

Tesseract-JAX is licensed under the [Apache License 2.0](LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.
