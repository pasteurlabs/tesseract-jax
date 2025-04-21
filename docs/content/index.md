# Tesseract-JAX

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
   $ tesseract build demo/simple/vectoradd_jax
   ```

3. Use it as part of a JAX program:

   ```python
   import jax
   import jax.numpy as jnp
   from tesseract_core import Tesseract
   from tesseract_jax import apply_tesseract

   # Load the Tesseract
   t = Tesseract.from_image("vectoradd_jax")

   # Run it with JAX
   x = jnp.ones((1000, 1000))
   y = jnp.ones((1000, 1000))

   def vector_add(x, y):
       return apply_tesseract(t, x, y)

    vector_add(x, y) # success!

    # You can also use it with JAX transformations like JIT and grad
    vector_add_jit = jax.jit(vector_add)
    vector_add_jit(x, y)

    vector_add_grad = jax.grad(vector_add)
    vector_add_grad(x, y)
    ```

```{tip}
Now you're ready to jump into our [demos](https://github.com/pasteurlabs/tesseract-jax/tree/main/demo) for more examples on how to use Tesseract-JAX.
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

## License

Tesseract JAX is licensed under the [Apache License 2.0](https://github.com/pasteurlabs/tesseract-jax/LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.
