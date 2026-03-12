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

- **Arrays vs. array-like objects**: Tesseract-JAX is stricter than Tesseract Core in that all array inputs to Tesseracts must be JAX or NumPy arrays, not just any array-like (such as Python floats or lists). As a result, you may need to convert your inputs to JAX arrays before passing them to Tesseract-JAX, including scalar values.

  ```python
  from tesseract_core import Tesseract
  from tesseract_jax import apply_tesseract

  tess = Tesseract.from_image("vectoradd_jax")
  with Tesseract.from_image("vectoradd_jax") as tess:
      apply_tesseract(tess, {"a": {"v": [1.0]}, "b": {"v": [2.0]}})  # ❌ raises an error
      apply_tesseract(tess, {"a": {"v": jnp.array([1.0])}, "b": {"v": jnp.array([2.0])}})  # ✅ works
  ```
- **Additional required endpoints**: Tesseract-JAX requires the [`abstract_eval`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#abstract-eval) Tesseract endpoint to be defined when used in conjunction with automatic differentiation and JAX transformations. This is because JAX, in these cases, mandates abstract evaluation of all operations before they are executed. Additionally, many gradient transformations like `jax.grad` require [`vector_jacobian_product`](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/api/endpoints.html#vector-jacobian-product) to be defined.

```{tip}
When creating a new Tesseract based on a JAX function, use `tesseract init --recipe jax` to define all required endpoints automatically, including `abstract_eval` and `vector_jacobian_product`.
```


- **Non-differentiable inputs**: When an input is not marked as `Differentiable[...]` in the Tesseract schema, differentiating with respect to it raises a `ValueError` in both forward and reverse mode. If you see this error, it likely means you forgot to annotate an input as `Differentiable[...]`, or you are accidentally including a non-differentiable input in your differentiation.

  - **Forward mode** (`jax.jvp`, `jax.jacfwd`): providing a non-symbolic-zero tangent for a non-differentiable input raises a `ValueError`.
  - **Reverse mode** (`jax.vjp`, `jax.grad`, `jax.jacrev`): requesting a gradient with respect to a non-differentiable input raises a `ValueError`.

  In both modes, use one of these strategies to exclude non-differentiable inputs from gradient computation:

  **Closure** — capture non-differentiable inputs outside the differentiated function:

  ```python
  # "b" is non-differentiable according to the Tesseract schema
  def loss_fn(a):
      c = apply_tesseract(tess, {"a": a, "b": b})["c"]  # b captured from outer scope
      return jnp.sum(c**2)

  jax.grad(loss_fn)(a)  # ✅ only differentiates w.r.t. "a"
  ```

  **`argnums`** — explicitly select which arguments to differentiate:

  ```python
  def loss_fn(a, b):
      c = apply_tesseract(tess, {"a": a, "b": b})["c"]
      return jnp.sum(c**2)

  jax.grad(loss_fn, argnums=0)(a, b)  # ✅ only differentiates w.r.t. "a"
  ```

  **`stop_gradient`** — apply `jax.lax.stop_gradient` to the non-differentiable input inside the function, before passing it to `apply_tesseract`. This converts it to a concrete value, so no tangent reaches the primitive boundary:

  ```python
  def loss_fn(a, b):
      b = jax.lax.stop_gradient(b)
      c = apply_tesseract(tess, {"a": a, "b": b})["c"]
      return jnp.sum(c**2)

  jax.grad(loss_fn)(a, b)  # ✅ stop_gradient prevents b from being differentiated
  ```


- **Non-differentiable outputs**: When an output is not marked as `Differentiable[...]` in the Tesseract schema, Tesseract-JAX makes the problem explicit rather than silently producing wrong gradients:

  - **Forward mode** (`jax.jvp`, `jax.jacfwd`): the tangent for the non-differentiable output is `NaN`, which propagates to any downstream computation that depends on it. A `ValueError` is not raised here because the JVP rule is executed before any post-processing (such as `pop` or `stop_gradient`) can discard the output.
  - **Reverse mode** (`jax.vjp`, `jax.grad`, `jax.jacrev`): passing any concrete value as the cotangent for a non-differentiable output raises a `ValueError`. Only a *symbolic zero* `jax._src.ad_util.Zero` is accepted. If you see this error, it most likely means you forgot to annotate an output as `Differentiable[...]` in the Tesseract schema.

  In both modes, you can use one of these strategies to exclude or insulate the non-differentiable output from gradient computation:

  **Pop** — remove it from the return value before differentiation:

  ```python
  def f(inputs):
      res = apply_tesseract(tess, inputs)
      res.pop("nondiff_res")
      return res
  ```

  **`has_aux`** — return it as an auxiliary value outside the differentiated pytree:

  ```python
  def f(inputs):
      res = apply_tesseract(tess, inputs)
      return res["result"], res["nondiff_res"]  # (differentiable outputs, aux)

  primals, f_vjp, nondiff_res = jax.vjp(f, inputs, has_aux=True)
  ```

  **`stop_gradient`** — keep it in the return value but block gradient flow through it. In forward mode this produces a zero tangent instead of `NaN`; in reverse mode it produces a symbolic zero cotangent so no error is raised:

  ```python
  def f(inputs):
      res = apply_tesseract(tess, inputs)
      res["nondiff_res"] = jax.lax.stop_gradient(res["nondiff_res"])
      return res
  ```

  Note that the cotangent/tangent pytree structure must always match the function's output structure. If you exclude outputs via `pop` or `has_aux`, including them in the cotangent raises a `ValueError`:

  ```
  ValueError: unexpected tree structure of argument to vjp function:
    got PyTreeDef({'nondiff_res': *, 'result': *}), but expected PyTreeDef({'result': *})
  ```
