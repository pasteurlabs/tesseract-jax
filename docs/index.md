# Tesseract-JAX

Tesseract-JAX is a lightweight extension to [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) that makes Tesseracts look and feel like regular [JAX](https://github.com/jax-ml/jax) primitives, and makes them jittable, differentiable, and composable.

The API of Tesseract-JAX consists of a single function, [`apply_tesseract(tesseract_client, inputs)`](tesseract_jax.apply_tesseract), which is fully traceable by JAX. This enables end-to-end autodifferentiation and JIT compilation of Tesseract-based pipelines:

```python
@jax.jit
def vector_sum(x, y):
    res = apply_tesseract(vectoradd_tesseract, {"a": {"v": x}, "b": {"v": y}})
    return res["vector_add"]["result"].sum()

jax.grad(vector_sum)(x, y) # 🎉
```

Want to learn more? See how to [get started](content/get-started.md) with Tesseract-JAX, explore the [API reference](content/api.md), or learn by [example](examples/simple/demo.ipynb).

## License

Tesseract JAX is licensed under the [Apache License 2.0](https://github.com/pasteurlabs/tesseract-jax/LICENSE) and is free to use, modify, and distribute (under the terms of the license).

Tesseract is a registered trademark of Pasteur Labs, Inc. and may not be used without permission.


```{toctree}
:caption: Usage
:maxdepth: 2
:hidden:

content/get-started
content/api
```

```{toctree}
:caption: Examples
:maxdepth: 2
:hidden:

examples/simple/demo.ipynb
examples/cfd/demo.ipynb
examples/fem-shapeopt/demo.ipynb
```

```{toctree}
:caption: See also
:maxdepth: 2
:hidden:

Tesseract Core docs <https://docs.pasteurlabs.ai/projects/tesseract-core/latest/>
Tesseract User Forums <https://si-tesseract.discourse.group/>
```
