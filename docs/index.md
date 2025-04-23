# Tesseract-JAX

`tesseract-jax` executes [Tesseracts](https://github.com/pasteurlabs/tesseract-core) as part of JAX programs, with full support for function transformations like JIT, `grad`, `jvp`, and more.

The API of Tesseract-JAX consists of a single function, [`apply_tesseract(tesseract_client, inputs)`](tesseract_jax.apply_tesseract), which is fully traceable by JAX. This enables end-to-end autodifferentiation and JIT compilation of Tesseract-based pipelines.

Now, learn how to [get started](content/get-started.md) with Tesseract-JAX, or explore the [API reference](content/api.md).

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

demo_notebooks/simple.ipynb
demo_notebooks/cfd.ipynb
```

```{toctree}
:caption: See also
:maxdepth: 2
:hidden:

Tesseract Core docs <https://docs.pasteurlabs.ai/projects/tesseract-core/latest/>
Tesseract User Forums <https://si-tesseract.discourse.group/>
```
