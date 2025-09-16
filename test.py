
from pprint import pprint
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import jax.numpy as jnp

vectoradd = Tesseract.from_tesseract_api("examples/simple/vectoradd_jax/tesseract_api.py")


a = {"v": jnp.array([1.0, 2.0, 3.0], dtype="float32")}
b = {
    "v": jnp.array([4.0, 5.0, 6.0], dtype="float32"),
    "s": jnp.array(2.0, dtype="float32"),
}

outputs = apply_tesseract(vectoradd, inputs={"a": a, "b": b})
pprint(outputs)