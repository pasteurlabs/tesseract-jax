
from pprint import pprint
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract
import jax.numpy as jnp

vectoradd = Tesseract.from_tesseract_api("examples/simple/partial/tesseract_api.py")


input_dict = {"a": jnp.array([1.0, 2.0, 3.0], dtype="float32")}


outputs = apply_tesseract(vectoradd, inputs=input_dict)
pprint(outputs)