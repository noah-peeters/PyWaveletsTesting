from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(r"algorithms", [r"algorithms.pyx"]),
]

setup(
    name="algorithms",
    ext_modules=cythonize(ext_modules, annotate=True),
    include_dirs=[numpy.get_include()]
)
