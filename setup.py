from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(r"algorithms", [r"algorithms.pyx"]),
]

setup(
    name="algorithms",
    ext_modules=cythonize(ext_modules, annotate=True),
)
