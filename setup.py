from distutils.core import setup, Extension
import os

extra_compile_args = ["-std=c++11", "-DNDEBUG", "-O3"]

extension = Extension(
    'seal.cpp_modules._fm_index',
    include_dirs=['generative_retrieval/cpp_modules', os.path.expanduser('~/include')],
    libraries=['stdc++', 'sdsl', 'divsufsort', 'divsufsort64'],
    library_dirs=[os.path.expanduser('~/lib')],
    sources=['seal/cpp_modules/fm_index.cpp', 'seal/cpp_modules/fm_index.i'],
    swig_opts=['-I../include', '-c++'],
    language='c++11',
    extra_compile_args=extra_compile_args,
)

setup(
    name='SEAL',
    version="1.0",
    ext_modules=[extension],
)
