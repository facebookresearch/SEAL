from distutils.core import setup, Extension
import os

extra_compile_args = ["-std=c++11", "-DNDEBUG", "-O3"]

module1 = Extension('generative_retrieval.cpp_modules._fm_index',
                    include_dirs = [
                           'generative_retrieval/cpp_modules', 
                           os.path.expanduser('~/include')],
                    libraries = ['stdc++', 'sdsl', 'divsufsort', 'divsufsort64'],
                    library_dirs = [os.path.expanduser('~/lib')],
                    sources = [
                           'generative_retrieval/cpp_modules/fm_index.cpp',
                           'generative_retrieval/cpp_modules/fm_index.i',
                     #       'generative_retrieval/cpp_modules/fm_index_wrap.cxx'

                     ],
                    swig_opts=['-I../include', '-c++'],
                    language = 'c++11',
                    extra_compile_args=extra_compile_args,
                    )

setup (name = 'generative_retrieval',
       ext_modules = [module1])
