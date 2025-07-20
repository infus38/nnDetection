from setuptools import setup
from pathlib import Path
import os
import sys

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def clean():
    """Custom clean command to tidy up the project root."""
    os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz')


def get_extensions():
    """
    Build C++/CUDA extensions for nnDetection.
    """
    print("=" * 60)
    print("Building nnDetection C++/CUDA extensions")
    print(f"Python version: {sys.version_info}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA_HOME: {CUDA_HOME}")
    print("=" * 60)

    this_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    extensions_dir = this_dir / 'nndet' / 'csrc'

    # Collect source files
    main_file = list(extensions_dir.glob('*.cpp'))
    source_cpu = []  # Empty for now - add when CPU-specific files are needed
    source_cuda = list((extensions_dir / 'cuda').glob('*.cu'))

    print(f"Main C++ files: {[f.name for f in main_file]}")
    print(f"CPU source files: {[f.name for f in source_cpu]}")
    print(f"CUDA source files: {[f.name for f in source_cuda]}")

    sources = main_file + source_cpu
    extension = CppExtension
    define_macros = []
    extra_compile_args = {"cxx": ["-O3", "-std=c++17"]}

    # Check if we should build with CUDA support
    build_cuda = (
        (torch.cuda.is_available() and CUDA_HOME is not None) or
        os.getenv('FORCE_CUDA', '0') == '1'
    )

    if build_cuda:
        print("Building with CUDA support")
        print(f"CUDA_ARCH_LIST: {os.getenv('TORCH_CUDA_ARCH_LIST', 'auto')}")

        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args["nvcc"] = [
            "-O3",
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        # Use custom compiler if specified
        cc = os.environ.get("CC", None)
        if cc is not None:
            extra_compile_args["nvcc"].append(f"-ccbin={cc}")
    else:
        print("Building without CUDA support")

    # Convert Path objects to strings (keep them relative!)
    sources = [str(s.relative_to(this_dir)) for s in sources]
    include_dirs = [str(extensions_dir.relative_to(this_dir))]

    print(f"Extension type: {extension.__name__}")
    print(f"Source files: {len(sources)} files")
    print(f"Include directories: {include_dirs}")
    print("=" * 60)

    ext_modules = [
        extension(
            name='nndet._C',
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


# Main setup call - pyproject.toml handles most metadata
setup(
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension.with_options(parallel=True),
        'clean': clean,
    },
    zip_safe=False,  # Required for C++ extensions
)
