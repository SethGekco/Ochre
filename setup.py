# Ochre - a layered image editor.
# Copyright (C) 2026 Ochre contributors.
# Licensed under the GNU General Public License v3 or later. See COPYING.
"""Build script. The C accelerator is OPTIONAL and must never break an install.

Ochre is a pure-Python application that happens to ship an optional
extension. A user without a compiler, with the wrong headers, or on an
unsupported platform must still get a working editor -- so a failed build is
reported as a warning and the numpy path takes over. The test suite runs
against both backends on every invocation, which is what makes that a real
promise rather than a hope.
"""

import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CCompilerError, ExecError, PlatformError

FAILURES = (CCompilerError, ExecError, PlatformError, OSError, ValueError)


class optional_build_ext(build_ext):
    """Build the accelerator if we can; carry on cheerfully if we cannot."""

    def run(self):
        try:
            super().run()
        except FAILURES as exc:
            self._warn(exc)

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except FAILURES as exc:
            self._warn(exc)

    @staticmethod
    def _warn(exc):
        sys.stderr.write(
            "\n*** Ochre: the C accelerator did not build (%s).\n"
            "*** This is not fatal. Ochre will use its numpy backend, which\n"
            "*** is a fully supported path and passes the same test suite.\n\n"
            % (exc,))


def _numpy_include():
    try:
        import numpy
        return [numpy.get_include()]
    except ImportError:
        return []


setup(
    cmdclass={"build_ext": optional_build_ext},
    ext_modules=[
        Extension(
            "ochre.engine.accel._ochre_accel",
            ["ochre/engine/accel/_ochre_accel.c"],
            include_dirs=_numpy_include(),
            # -fwrapv and -fno-fast-math are load-bearing, not tuning. The
            # accelerator's contract is bit-identical output to the numpy
            # specification, and both flags forbid the compiler from
            # reassociating or exploiting undefined overflow in ways that
            # would silently break that.
            extra_compile_args=["-O3", "-fwrapv", "-fno-fast-math",
                                "-Wall", "-Wextra", "-Wno-unused-parameter"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
    ],
)
