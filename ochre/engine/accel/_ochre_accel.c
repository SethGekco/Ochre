/* Ochre - a layered image editor.
 * Copyright (C) 2026 Ochre contributors.
 * Licensed under the GNU General Public License v3 or later. See COPYING.
 *
 * The optional C accelerator.
 *
 * This file has exactly one job: produce output BIT-IDENTICAL to
 * ochre/engine/accel/fallback.py, faster. That file is the specification,
 * not this one. When the two disagree, this is wrong.
 *
 * Bit-identity is achievable only because every operation is specified in
 * INTEGER arithmetic. A floating-point specification could not be matched by
 * an independent implementation: compilers contract a*b+c into FMA,
 * auto-vectorise with different association orders, and numpy's reductions
 * are pairwise. Integers are exactly defined in both languages, so agreement
 * here is by construction rather than by luck -- and tests/test_accel_parity.py
 * is a meaningful check rather than a flaky one.
 *
 * Two compiler flags in setup.py are load-bearing for the same reason:
 * -fwrapv (defined signed overflow) and -fno-fast-math (no reassociation).
 *
 * Blend formulas are Paint.NET's, from OpenPDN (MIT). See docs/MINING.md.
 * A is the LOWER layer, B the UPPER.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/* Opcodes. ABI: append-only, never renumber. Must match fallback.py. */
enum {
    OP_NORMAL = 0, OP_MULTIPLY, OP_ADDITIVE, OP_COLORBURN, OP_COLORDODGE,
    OP_REFLECT, OP_GLOW, OP_OVERLAY, OP_DIFFERENCE, OP_NEGATION,
    OP_LIGHTEN, OP_DARKEN, OP_SCREEN, OP_XOR, OP_COUNT
};

/* round(a*b/255), exactly, for a,b in 0..255.
 * Verified against round(x/255) over all 65536 pairs by test_arith.py. */
static inline uint32_t mul255(uint32_t a, uint32_t b)
{
    uint32_t t = a * b + 128u;
    return (t + (t >> 8)) >> 8;
}

/* num/den with den == 0 yielding 0, matching fallback._safe_div. */
static inline uint32_t safe_div(uint32_t num, uint32_t den)
{
    return den == 0u ? 0u : num / den;
}

static inline uint32_t umin(uint32_t a, uint32_t b) { return a < b ? a : b; }
static inline uint32_t umax(uint32_t a, uint32_t b) { return a > b ? a : b; }

/* B(Cb, Cs) for one channel. Both operands treated as opaque; the caller
 * applies alpha. Mirrors fallback.blend_channel exactly, branch for branch. */
static inline uint32_t blend_channel_one(int mode, uint32_t a, uint32_t b)
{
    switch (mode) {
    case OP_NORMAL:
        return b;
    case OP_MULTIPLY:
        return mul255(a, b);
    case OP_ADDITIVE:
        return umin(255u, a + b);
    case OP_SCREEN:
        return a + b - mul255(a, b);
    case OP_LIGHTEN:
        return umax(a, b);
    case OP_DARKEN:
        return umin(a, b);
    case OP_DIFFERENCE:
        /* max-min, not abs(b-a): the operands are unsigned. */
        return umax(a, b) - umin(a, b);
    case OP_NEGATION: {
        /* 255 - |255 - a - b| simplifies to s or 510-s, with no subtraction
         * that could underflow. The obvious form computes an underflow in
         * the branch it does not take, which numpy warns about and C does
         * silently -- so both implementations use the closed form. */
        uint32_t s = a + b;
        return s > 255u ? (510u - s) : s;
    }
    case OP_XOR:
        /* Genuinely bitwise, not an arithmetic difference. */
        return (a ^ b) & 0xFFu;
    case OP_COLORBURN:
        if (b == 0u) return 0u;
        return 255u - umin(255u, safe_div((255u - a) * 255u, b));
    case OP_COLORDODGE:
        if (b == 255u) return 255u;
        return umin(255u, safe_div(a * 255u, 255u - b));
    case OP_REFLECT:
        if (b == 255u) return 255u;
        return umin(255u, safe_div(a * a, 255u - b));
    case OP_GLOW:
        /* Reflect with the arguments swapped. Not its own formula. */
        if (a == 255u) return 255u;
        return umin(255u, safe_div(b * b, 255u - a));
    case OP_OVERLAY:
        /* Keys on A, the LOWER layer. Several editors key on the upper one
         * and produce visibly different output. */
        if (a < 128u) return mul255(2u * a, b);
        return 255u - mul255(2u * (255u - a), 255u - b);
    default:
        return b;
    }
}

/* ---- blend_channel(mode, a, b) over uint32 arrays ---------------------- */

static PyObject *py_blend_channel(PyObject *self, PyObject *args)
{
    int mode;
    PyObject *ao, *bo;
    if (!PyArg_ParseTuple(args, "iOO", &mode, &ao, &bo))
        return NULL;
    if (mode < 0 || mode >= OP_COUNT) {
        PyErr_Format(PyExc_ValueError, "unknown blend mode id %d", mode);
        return NULL;
    }

    PyArrayObject *a = (PyArrayObject *)PyArray_FROM_OTF(
        ao, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (!a) return NULL;
    PyArrayObject *b = (PyArrayObject *)PyArray_FROM_OTF(
        bo, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (!b) { Py_DECREF(a); return NULL; }

    npy_intp na = PyArray_SIZE(a), nb = PyArray_SIZE(b);
    /* Only the shapes fallback.py is actually called with: equal, or one
     * scalar-like. Anything else defers to numpy rather than guessing at
     * broadcast semantics and getting them subtly wrong. */
    if (na != nb && na != 1 && nb != 1) {
        Py_DECREF(a); Py_DECREF(b);
        PyErr_SetString(PyExc_ValueError, "blend_channel: incompatible shapes");
        return NULL;
    }

    npy_intp n = na > nb ? na : nb;
    PyArrayObject *src = na >= nb ? a : b;
    PyArrayObject *out = (PyArrayObject *)PyArray_NewLikeArray(
        src, NPY_CORDER, NULL, 0);
    if (!out) { Py_DECREF(a); Py_DECREF(b); return NULL; }

    const uint32_t *pa = (const uint32_t *)PyArray_DATA(a);
    const uint32_t *pb = (const uint32_t *)PyArray_DATA(b);
    uint32_t *po = (uint32_t *)PyArray_DATA(out);

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp i = 0; i < n; ++i) {
        uint32_t va = pa[na == 1 ? 0 : i];
        uint32_t vb = pb[nb == 1 ? 0 : i];
        po[i] = blend_channel_one(mode, va, vb);
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(a); Py_DECREF(b);
    return (PyObject *)out;
}

/* ---- blend_rect --------------------------------------------------------- */

/* Row-strided access. The compositor's arguments are SUB-RECT VIEWS of a
 * larger canvas, so their row stride is the parent's width rather than the
 * slice's -- demanding C-contiguity here would either reject the editor's
 * most common call or force a copy on every blend. What is required is only
 * that each pixel's four channels are adjacent, which is true of any
 * (H, W, 4) uint8 slice. */
typedef struct {
    uint8_t *data;
    npy_intp row;          /* bytes between vertically adjacent pixels */
    npy_intp col;          /* bytes between horizontally adjacent pixels */
} plane_t;

static int plane_of(PyArrayObject *arr, int channels, plane_t *out)
{
    const int nd = PyArray_NDIM(arr);
    const npy_intp *st = PyArray_STRIDES(arr);
    if (channels == 4) {
        if (nd != 3 || PyArray_DIM(arr, 2) != 4 || st[2] != 1)
            return 0;
    } else if (nd != 2) {
        return 0;
    }
    out->data = (uint8_t *)PyArray_DATA(arr);
    out->row = st[0];
    out->col = st[1];
    return 1;
}

static PyObject *py_blend_rect(PyObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"dst", "src", "mode", "opacity", "mask", "out", NULL};
    PyObject *dsto, *srco, *masko = Py_None, *outo = Py_None;
    int mode = OP_NORMAL, opacity = 255;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|iiOO", kwlist,
                                     &dsto, &srco, &mode, &opacity,
                                     &masko, &outo))
        return NULL;
    if (mode < 0 || mode >= OP_COUNT) {
        PyErr_Format(PyExc_ValueError, "unknown blend mode id %d", mode);
        return NULL;
    }

    /* FORCECAST but NOT contiguity: keep strided views as they are. */
    PyArrayObject *dst = (PyArrayObject *)PyArray_FROMANY(
        dsto, NPY_UINT8, 3, 3, 0);
    if (!dst) return NULL;
    PyArrayObject *src = (PyArrayObject *)PyArray_FROMANY(
        srco, NPY_UINT8, 3, 3, 0);
    if (!src) { Py_DECREF(dst); return NULL; }

    if (PyArray_DIM(dst, 2) != 4 || PyArray_DIM(src, 2) != 4) {
        Py_DECREF(dst); Py_DECREF(src);
        PyErr_SetString(PyExc_ValueError, "blend_rect needs (H, W, 4) uint8");
        return NULL;
    }
    if (PyArray_DIM(dst, 0) != PyArray_DIM(src, 0) ||
        PyArray_DIM(dst, 1) != PyArray_DIM(src, 1)) {
        Py_DECREF(dst); Py_DECREF(src);
        PyErr_SetString(PyExc_ValueError, "blend_rect shape mismatch");
        return NULL;
    }

    const npy_intp h = PyArray_DIM(dst, 0), w = PyArray_DIM(dst, 1);

    PyArrayObject *mask = NULL;
    if (masko != Py_None) {
        mask = (PyArrayObject *)PyArray_FROMANY(masko, NPY_UINT8, 2, 2, 0);
        if (!mask) { Py_DECREF(dst); Py_DECREF(src); return NULL; }
        if (PyArray_DIM(mask, 0) != h || PyArray_DIM(mask, 1) != w) {
            Py_DECREF(dst); Py_DECREF(src); Py_DECREF(mask);
            PyErr_SetString(PyExc_ValueError, "blend_rect mask shape mismatch");
            return NULL;
        }
    }

    PyArrayObject *out = NULL;
    if (outo != Py_None) {
        if (!PyArray_Check(outo)) {
            Py_DECREF(dst); Py_DECREF(src); Py_XDECREF(mask);
            PyErr_SetString(PyExc_ValueError, "blend_rect out= must be an array");
            return NULL;
        }
        out = (PyArrayObject *)outo;
        if (PyArray_TYPE(out) != NPY_UINT8 || PyArray_NDIM(out) != 3 ||
            PyArray_DIM(out, 0) != h || PyArray_DIM(out, 1) != w ||
            PyArray_DIM(out, 2) != 4 || PyArray_STRIDES(out)[2] != 1 ||
            !PyArray_ISWRITEABLE(out)) {
            Py_DECREF(dst); Py_DECREF(src); Py_XDECREF(mask);
            PyErr_SetString(PyExc_ValueError,
                            "blend_rect out= must be a writeable (H, W, 4) uint8 "
                            "array of the same shape with contiguous channels");
            return NULL;
        }
        Py_INCREF(out);
    } else {
        npy_intp dims[3] = {h, w, 4};
        out = (PyArrayObject *)PyArray_SimpleNew(3, dims, NPY_UINT8);
        if (!out) { Py_DECREF(dst); Py_DECREF(src); Py_XDECREF(mask); return NULL; }
    }

    /* pm is only read when have_mask, but zeroing it keeps the compiler
     * from warning about a path it cannot prove unreachable -- and a
     * maybe-uninitialized warning is not something to leave standing in
     * a numerics extension. */
    plane_t pd, ps, po;
    plane_t pm = {NULL, 0, 0};
    if (!plane_of(dst, 4, &pd) || !plane_of(src, 4, &ps) ||
        !plane_of(out, 4, &po) || (mask && !plane_of(mask, 1, &pm))) {
        Py_DECREF(dst); Py_DECREF(src); Py_XDECREF(mask); Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError,
                        "blend_rect needs each pixel's channels adjacent");
        return NULL;
    }
    const int have_mask = mask != NULL;

    Py_BEGIN_ALLOW_THREADS

    for (npy_intp y = 0; y < h; ++y) {
        const uint8_t *rd = pd.data + y * pd.row;
        const uint8_t *rs = ps.data + y * ps.row;
        const uint8_t *rm = have_mask ? pm.data + y * pm.row : NULL;
        uint8_t *ro = po.data + y * po.row;

        for (npy_intp x = 0; x < w; ++x) {
            const uint8_t *d = rd + x * pd.col;
            const uint8_t *s = rs + x * ps.col;
            uint8_t *o = ro + x * po.col;

            uint32_t as = s[3];
            if (opacity != 255) as = mul255(as, (uint32_t)opacity);
            if (have_mask) as = mul255(as, rm[x * pm.col]);

            /* Fast path, matching fallback.py: an opaque source in Normal
             * mode simply replaces the backdrop. Applied per pixel here
             * rather than per call, which is strictly more often. */
            if (mode == OP_NORMAL && as == 255u) {
                o[0] = s[0]; o[1] = s[1]; o[2] = s[2]; o[3] = 255u;
                continue;
            }

            uint32_t ab = d[3];
            uint32_t inv_as = 255u - as;
            uint32_t ab_keep = mul255(ab, inv_as);
            uint32_t ao = as + ab_keep;

            if (ao == 0u) {
                o[0] = o[1] = o[2] = o[3] = 0u;
                continue;
            }

            /* Channels are read BEFORE any write, because out may alias dst
             * -- which is exactly how the compositor calls this. */
            const uint32_t cb0 = d[0], cb1 = d[1], cb2 = d[2];
            const uint32_t cs0 = s[0], cs1 = s[1], cs2 = s[2];
            const uint32_t cbv[3] = {cb0, cb1, cb2};
            const uint32_t csv[3] = {cs0, cs1, cs2};

            o[3] = (uint8_t)ao;
            for (int c = 0; c < 3; ++c) {
                uint32_t cs_eff;
                if (mode == OP_NORMAL) {
                    /* Cs' = (1-ab)*Cs + ab*B(Cb,Cs) and B is the identity,
                     * so Cs' == Cs exactly. The long form would be slower
                     * AND would introduce rounding the closed form lacks. */
                    cs_eff = csv[c];
                } else {
                    uint32_t blended = blend_channel_one(mode, cbv[c], csv[c]);
                    cs_eff = mul255(255u - ab, csv[c]) + mul255(ab, blended);
                }
                uint32_t num = as * cs_eff + ab_keep * cbv[c];
                o[c] = (uint8_t)((num + ao / 2u) / ao);
            }
        }
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(dst); Py_DECREF(src); Py_XDECREF(mask);
    return (PyObject *)out;
}

/* ---- module ------------------------------------------------------------- */

static PyMethodDef methods[] = {
    {"blend_channel", py_blend_channel, METH_VARARGS,
     "blend_channel(mode, a, b) -> uint32 array"},
    {"blend_rect", (PyCFunction)(void (*)(void))py_blend_rect,
     METH_VARARGS | METH_KEYWORDS,
     "blend_rect(dst, src, mode=0, opacity=255, mask=None, out=None)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_ochre_accel",
    "Ochre's optional C accelerator. Must match accel/fallback.py exactly.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__ochre_accel(void)
{
    import_array();
    return PyModule_Create(&moduledef);
}
