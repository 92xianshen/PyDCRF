# distutils: language = c++
# cython: language_level = 3

from permutohedral_lattice cimport PermutohedralLattice
import numpy as np
cimport numpy as np

cdef class PyPermutohedralLattice:
    cdef PermutohedralLattice *c_lattice_ptr

    def __cinit__(self, int d_, int vd_, int nData_):
        self.c_lattice_ptr = new PermutohedralLattice(d_, vd_, nData_)

    def __dealloc__(self):
        del self.c_lattice_ptr

    cdef void splat(self, np.ndarray[np.float32_t, ndim=1] pos, np.ndarray[np.float32_t, ndim=1] col):
        cdef float[::1] pos_memview, col_memview
        pos_memview = pos
        col_memview = col
        self.c_lattice_ptr.splat(&pos_memview[0], &col_memview[0])

    cdef void blur(self):
        self.c_lattice_ptr.blur()

    cdef void beginSlice(self):
        self.c_lattice_ptr.beginSlice()

    cdef np.ndarray[np.float32_t, ndim=1] slice(self, np.ndarray[np.float32_t, ndim=1] col):
        cdef float[::1] col_memview
        col_memview = col
        self.c_lattice_ptr.slice(&col_memview[0])
        return col

def permutohedral_lattice_filter(im, ref):
    ref_channels, im_channels = ref.shape[-1], im.shape[-1]
    height, width = im.shape[0], im.shape[1]
    col = np.zeros((im_channels + 1, ), dtype=np.float32)
    pos = np.zeros((ref_channels, ), dtype=np.float32)

    lattice = PyPermutohedralLattice(ref_channels, im_channels + 1, height * width)

    col[-1] = 1.

    for r in range(height):
        for c in range(width):
            col[:-1] = im[r, c]
            pos = ref[r, c]
            lattice.splat(pos, col)

    lattice.blur()

    out = np.zeros_like(im, dtype=np.float32)

    lattice.beginSlice()
    for r in range(height):
        for c in range(width):
            col = lattice.slice(col)
            scale = 1. / col[-1]
            out[r, c] = col[:-1] * scale

    return out

# cdef np.ndarray[np.float32_t, ndim=3] _filter(self, np.ndarray[np.float32_t, ndim=3] im, np.ndarray[np.float32_t, ndim=3] ref):
#     cdef int ref_channels, im_channels, im_channels1, height, width
#     cdef float scale
#     cdef np.ndarray[np.float32_t, ndim=1] col, pos
#     cdef float[::1] pos_memview, col_memview
#     cdef np.ndarray[np.float32_t, ndim=3] out
    
#     ref_channels, im_channels, im_channels1 = ref.shape[-1], im.shape[-1], im.shape[-1] + 1
#     height, width = im.shape[0], im.shape[1]
#     col = np.zeros((im_channels + 1, ), dtype=np.float32)
#     pos = np.zeros((ref_channels, ), dtype=np.float32)

#     lattice = PyPermutohedralLattice(ref_channels, im_channels + 1, height * width)
    
#     col[-1] = 1.

#     for r in range(height):
#         for c in range(width):
#             col[:-1] = im[r, c]
#             pos = ref[r, c]
#             lattice.splat(&pos_memview[0], &col_memview[0])

#     lattice.blur()

#     out = np.zeros_like(im, dtype=np.float32)

#     lattice.beginSlice()
#     for r in range(height):
#         for c in range(width):
#             lattice.slice(&col_memview[0])
#             scale = 1. / col[-1]
#             out[r, c] = col[:-1] * scale

#     return out

# def permutohedral_lattice_filter(image, reference):
#     # TODO: compute positions from image of numpy.ndarray
#     return _filter(image, reference)