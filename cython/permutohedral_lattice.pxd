cdef extern from "permutohedral.h":
    cdef cppclass PermutohedralLattice:
        PermutohedralLattice(int d_, int vd_, int nData_) except +
        void splat(float *position, float *value) except +
        void blur() except +
        void beginSlice() except +
        void slice(float *col) except +