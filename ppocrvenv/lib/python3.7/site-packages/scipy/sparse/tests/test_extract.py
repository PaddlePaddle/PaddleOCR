"""test sparse matrix construction functions"""

from numpy.testing import assert_equal
from scipy.sparse import csr_matrix

import numpy as np
from scipy.sparse import extract


class TestExtract:
    def setup_method(self):
        self.cases = [
            csr_matrix([[1,2]]),
            csr_matrix([[1,0]]),
            csr_matrix([[0,0]]),
            csr_matrix([[1],[2]]),
            csr_matrix([[1],[0]]),
            csr_matrix([[0],[0]]),
            csr_matrix([[1,2],[3,4]]),
            csr_matrix([[0,1],[0,0]]),
            csr_matrix([[0,0],[1,0]]),
            csr_matrix([[0,0],[0,0]]),
            csr_matrix([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]),
            csr_matrix([[1,2,0,0,3],[4,5,0,6,7],[0,0,8,9,0]]).T,
        ]

    def find(self):
        for A in self.cases:
            I,J,V = extract.find(A)
            assert_equal(A.toarray(), csr_matrix(((I,J),V), shape=A.shape))

    def test_tril(self):
        for A in self.cases:
            B = A.toarray()
            for k in [-3,-2,-1,0,1,2,3]:
                assert_equal(extract.tril(A,k=k).toarray(), np.tril(B,k=k))

    def test_triu(self):
        for A in self.cases:
            B = A.toarray()
            for k in [-3,-2,-1,0,1,2,3]:
                assert_equal(extract.triu(A,k=k).toarray(), np.triu(B,k=k))
