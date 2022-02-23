from .._shared.utils import deprecated

from ._canny import canny
from ._cascade import Cascade
from ._daisy import daisy
from ._hog import hog
from .texture import (graycomatrix, graycoprops,
                      local_binary_pattern,
                      multiblock_lbp,
                      draw_multiblock_lbp)

from .peak import peak_local_max
from .corner import (corner_kitchen_rosenfeld, corner_harris,
                     corner_shi_tomasi, corner_foerstner, corner_subpix,
                     corner_peaks, corner_fast, structure_tensor,
                     structure_tensor_eigenvalues,
                     structure_tensor_eigvals, hessian_matrix,
                     hessian_matrix_eigvals, hessian_matrix_det,
                     corner_moravec, corner_orientations,
                     shape_index)
from .template import match_template
from .brief import BRIEF
from .censure import CENSURE
from .orb import ORB
from .sift import SIFT
from .match import match_descriptors
from .util import plot_matches
from .blob import blob_dog, blob_log, blob_doh
from .haar import (haar_like_feature, haar_like_feature_coord,
                   draw_haar_like_feature)
from ._basic_features import multiscale_basic_features


@deprecated(alt_func='skimage.feature.graycomatrix',
            removed_version='1.0')
def greycomatrix(image, distances, angles, levels=None, symmetric=False,
                 normed=False):
    return graycomatrix(image, distances, angles, levels, symmetric, normed)


@deprecated(alt_func='skimage.feature.graycoprops',
            removed_version='1.0')
def greycoprops(P, prop='contrast'):
    return graycoprops(P, prop)


__all__ = ['canny',
           'Cascade',
           'daisy',
           'hog',
           'graycomatrix',
           'graycoprops',
           'greycomatrix',
           'greycoprops',
           'local_binary_pattern',
           'multiblock_lbp',
           'draw_multiblock_lbp',
           'peak_local_max',
           'structure_tensor',
           'structure_tensor_eigenvalues',
           'structure_tensor_eigvals',
           'hessian_matrix',
           'hessian_matrix_det',
           'hessian_matrix_eigvals',
           'shape_index',
           'corner_kitchen_rosenfeld',
           'corner_harris',
           'corner_shi_tomasi',
           'corner_foerstner',
           'corner_subpix',
           'corner_peaks',
           'corner_moravec',
           'corner_fast',
           'corner_orientations',
           'match_template',
           'BRIEF',
           'CENSURE',
           'ORB',
           'SIFT',
           'match_descriptors',
           'plot_matches',
           'blob_dog',
           'blob_doh',
           'blob_log',
           'haar_like_feature',
           'haar_like_feature_coord',
           'draw_haar_like_feature',
           'multiscale_basic_features',
           ]
