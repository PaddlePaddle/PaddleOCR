#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

#include "lanms.h"

namespace py = pybind11;


namespace lanms_adaptor {

	std::vector<std::vector<float>> polys2floats(const std::vector<lanms::Polygon> &polys) {
		std::vector<std::vector<float>> ret;
		for (size_t i = 0; i < polys.size(); i ++) {
			auto &p = polys[i];
			auto &poly = p.poly;
			ret.emplace_back(std::vector<float>{
					float(poly[0].X), float(poly[0].Y),
					float(poly[1].X), float(poly[1].Y),
					float(poly[2].X), float(poly[2].Y),
					float(poly[3].X), float(poly[3].Y),
					float(p.score),
					});
		}

		return ret;
	}


	/**
	 *
	 * \param quad_n9 an n-by-9 numpy array, where first 8 numbers denote the
	 *		quadrangle, and the last one is the score
	 * \param iou_threshold two quadrangles with iou score above this threshold
	 *		will be merged
	 *
	 * \return an n-by-9 numpy array, the merged quadrangles
	 */
	std::vector<std::vector<float>> merge_quadrangle_n9(
			py::array_t<float, py::array::c_style | py::array::forcecast> quad_n9,
			float iou_threshold) {
		auto pbuf = quad_n9.request();
		if (pbuf.ndim != 2 || pbuf.shape[1] != 9)
			throw std::runtime_error("quadrangles must have a shape of (n, 9)");
		auto n = pbuf.shape[0];
		auto ptr = static_cast<float *>(pbuf.ptr);
		return polys2floats(lanms::merge_quadrangle_n9(ptr, n, iou_threshold));
	}

}

PYBIND11_PLUGIN(adaptor) {
	py::module m("adaptor", "NMS");

	m.def("merge_quadrangle_n9", &lanms_adaptor::merge_quadrangle_n9,
			"merge quadrangels");

	return m.ptr();
}

