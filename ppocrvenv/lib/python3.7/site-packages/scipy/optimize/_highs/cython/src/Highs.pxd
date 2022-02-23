# distutils: language=c++
# cython: language_level=3

from libc.stdio cimport FILE

from libcpp cimport bool
from libcpp.string cimport string

from .HighsStatus cimport HighsStatus
from .HighsOptions cimport HighsOptions
from .HighsInfo cimport HighsInfo
from .HighsLp cimport (
    HighsLp,
    HighsSolution,
    HighsBasis,
    ObjSense,
)
from .HConst cimport HighsModelStatus

cdef extern from "Highs.h":
    # From HiGHS/src/Highs.h
    cdef cppclass Highs:
        HighsStatus passHighsOptions(const HighsOptions& options)
        HighsStatus passModel(const HighsLp& lp)
        HighsStatus run()
        HighsStatus setHighsLogfile(FILE* logfile)
        HighsStatus setHighsOutput(FILE* output)
        HighsStatus writeHighsOptions(const string filename, const bool report_only_non_default_values = true)

        # split up for cython below
        #const HighsModelStatus& getModelStatus(const bool scaled_model = False) const
        const HighsModelStatus & getModelStatus() const
        const HighsModelStatus & getModelStatus(const bool scaled_model) const

        const HighsInfo& getHighsInfo() const
        string highsModelStatusToString(const HighsModelStatus model_status) const
        #HighsStatus getHighsInfoValue(const string& info, int& value)
        HighsStatus getHighsInfoValue(const string& info, double& value) const
        const HighsOptions& getHighsOptions() const

        HighsStatus writeSolution(const string filename, const bool pretty) const

        HighsStatus setBasis()
        const HighsSolution& getSolution() const
        const HighsBasis& getBasis() const

        bool changeObjectiveSense(const ObjSense sense)

        HighsStatus setHighsOptionValueBool "setHighsOptionValue" (const string & option, const bool value)
        HighsStatus setHighsOptionValueInt "setHighsOptionValue" (const string & option, const int value)
        HighsStatus setHighsOptionValueStr "setHighsOptionValue" (const string & option, const string & value)
        HighsStatus setHighsOptionValueDbl "setHighsOptionValue" (const string & option, const double value)

        string primalDualStatusToString(const int primal_dual_status)
