# distutils: language=c++
# cython: language_level=3

from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "HConst.h" nogil:

    const int HIGHS_CONST_I_INF
    const double HIGHS_CONST_INF
    const double HIGHS_CONST_TINY
    const double HIGHS_CONST_ZERO
    const int HIGHS_THREAD_LIMIT
    const bool allow_infinite_costs
    const string FILENAME_DEFAULT

    ctypedef enum HighsModelStatus:
        HighsModelStatusNOTSET "HighsModelStatus::NOTSET" = 0
        HighsModelStatusHIGHS_MODEL_STATUS_MIN "HighsModelStatus::HIGHS_MODEL_STATUS_MIN" = HighsModelStatusNOTSET
        HighsModelStatusLOAD_ERROR "HighsModelStatus::LOAD_ERROR"
        HighsModelStatusMODEL_ERROR "HighsModelStatus::MODEL_ERROR"
        HighsModelStatusPRESOLVE_ERROR "HighsModelStatus::PRESOLVE_ERROR"
        HighsModelStatusSOLVE_ERROR "HighsModelStatus::SOLVE_ERROR"
        HighsModelStatusPOSTSOLVE_ERROR "HighsModelStatus::POSTSOLVE_ERROR"
        HighsModelStatusMODEL_EMPTY "HighsModelStatus::MODEL_EMPTY"
        HighsModelStatusPRIMAL_INFEASIBLE "HighsModelStatus::PRIMAL_INFEASIBLE"
        HighsModelStatusPRIMAL_UNBOUNDED "HighsModelStatus::PRIMAL_UNBOUNDED"
        HighsModelStatusOPTIMAL "HighsModelStatus::OPTIMAL"
        HighsModelStatusREACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND "HighsModelStatus::REACHED_DUAL_OBJECTIVE_VALUE_UPPER_BOUND"
        HighsModelStatusREACHED_TIME_LIMIT "HighsModelStatus::REACHED_TIME_LIMIT"
        HighsModelStatusREACHED_ITERATION_LIMIT "HighsModelStatus::REACHED_ITERATION_LIMIT"
        HighsModelStatusPRIMAL_DUAL_INFEASIBLE "HighsModelStatus::PRIMAL_DUAL_INFEASIBLE"
        HighsModelStatusDUAL_INFEASIBLE "HighsModelStatus::DUAL_INFEASIBLE"
        HighsModelStatusHIGHS_MODEL_STATUS_MAX "HighsModelStatus::HIGHS_MODEL_STATUS_MAX" = HighsModelStatusDUAL_INFEASIBLE


    cdef enum HighsBasisStatus:
        HighsBasisStatusLOWER "HighsBasisStatus::LOWER" = 0, # (slack) variable is at its lower bound [including fixed variables]
        HighsBasisStatusBASIC "HighsBasisStatus::BASIC" # (slack) variable is basic
        HighsBasisStatusUPPER "HighsBasisStatus::UPPER" # (slack) variable is at its upper bound
        HighsBasisStatusZERO "HighsBasisStatus::ZERO" # free variable is non-basic and set to zero
        HighsBasisStatusNONBASIC "HighsBasisStatus::NONBASIC" # nonbasic with no specific bound information - useful for users and postsolve
        HighsBasisStatusSUPER "HighsBasisStatus::SUPER" # Super-basic variable: non-basic and either free and
                                                        # nonzero or not at a bound. No SCIP equivalent
                                        
    cdef enum SolverOption:
        SOLVER_OPTION_SIMPLEX "SolverOption::SOLVER_OPTION_SIMPLEX" = -1
        SOLVER_OPTION_CHOOSE "SolverOption::SOLVER_OPTION_CHOOSE"
        SOLVER_OPTION_IPM "SolverOption::SOLVER_OPTION_IPM"

    cdef enum PrimalDualStatus:
        PrimalDualStatusSTATUS_NOT_SET "PrimalDualStatus::STATUS_NOT_SET" = -1
        PrimalDualStatusSTATUS_MIN "PrimalDualStatus::STATUS_MIN" = PrimalDualStatusSTATUS_NOT_SET
        PrimalDualStatusSTATUS_NO_SOLUTION "PrimalDualStatus::STATUS_NO_SOLUTION"
        PrimalDualStatusSTATUS_UNKNOWN "PrimalDualStatus::STATUS_UNKNOWN"
        PrimalDualStatusSTATUS_INFEASIBLE_POINT "PrimalDualStatus::STATUS_INFEASIBLE_POINT"
        PrimalDualStatusSTATUS_FEASIBLE_POINT "PrimalDualStatus::STATUS_FEASIBLE_POINT"
        PrimalDualStatusSTATUS_MAX "PrimalDualStatus::STATUS_MAX" = PrimalDualStatusSTATUS_FEASIBLE_POINT

    cdef enum HighsOptionType:
        HighsOptionTypeBOOL "HighsOptionType::BOOL" = 0
        HighsOptionTypeINT "HighsOptionType::INT"
        HighsOptionTypeDOUBLE "HighsOptionType::DOUBLE"
        HighsOptionTypeSTRING "HighsOptionType::STRING"
