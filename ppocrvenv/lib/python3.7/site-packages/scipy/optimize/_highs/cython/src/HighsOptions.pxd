# distutils: language=c++
# cython: language_level=3

from libc.stdio cimport FILE

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector

from .HConst cimport HighsOptionType

cdef extern from "HighsOptions.h" nogil:

    cdef cppclass OptionRecord:
        HighsOptionType type
        string name
        string description
        bool advanced

    cdef cppclass OptionRecordBool(OptionRecord):
        bool* value
        bool default_value

    cdef cppclass OptionRecordInt(OptionRecord):
        int* value
        int lower_bound
        int default_value
        int upper_bound

    cdef cppclass OptionRecordDouble(OptionRecord):
        double* value
        double lower_bound
        double default_value
        double upper_bound

    cdef cppclass OptionRecordString(OptionRecord):
        string* value
        string default_value

    cdef cppclass HighsOptions:
        # From HighsOptionsStruct:

        # Options read from the command line
        string model_file
        string presolve
        string solver
        string parallel
        double time_limit
        string options_file

        # Options read from the file
        double infinite_cost
        double infinite_bound
        double small_matrix_value
        double large_matrix_value
        double primal_feasibility_tolerance
        double dual_feasibility_tolerance
        double ipm_optimality_tolerance
        double dual_objective_value_upper_bound
        int highs_debug_level
        int simplex_strategy
        int simplex_scale_strategy
        int simplex_crash_strategy
        int simplex_dual_edge_weight_strategy
        int simplex_primal_edge_weight_strategy
        int simplex_iteration_limit
        int simplex_update_limit
        int ipm_iteration_limit
        int highs_min_threads
        int highs_max_threads
        int message_level
        string solution_file
        bool write_solution_to_file
        bool write_solution_pretty

        # Advanced options
        bool run_crossover
        bool mps_parser_type_free
        int keep_n_rows
        int allowed_simplex_matrix_scale_factor
        int allowed_simplex_cost_scale_factor
        int simplex_dualise_strategy
        int simplex_permute_strategy
        int dual_simplex_cleanup_strategy
        int simplex_price_strategy
        int dual_chuzc_sort_strategy
        bool simplex_initial_condition_check
        double simplex_initial_condition_tolerance
        double dual_steepest_edge_weight_log_error_threshhold
        double dual_simplex_cost_perturbation_multiplier
        double start_crossover_tolerance
        bool less_infeasible_DSE_check
        bool less_infeasible_DSE_choose_row
        bool use_original_HFactor_logic

        # Options for MIP solver
        int mip_max_nodes
        int mip_report_level

        # Switch for MIP solver
        bool mip

        # Options for HighsPrintMessage and HighsLogMessage
        FILE* logfile
        FILE* output
        int message_level
        string solution_file
        bool write_solution_to_file
        bool write_solution_pretty

        vector[OptionRecord*] records
