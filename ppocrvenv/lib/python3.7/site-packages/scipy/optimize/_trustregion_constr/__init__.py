"""This module contains the equality constrained SQP solver."""


from .minimize_trustregion_constr import _minimize_trustregion_constr

__all__ = ['_minimize_trustregion_constr']
