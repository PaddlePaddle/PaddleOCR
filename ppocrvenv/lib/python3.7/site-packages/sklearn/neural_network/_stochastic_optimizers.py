"""Stochastic optimization methods for MLP
"""

# Authors: Jiyuan Qian <jq401@nyu.edu>
# License: BSD 3 clause

import numpy as np


class BaseOptimizer:
    """Base (Stochastic) gradient descent optimizer

    Parameters
    ----------
    learning_rate_init : float, default=0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    Attributes
    ----------
    learning_rate : float
        the current learning rate
    """

    def __init__(self, learning_rate_init=0.1):
        self.learning_rate_init = learning_rate_init
        self.learning_rate = float(learning_rate_init)

    def update_params(self, params, grads):
        """Update parameters with given gradients

        Parameters
        ----------
        params : list of length = len(coefs_) + len(intercepts_)
            The concatenated list containing coefs_ and intercepts_ in MLP
            model. Used for initializing velocities and updating params

        grads : list of length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip((p for p in params), updates):
            param += update

    def iteration_ends(self, time_step):
        """Perform update to learning rate and potentially other states at the
        end of an iteration
        """
        pass

    def trigger_stopping(self, msg, verbose):
        """Decides whether it is time to stop training

        Parameters
        ----------
        msg : str
            Message passed in for verbose output

        verbose : bool
            Print message to stdin if True

        Returns
        -------
        is_stopping : bool
            True if training needs to stop
        """
        if verbose:
            print(msg + " Stopping.")
        return True


class SGDOptimizer(BaseOptimizer):
    """Stochastic gradient descent optimizer with momentum

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, default=0.1
        The initial learning rate used. It controls the step-size in updating
        the weights

    lr_schedule : {'constant', 'adaptive', 'invscaling'}, default='constant'
        Learning rate schedule for weight updates.

        -'constant', is a constant learning rate given by
         'learning_rate_init'.

        -'invscaling' gradually decreases the learning rate 'learning_rate_' at
          each time step 't' using an inverse scaling exponent of 'power_t'.
          learning_rate_ = learning_rate_init / pow(t, power_t)

        -'adaptive', keeps the learning rate constant to
         'learning_rate_init' as long as the training keeps decreasing.
         Each time 2 consecutive epochs fail to decrease the training loss by
         tol, or fail to increase validation score by tol if 'early_stopping'
         is on, the current learning rate is divided by 5.

    momentum : float, default=0.9
        Value of momentum used, must be larger than or equal to 0

    nesterov : bool, default=True
        Whether to use nesterov's momentum or not. Use nesterov's if True

    power_t : float, default=0.5
        Power of time step 't' in inverse scaling. See `lr_schedule` for
        more details.

    Attributes
    ----------
    learning_rate : float
        the current learning rate

    velocities : list, length = len(params)
        velocities that are used to update params
    """

    def __init__(
        self,
        params,
        learning_rate_init=0.1,
        lr_schedule="constant",
        momentum=0.9,
        nesterov=True,
        power_t=0.5,
    ):
        super().__init__(learning_rate_init)

        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.nesterov = nesterov
        self.power_t = power_t
        self.velocities = [np.zeros_like(param) for param in params]

    def iteration_ends(self, time_step):
        """Perform updates to learning rate and potential other states at the
        end of an iteration

        Parameters
        ----------
        time_step : int
            number of training samples trained on so far, used to update
            learning rate for 'invscaling'
        """
        if self.lr_schedule == "invscaling":
            self.learning_rate = (
                float(self.learning_rate_init) / (time_step + 1) ** self.power_t
            )

    def trigger_stopping(self, msg, verbose):
        if self.lr_schedule != "adaptive":
            if verbose:
                print(msg + " Stopping.")
            return True

        if self.learning_rate <= 1e-6:
            if verbose:
                print(msg + " Learning rate too small. Stopping.")
            return True

        self.learning_rate /= 5.0
        if verbose:
            print(msg + " Setting learning rate to %f" % self.learning_rate)
        return False

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients

        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params

        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        updates = [
            self.momentum * velocity - self.learning_rate * grad
            for velocity, grad in zip(self.velocities, grads)
        ]
        self.velocities = updates

        if self.nesterov:
            updates = [
                self.momentum * velocity - self.learning_rate * grad
                for velocity, grad in zip(self.velocities, grads)
            ]

        return updates


class AdamOptimizer(BaseOptimizer):
    """Stochastic gradient descent optimizer with Adam

    Note: All default values are from the original Adam paper

    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params

    learning_rate_init : float, default=0.001
        The initial learning rate used. It controls the step-size in updating
        the weights

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector, should be
        in [0, 1)

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector, should be
        in [0, 1)

    epsilon : float, default=1e-8
        Value for numerical stability

    Attributes
    ----------
    learning_rate : float
        The current learning rate

    t : int
        Timestep

    ms : list, length = len(params)
        First moment vectors

    vs : list, length = len(params)
        Second moment vectors

    References
    ----------
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(
        self, params, learning_rate_init=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    ):
        super().__init__(learning_rate_init)

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients

        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params

        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [
            self.beta_1 * m + (1 - self.beta_1) * grad
            for m, grad in zip(self.ms, grads)
        ]
        self.vs = [
            self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
            for v, grad in zip(self.vs, grads)
        ]
        self.learning_rate = (
            self.learning_rate_init
            * np.sqrt(1 - self.beta_2 ** self.t)
            / (1 - self.beta_1 ** self.t)
        )
        updates = [
            -self.learning_rate * m / (np.sqrt(v) + self.epsilon)
            for m, v in zip(self.ms, self.vs)
        ]
        return updates
