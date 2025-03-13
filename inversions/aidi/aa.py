# original repository: https://github.com/joshnguyen99/anderson_acceleration
import numpy as np
import numpy.linalg as LA


class AndersonAcceleration:
    """Anderson acceleration for fixed-point iteration

    (Regularized) Anderson acceleration algorithm, also known as Approximate
    Maximal Polynomial Extrapolation (AMPE). The goal is to find a fixed point
    to some Lipschitz continuous function `g`, that is, find an `x` such that 
    `g(x) = x`. AMPE uses some previous iterates and residuals to solve for 
    coefficients, and then use them to extrapolate to the next iterate. The
    parameters used in this implementation are in [1].

    Parameters
    ----------
    window_size : int (optional, default=5)
        The number of previous iterates to use in the extrapolation. This is
        `m` in the algorithm.

    reg : float (optional, default=0)
        The L2 regularization parameter. This is `lambda` in the algorithm.

    mixing_param : float (optional, default=1)
        The mixing parameter. Must be between 0 and 1. This is `beta` in the
        algorithm.

    Attributes
    ----------
    x_hist_ : list
        History of the previous accelerated iterates. 
        Maximum size = `window_size` + 1.

    gx_hist_ : list
        History of the previous function applications. These are 
        pre-accelerated iterates.
        Maximum size = `window_size` + 1.

    residuals_hist_ : list
        History of the previous residuals.
        Maximum size = `window_size` + 1.

    param_shape_ : tuple
        Shape of the parameters, defined when the first iterate is applied.

    References
    ----------
    [1] T. D. Nguyen, A. R. Balef, C. T. Dinh, N. H. Tran, D. T. Ngo, 
        T. A. Le, and P. L. Vo. "Accelerating federated edge learning," 
        in IEEE Communications Letters, 25(10):3282–3286, 2021.
    [2] D. Scieur, A. d’Aspremont, and F. Bach, “Regularized nonlinear 
        acceleration,” in Advances in Neural Information Processing Systems,
        2016.
    [3] A. d’Aspremont, D. Scieur, and A. Taylor, “Acceleration methods,”
        arXiv preprint arXiv:2101.09545, 2021.
    [4] "Anderson Acceleration," Stanford University Convex Optimization Group.
        https://github.com/cvxgrp/aa

    Examples
    --------
    >>> # x is a d-dimensional numpy array, produced by applying
    >>> # the function g to the previous iterate x_{t-1}
    >>> acc = AndersonAccelerationModule(reg=1e-8)
    >>> x_acc = acc.apply(x)  # accelerate from x 
    """

    def __init__(self, window_size=5, reg=0, mixing_param=1.):
        window_size = int(window_size)
        assert window_size > 0, "Window size must be positive"
        self.window_size = int(window_size)

        assert reg >= 0, "Regularization parameter must be non-negative"
        self.reg = reg

        assert 0 <= mixing_param <= 1, "Mixing parameter must be between 0 and 1"
        self.mixing_param = mixing_param

        # History of function applications
        self.gx_hist_ = []

        # History of iterates
        self.x_hist_ = []

        # History of residuals
        self.residuals_hist_ = []

        # Shape of the parameters, defined when the first iterate is applied
        self.param_shape_ = None

    def apply(self, x):
        """Perform acceleration on an iterate.

        Parameters
        ----------
        x : numpy array
            The iterate to accelerate. This is the application of `g` to the
            previous iterate.

        Returns
        -------
        x_acc : numpy array
            The accelerated iterate of the same shape as `x`.
        """

        if len(self.x_hist_) <= 0:
            # First iteration, so no acceleration can be done
            self.x_hist_.append(x)
            self.param_shape_ = x.shape
            return x

        # Check the shape of the iterate
        assert x.shape == self.param_shape_, \
            "Iterate shape must be the same as the previous iterate"

        x_prev = self.x_hist_[-1]

        residual = x - x_prev
        self.residuals_hist_.append(residual)
        if len(self.residuals_hist_) > self.window_size + 1:
            self.residuals_hist_.pop(0)

        self.gx_hist_.append(x)
        if len(self.gx_hist_) > self.window_size + 1:
            self.gx_hist_.pop(0)

        # Solve for alpha: min ||alpha_i F_{t-i}||
        Ft = np.stack(self.residuals_hist_)  # shape = (m_t + 1, dim)
        RR = Ft @ Ft.T
        RR += self.reg * np.eye(RR.shape[0])
        try:
            RR_inv = LA.inv(RR)
            alpha = np.sum(RR_inv, 1)
        except LA.LinAlgError:
            # Singular matrix, so solve least squares instead
            alpha = LA.lstsq(RR, np.ones(Ft.shape[0]), -1)[0]
        # Normalize alpha
        alpha /= alpha.sum() + 1e-12

        # Extrapolate to get accelerated iterate
        if len(self.x_hist_) <= 0:
            x_acc = x
        else:
            x_acc = 0
            for alpha_i, x_i, Gx_i in zip(alpha, self.x_hist_, self.gx_hist_):
                x_acc += (1 - self.mixing_param) * alpha_i * x_i
                x_acc += self.mixing_param * alpha_i * Gx_i

        self.x_hist_.append(x_acc)
        if len(self.x_hist_) > self.window_size + 1:
            self.x_hist_.pop(0)

        return x_acc

    def reset_hist(self):
        """Empty the histories of iterates and residuals.
        """
        self.x_hist_ = []
        self.gx_hist_ = []
        self.residuals_hist_ = []
