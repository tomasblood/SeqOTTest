"""
Forward-Backward Sinkhorn Algorithm for Global SeqOT
Based on Algorithm 1 in Watanabe & Isobe (2024/2025)
"""

import numpy as np
from scipy.special import logsumexp
from .utils import safe_log, safe_exp, normalize_distribution, log_row_sums, log_col_sums


class ForwardBackwardSinkhorn:
    """
    Forward-Backward Sinkhorn solver for chain-structured multi-marginal OT.

    This implements Algorithm 1 from the paper, which solves the global
    optimization problem:

        min sum_{t=1}^{N-1} <P^(t), C^(t)> - epsilon * H(P^(t))

    subject to:
        - P^(1) @ 1 = mu (fixed source)
        - P^(N-1)^T @ 1 = nu (fixed target)
        - P^(t)^T @ 1 = P^(t+1) @ 1 for t=1..N-2 (flow conservation)

    Parameters
    ----------
    epsilon : float, default=0.01
        Entropic regularization parameter
    max_iter : int, default=1000
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance
    verbose : bool, default=False
        Print convergence information
    """

    def __init__(self, epsilon=0.01, max_iter=1000, tol=1e-6, verbose=False):
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Will be set during fit
        self.log_u_ = None
        self.log_v_ = None
        self.log_K_ = None
        self.n_steps_ = None
        self.converged_ = False
        self.n_iter_ = 0

    def fit(self, cost_matrices, mu=None, nu=None):
        """
        Solve the global SeqOT problem.

        Parameters
        ----------
        cost_matrices : list of np.ndarray
            List of cost matrices [C^(1), C^(2), ..., C^(N-1)]
            C^(t) has shape (n_t, n_{t+1})
        mu : np.ndarray, shape (n_1,), optional
            Source distribution. If None, uses uniform.
        nu : np.ndarray, shape (n_N,), optional
            Target distribution. If None, uses uniform.

        Returns
        -------
        self : ForwardBackwardSinkhorn
            Fitted instance
        """
        self.n_steps_ = len(cost_matrices)

        # Compute log kernels: K^(t) = exp(-C^(t) / epsilon)
        self.log_K_ = [safe_log(np.exp(-C / self.epsilon)) for C in cost_matrices]

        # Initialize source and target distributions
        if mu is None:
            mu = np.ones(cost_matrices[0].shape[0]) / cost_matrices[0].shape[0]
        else:
            mu = normalize_distribution(mu)

        if nu is None:
            nu = np.ones(cost_matrices[-1].shape[1]) / cost_matrices[-1].shape[1]
        else:
            nu = normalize_distribution(nu)

        log_mu = safe_log(mu)
        log_nu = safe_log(nu)

        # Initialize dual variables (in log domain)
        self.log_u_ = [np.zeros(self.log_K_[t].shape[0]) for t in range(self.n_steps_)]
        self.log_v_ = [np.zeros(self.log_K_[t].shape[1]) for t in range(self.n_steps_)]

        # Main iteration loop
        for iteration in range(self.max_iter):
            log_u_old = [u.copy() for u in self.log_u_]
            log_v_old = [v.copy() for v in self.log_v_]

            # FORWARD SWEEP: Update u from left to right
            self._forward_sweep(log_mu)

            # BACKWARD SWEEP: Update v from right to left
            self._backward_sweep(log_nu)

            # Check convergence
            u_diff = max(np.max(np.abs(self.log_u_[t] - log_u_old[t]))
                        for t in range(self.n_steps_))
            v_diff = max(np.max(np.abs(self.log_v_[t] - log_v_old[t]))
                        for t in range(self.n_steps_))
            error = max(u_diff, v_diff)

            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}: error = {error:.2e}")

            if error < self.tol:
                self.converged_ = True
                self.n_iter_ = iteration + 1
                if self.verbose:
                    print(f"Converged after {self.n_iter_} iterations")
                break
        else:
            if self.verbose:
                print(f"Did not converge after {self.max_iter} iterations")
            self.n_iter_ = self.max_iter

        return self

    def _forward_sweep(self, log_mu):
        """
        Forward sweep: Update u vectors from left to right.

        At t=1: Fix source marginal to mu
        At t>1: Match incoming flow from previous step
        """
        # Step 1: Update u^(1) to match source mu
        # We want: diag(u^(1)) @ K^(1) @ diag(v^(1)) @ 1 = mu
        # Row sums should equal mu
        current_row_sums = log_row_sums(self.log_u_[0], self.log_K_[0], self.log_v_[0])
        self.log_u_[0] = self.log_u_[0] + log_mu - current_row_sums

        # Steps 2 to N-1: Match incoming flow
        for t in range(1, self.n_steps_):
            # Compute mass arriving from step t-1
            # mass_in = v^(t-1) ⊙ (K^(t-1)^T @ u^(t-1))
            # In log domain:
            log_Kt_T_u = logsumexp(
                self.log_u_[t-1][:, None] + self.log_K_[t-1],
                axis=0
            )
            log_mass_in = self.log_v_[t-1] + log_Kt_T_u

            # Update u^(t) so row sums match mass_in
            current_row_sums = log_row_sums(self.log_u_[t], self.log_K_[t], self.log_v_[t])
            self.log_u_[t] = self.log_u_[t] + log_mass_in - current_row_sums

    def _backward_sweep(self, log_nu):
        """
        Backward sweep: Update v vectors from right to left.

        At t=N-1: Fix target marginal to nu
        At t<N-1: Match outgoing flow to next step
        """
        # Step N-1: Update v^(N-1) to match target nu
        # We want: diag(u^(N-1)) @ K^(N-1) @ diag(v^(N-1)) @ 1^T = nu
        # Column sums should equal nu
        current_col_sums = log_col_sums(
            self.log_u_[self.n_steps_-1],
            self.log_K_[self.n_steps_-1],
            self.log_v_[self.n_steps_-1]
        )
        self.log_v_[self.n_steps_-1] = (
            self.log_v_[self.n_steps_-1] + log_nu - current_col_sums
        )

        # Steps N-2 down to 1: Match required outgoing flow
        for t in range(self.n_steps_ - 2, -1, -1):
            # Compute demand from step t+1
            # mass_req = u^(t+1) ⊙ (K^(t+1) @ v^(t+1))
            # In log domain:
            log_K_v = logsumexp(
                self.log_K_[t+1] + self.log_v_[t+1][None, :],
                axis=1
            )
            log_mass_req = self.log_u_[t+1] + log_K_v

            # Update v^(t) so column sums match mass_req
            current_col_sums = log_col_sums(self.log_u_[t], self.log_K_[t], self.log_v_[t])
            self.log_v_[t] = self.log_v_[t] + log_mass_req - current_col_sums

    def get_couplings(self):
        """
        Recover the transport plan matrices P^(t).

        Returns
        -------
        couplings : list of np.ndarray
            List of transport matrices [P^(1), ..., P^(N-1)]
        """
        if self.log_u_ is None:
            raise RuntimeError("Must call fit() before get_couplings()")

        couplings = []
        for t in range(self.n_steps_):
            # P^(t) = diag(u^(t)) @ K^(t) @ diag(v^(t))
            # In log domain: log(P^(t)_ij) = log_u^(t)_i + log_K^(t)_ij + log_v^(t)_j
            log_P = (
                self.log_u_[t][:, None] +
                self.log_K_[t] +
                self.log_v_[t][None, :]
            )
            P = safe_exp(log_P)
            couplings.append(P)

        return couplings

    def compute_objective(self, cost_matrices):
        """
        Compute the objective value of the current solution.

        Returns
        -------
        objective : float
            Total cost + regularization
        """
        couplings = self.get_couplings()
        objective = 0.0

        for t, (P, C) in enumerate(zip(couplings, cost_matrices)):
            # Transport cost
            transport_cost = np.sum(P * C)

            # Entropy (avoiding log(0))
            entropy = -np.sum(P * safe_log(P + 1e-100))

            objective += transport_cost - self.epsilon * entropy

        return objective
