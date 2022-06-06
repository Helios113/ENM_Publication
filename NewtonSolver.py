# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:19:43 2022

@author: Preslav
"""
from SolverTypes import SolverTypes
import numpy as np
import numdifftools as nd
from numpy.linalg import pinv, LinAlgError
import scipy.optimize as opt
import warnings


class Solver:
    def __init__(self, system_equations, method, valid_roots=None, iteration_limit=100, tolerance=1e-5):
        """
        :param system_equations:
        :param method:
        :param iteration_limit:
        :param tolerance: default is set at 1e-12
        """
        self.ITERATION_LIMIT = iteration_limit
        self.TOLERANCE = tolerance
        self.x0 = None
        self.c0 = None
        self.error = []
        if system_equations is None:
            raise ValueError("No system of equations is given")
        if not callable(system_equations):
            raise ValueError("No system of equations must be callable")
        self.system = system_equations
        if valid_roots is None:
            warnings.warn("No roots given, system will check root manually")
        elif not isinstance(valid_roots, np.ndarray):
            raise ValueError("The valid roots must be an ndarray")
        self.valid_roots = valid_roots

        self.system = system_equations
        self.jac_fun = nd.Jacobian(self.system)
        if method is None:
            raise ValueError("No method is given")
        if not isinstance(method, SolverTypes):
            raise ValueError("The method is not an enum")
        self.method = method

    def run(self, x0, c0=None, continually_change=False, change_first=False):
        self.error = []
        self.continually_change = continually_change
        self.change_first = change_first
        if x0 is None:
            raise ValueError("No initial values are given")
        if not isinstance(x0, np.ndarray):
            raise ValueError("The initial value must be an ndarray")
        self.x0 = x0
        if self.method == SolverTypes.ENM:
            if c0 is None:
                raise ValueError("No second initial values are given")
            if not isinstance(c0, np.ndarray):
                raise ValueError("The second initial value must be an ndarray")
            self.c0 = c0
            return self.solve_enm()
        elif self.method == SolverTypes.NM:
            return self.solve_nm()
        elif self.method == SolverTypes.Powell:
            return self.solve_powell()
        elif self.method == SolverTypes.LM:
            return self.solve_lm()
        elif self.method == SolverTypes.Broyden:
            return self.solve_broyden()
        elif self.method == SolverTypes.EBM:
            self.c0=c0
            return self.solve_ebm()
        elif self.method == SolverTypes.ELM:
            self.c0=c0
            return self.solve_elm()
        else:
            print("No method selected")

    def p_func(self, x):
        v = x - self.c0
        # print(self.system(x))
        # print(self.system(self.c0))

        g = self.system(x) / (self.system(x) - self.system(self.c0))
        return np.einsum("i,j->ij", v, g).reshape(1, -1)  # column vector

    def get_partial(self, x):
        v = x - self.c0
        g = self.system(x) / (self.system(x) - self.system(self.c0))  # same as G

        h = (-self.system(self.c0) / ((self.system(x) - self.system(self.c0)) ** 2)).reshape(-1, 1)
        w = (np.einsum('ik,j->ijk', np.identity(len(g)), g) +
             np.einsum('i,jk->ijk', v, self.jac_fun(x) * h))
        w = w.reshape(len(x) ** 2, len(x))
        return w.T

    def solve_enm(self):
        cnt = 0
        x = self.x0
        if self.change_first:
            self.c0 = x + 0.1
        for i in range(self.ITERATION_LIMIT):
            cnt += 1
            try:
                q = self.p_func(x)
                p = np.linalg.pinv(self.get_partial(x))
            except:
                return [-1, self.ITERATION_LIMIT]
            step = np.matmul(q, p)

            x = x - step.flatten()
            if self.continually_change:
                self.c0 = x + 0.1
            self.error_record(x)
            # Termination criteria
            root = self.stopping_criterion(step, x)
            if root > 0:
                return [root - 1, cnt]

        return [-1, self.ITERATION_LIMIT]

    def solve_nm(self):
        cnt = 0
        x = self.x0
        for i in range(self.ITERATION_LIMIT):
            cnt += 1
            # Implement NR method
            f = self.system(x)
            try:
                j = np.linalg.inv(self.jac_fun(x))
            except LinAlgError:
                return [-1,0]

            step = j.dot(f)
            x = x - step.flatten()
            self.error_record(x)
            # Termination criteria
            root = self.stopping_criterion(step, x)
            if root > 0:
                return [root - 1, cnt]
            # print("{cur_x} - {cnt}".format(cur_x=x.flatten(), cnt=cnt))
        return [-1, self.ITERATION_LIMIT]

    def solve_powell(self):
        cnt = 0
        x = self.x0
        try:
            sol = opt.root(self.system, x, method='hybr', jac=self.jac_fun,
                       options={'maxfev': self.ITERATION_LIMIT, 'xtol': self.TOLERANCE})
        except:
            return [-1, self.ITERATION_LIMIT]
        if sol.success:
            ret = self.stopping_criterion(0, sol.x)
            if ret > 0:
                return [ret - 1, sol.nfev]
        return [-1, self.ITERATION_LIMIT]

    def solve_lm(self):
        cnt = 0
        x = self.x0
        try:
            sol = opt.root(self.system, x, method='lm', jac=self.jac_fun,
                       options={'maxiter': self.ITERATION_LIMIT, 'xtol': self.TOLERANCE})
        except:
            return [-1, self.ITERATION_LIMIT]
        if sol.success:
            ret = self.stopping_criterion(0, sol.x)
            if ret > 0:
                return [ret - 1, sol.nfev]
        return [-1, self.ITERATION_LIMIT]

    def solve_broyden(self):
        cnt = 0
        x = self.x0
        try:
            sol = opt.root(self.system, x, method="broyden1",
                           options={'maxiter': self.ITERATION_LIMIT, 'xtol': self.TOLERANCE})
        except:
            return [-1, self.ITERATION_LIMIT]
        if sol.success:
            ret = self.stopping_criterion(0, sol.x)
            if ret > 0:
                return [ret - 1, sol.nit]
        return [-1, self.ITERATION_LIMIT]

    def solve_ebm(self):
        #Extended broyden method
        #Applying broyden on P(x)r(x)
        cnt = 0
        x = self.x0
        try:
            sol = opt.root(self.p_func, x, method="broyden1",
                           options={'maxiter': self.ITERATION_LIMIT, 'xtol': self.TOLERANCE})
        except:
            return [-1, self.ITERATION_LIMIT]
        if sol.success:
            ret = self.stopping_criterion(0, sol.x)
            if ret > 0:
                return [ret - 1, sol.nit]
        return [-1, self.ITERATION_LIMIT]
    def solve_elm(self):
        cnt = 0
        x = self.x0
        try:
            sol = opt.root(self.p_func, x, method='lm', jac=self.jac_fun,
                       options={'maxiter': self.ITERATION_LIMIT, 'xtol': self.TOLERANCE})
        except:
            return [-1, self.ITERATION_LIMIT]
        if sol.success:
            ret = self.stopping_criterion(0, sol.x)
            if ret > 0:
                return [ret - 1, sol.nfev]
        return [-1, self.ITERATION_LIMIT]

    def stopping_criterion(self, step, x):
        """
        implements xtol from scipy: The calculation will terminate if the relative error between two consecutive
        iterates is at most xtol.
        """
        if np.linalg.norm(step) <= self.TOLERANCE:
            if self.valid_roots is not None:
                for i, ii in enumerate(np.linalg.norm(self.valid_roots - x, axis=1)):
                    if ii < 1e-5:
                        return i + 1
                print("Potentially false root is:", x)
                print(self.system(x))
            else:
                if np.linalg.norm(self.system(x)) < 1e-10:
                    return 1
        return 0

    def error_record(self,x):
        self.error.append(x)
