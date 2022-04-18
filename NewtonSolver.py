# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 14:19:43 2022

@author: Preslav
"""
from SolverTypes import SolverTypes
import numpy as np
import numdifftools as nd
from numpy.linalg import pinv


class Solver:
    def __init__(self, system_equations, method, valid_roots, iteration_limit=100, tolerance=1e-12):
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
        if system_equations is None:
            raise ValueError("No system of equations is given")
        if not callable(system_equations):
            raise ValueError("No system of equations must be callable")
        self.system = system_equations
        if valid_roots is None:
            raise ValueError("No valid roots given")
        if not isinstance(valid_roots, np.ndarray):
            raise ValueError("The valid roots must be an ndarray")
        self.valid_roots = valid_roots

        self.system = system_equations
        self.jac_fun = nd.Jacobian(self.system)
        if method is None:
            raise ValueError("No method is given")
        if not isinstance(method, SolverTypes):
            raise ValueError("The method is not an enum")
        self.method = method

    def run(self, x0, c0=None):
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
            self.c0=c0
            return self.solve_enm()
        elif self.method == SolverTypes.NM:
            return self.solve_nm()
        else:
            print("No method selected")

    def p_func(self, x):
        v = x - self.c0
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
        for i in range(self.ITERATION_LIMIT):
            cnt += 1
            q = self.p_func(x)
            p = np.linalg.pinv(self.get_partial(x))
            step = np.matmul(q, p)
            x = x - step.flatten()
            # Termination criteria
            if self.stopping_criterion(step, x) > 0:
                return [i-1, cnt]
            # print("{cur_x} - {cnt}".format(cur_x=x.flatten(), cnt=cnt))
        return [-1, self.ITERATION_LIMIT]

    def solve_nm(self):
        cnt = 0
        x = self.x0
        for i in range(self.ITERATION_LIMIT):
            cnt += 1
            q = self.system(x)
            p = np.linalg.pinv(self.jac_fun(x))
            step = np.matmul(q, p)
            x = x - step.flatten()
            # Termination criteria
            if self.stopping_criterion(step, x) > 0:
                return [i - 1, cnt]
            # print("{cur_x} - {cnt}".format(cur_x=x.flatten(), cnt=cnt))
        return [-1, self.ITERATION_LIMIT]

    def stopping_criterion(self, step, x):
        """
        stopping criterion taken from "Iterative methods of order four and five for systems of
        nonlinear equations" - Alicia Cordero
        """
        if (np.linalg.norm(step)+np.linalg.norm(self.system(x)))< self.TOLERANCE:
            for i,ii in enumerate(np.linalg.norm(self.valid_roots-x,axis=1)):
                if ii<self.TOLERANCE:
                    print(i+1)
                    return i+1
        return 0

