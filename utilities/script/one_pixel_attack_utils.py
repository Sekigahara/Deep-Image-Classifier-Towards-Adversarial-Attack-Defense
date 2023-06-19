"""Original code adapted from
https://github.com/scipy/scipy/blob/70e61dee181de23fdd8d893eaa9491100e2218d7/scipy/optimize/_differentialevolution.py


differential_evolution: The differential evolution global optimization algorithm
Added by Andrew Nelson 2014

This code adapted to my project from this repo https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/differential_evolution.py
credit to Hyperparticle
"""

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.optimize.optimize import _status_message
from scipy._lib._util import check_random_state
from scipy._lib.six import xrange, string_types
import warnings

__all__ = ['differential_evolution']

_MACHEPS = np.finfo(np.float64).eps

def differential_evolution_call(func, bounds, args=(), strategy='best1bin',
                           maxiter=1000, popsize=15, tol=0.01,
                           mutation=(0.5, 1), recombination=0.7, seed=None,
                           callback=None, disp=False, polish=True,
                           init='latinhypercube', atol=0):
    


    return 0

class DifferentialEvolution(object):
    # Dispatch of mutation strategy method (binomial or exponential).
    binomial = {'best1bin': '_best1',
                'randtobest1bin': '_randtobest1',
                'currenttobest1bin': '_currenttobest1',
                'best2bin': '_best2',
                'rand2bin': '_rand2',
                'rand1bin': '_rand1'}
    exponential = {'best1exp': '_best1',
                    'rand1exp': '_rand1',
                    'randtobest1exp': '_randtobest1',
                    'currenttobest1exp': '_currenttobest1',
                    'best2exp': '_best2',
                    'rand2exp': '_rand2'}

    init_error_msg = ("The population initialization method must be one of "
                        "'latinhypercube' or 'random', or an array of shape "
                        "(M, N) where N is the number of parameters and M>5")
        
    def __init__(self, func, bounds, args=(), strategy='best1bin', 
                 maxiter=1000, popsize=15, tolerance=0.01, mutation=(0.5, 1), 
                 recombination=0.7, seed=None, callback=None, disp=False, 
                 polish=True, init='latinhypercube', abs_tolerance=0):
        
        

        if strategy in self.binomial:
            self.mutation_func = getattr(self, self._binomial[strategy])
        elif strategy in self.exponential:
            self.mutation_func = getattr(self, self._exponential[strategy])
        else:
            print("Select a valid mutation strategy")

        # Initializes Setup
        self.strategy = strategy
        self.callback = callback
        self.polish = polish

        # Relative and absolute tolerances for child convergences
        self.tol, self.atol = tolerance, abs_tolerance

        # Mutation constant is [0, 2]
        self.scale = mutation
        if (not np.all(np.isfinite(mutation)) or
                np.any(np.array(mutation) >= 2) or
                np.any(np.array(mutation) < 0)):
            print("The mutation constant must be a float in ''[0, 2], or specified as a tuple(min, max)'' where min < max and min, max are in [0, 2].")
        
        self.dither = None
        if hasattr(mutation, '__iter__') and len(mutation) > 1:
            self.dither = [mutation[0], mutation[1]]
            self.dither.sort()

        # Initiate cross over prob
        self.cross_over_probability = recombination

        # Initiate function argument
        self.func = func
        self.args = args

        self.limits = np.array(bounds, dtype='float').T
        if (np.size(self.limits, 0) != 2 or not np.all(np.isfinite(self.limits))):
            print("bounds should be a sequence containing ''real valued (min, max) pairs for each value'' in x")

        if maxiter is None:
            maxiter = 1000
        self.maxiter = maxiter
        if maxfun is None:
            maxfun = np.inf
        self.maxfun = maxfun

        # population is scaled to between [0, 1].
        # We have to scale between parameter <-> population
        # save these arguments for _scale_parameter and
        # _unscale_parameter. This is an optimization
        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.parameter_count = np.size(self.limits, 1)

        self.random_number_generator = check_random_state(seed)

        # default population initialization is a latin hypercube design, but
        # there are other population initializations possible.
        # the minimum is 5 because 'best2bin' requires a population that's at
        # least 5 long
        self.num_population_members = max(5, popsize * self.parameter_count)
        self.population_shape = (self.num_population_members, self.parameter_count)

        self._nfev = 0
        if isinstance(init, string_types):
            if init == 'latinhypercube':
                self.init_population_lhs()
            elif init == 'random':
                self.init_population_random()
            else:
                print(self.init_error_msg)
        else:
            self.init_population_array(init)

        self.disp = disp

    def reset(self):
        self.population_energies = (np.ones(self.num_population_members) * np.inf)
        self._nfev = 0

    def init_population_lhs(self):
        """
        Initializes the population with Latin Hypercube Sampling.
        Latin Hypercube Sampling ensures that each parameter is uniformly
        sampled over its range.
        """
        rng = self.random_number_generator

        # Sample uniform random distribution to sampling parameter
        # Offset each segment to cover the entire parameter with range 0 to 1
        segsize = 1.0 / self.num_population_members
        samples = (segsize * rng.random_sample(self.population_shape) + np.linspace(0., 1., self.num_population_members, endpoint=False)[:, np.newaxis])
        
        self.population = np.zeros_like(samples)

        # Init population of canditate solution by permutation of random samples
        for j in range(self.parameter_count):
            order = rng.permutation(range(self.num_population_members))
            self.population[:, j] = samples[order, j]

        # Reset population energies and function eval counter
        self.reset(self)

    def init_population_random(self):
        """
        Initialises the population at random.  This type of initialization
        can possess clustering, Latin Hypercube sampling is generally better.
        """
        rng = self.random_number_generator
        self.population = rng.random_sample(self.population_shape)

        # Reset population energies and function eval counter
        self.reset(self)

    def init_population_array(self, init):
        """
        Initialises the population with a user specified population.
        Parameters
        ----------
        init : np.ndarray
            Array specifying subset of the initial population. The array should
            have shape (M, len(x)), where len(x) is the number of parameters.
            The population is clipped to the lower and upper `bounds`.
        """

        popn = np.asfarray(init)

        if (np.size(popn, 0) < 5 or popn.shape[1] != self.parameter_count or len(popn.shape) != 2):
            print("The population supplied needs to have shape (M, len(x)), where M > 4.")

        # Scale values and clip to bounds then assign to population
        self.population = np.clip(self._unscale_parameters(popn), 0, 1)
        self.num_population_members = np.size(self.population, 0)

        self.population_shape = (self.num_population_members, self.parameter_count)

        # Reset population energies and function eval counter
        self.reset(self)

    @property
    def x(self):
        """
            The best solution from the solver
            Returns
            -------
            x : ndarray
                The best solution from the solver.
        """
        return self._scale_parameters(self.population[0])
    
    @property
    def convergence(self):
        """
        The standard deviation of the population energies divided by their
        mean.
        """
        return (np.std(self.population_energies) / np.abs(np.mean(self.population_energies) + _MACHEPS))
    
    def solve(self):
        """
        Runs the DifferentialEvolutionSolver.
        Returns
        -------
        res : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.  If `polish`
            was employed, and a lower minimum was obtained by the polishing,
            then OptimizeResult also contains the ``jac`` attribute.
        """
        nlt, warning_flag = 0, False
        _status_message = _status_message['success']

        if np.all(np.isinf(self.population_energies)):
            self._calculate_population_energies()

        # perform optimisation.
        for nit in xrange(1, self.maxiter + 1):
            # evolve the population by a generation
            try:
                next(self)
            except StopIteration:
                warning_flag = True
                status_message = _status_message['maxfev']
                break

            if self.disp:
                print("differential_evolution step %d: f(x)= %g" % (nit, self.population_energies[0]))

            convergences = self.convergence

            
