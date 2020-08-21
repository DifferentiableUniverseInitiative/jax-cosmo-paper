import scipy.linalg
import numpy as onp


class ExtremeJumpError(RuntimeError):
    pass

class HMC:
    def __init__(self, log_post_and_gradient, covariance_estimate, epsilon, steps_per_iteration, limits, args=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        if args is None:
            args = []
        self.fun = log_post_and_gradient
        self.fun_args = args
        self.fun_kwargs = kwargs

        # The mid-point.
        # We use this to transform the boundary planes
        self.limits = onp.array(limits)
        self.x0 = onp.array([0.5*(lim[0] + lim[1]) for lim in limits])
        self.ndim = len(self.x0)

        # We use this to transform between the original space and the
        # transformed one.  We use this often enough that it's worth
        # doing the full transform here.
        # We transform the q coordinates instead of the p momenta,
        # since otherwise the velocities and the momenta can point
        # in different directions, and this messes up the reflection
        # in a way I couldn't figure out.
        self.L = onp.linalg.cholesky(covariance_estimate)
        self.Linv = onp.linalg.inv(self.L)

        # Integration parameters
        self.epsilon = epsilon
        self.steps_per_iteration = steps_per_iteration

        # We trace the accepted samples
        self.trace_logP = []
        self.trace = []
        self.trace_accept = []

        # As well as tracing the accepted samples it's also nice
        # to trace the integration paths between those samples.
        # Improvements to the algorithm make use of those.
        self.paths = []
        self.paths_logP = []

        # We need the boundary planes that correspond to the parameter
        # limits, transformed into our new coordinates.
        self.lower_bound_planes = []
        self.upper_bound_planes = []
        self.bound_normals = []

        for i in range(self.ndim):
            xb = self.x0.copy()

            # A point on the upper bound plane
            xb[i] = self.limits[i, 1]
            uq = self.x2q(xb)

            # A point on the lower bound plane
            xb[i] = self.limits[i, 0]
            lq = self.x2q(xb)

            # Normals transform with the inverse of the
            # point transform.  Forgetting this cost me a lot
            # of time, until I drew some parallelograms.
            n = onp.zeros(self.ndim)
            n[i] = 1.0
            m = self.L.T @ n
            # normalize
            m /= onp.linalg.norm(m)

            self.upper_bound_planes.append(uq)
            self.lower_bound_planes.append(lq)
            self.bound_normals.append(m)


    
    def get_u(self, q):
        """Compute the posterior and gradient from (and in) normalized coordinates"""
        # convert from normalized q coordinates to x
        x = self.q2x(q)
        # Get the posterior and gradient
        logP, grad_logP = self.fun(x, *self.fun_args, **self.fun_kwargs)
        # convert the gradient to the transformed coordinates, using the Jacobian.
        # we  don't have to convert logP because the coordinate transformation
        # is linear, so the change to P(x) is just a scaling, so the change to
        # logP is just a constant
        return -logP, -self.L.T @ grad_logP
        

    def integrate(self, q, p):
        U, gradU = self.get_u(q)
        # Hamiltonian at the start of the integration
        H0 = 0.5 * (p @ p) + U
        # I don't really know what symplectic means, but I know
        # this is it.
        for i in range(self.steps_per_iteration):
            # half-update momenta
            p = p - 0.5 * self.epsilon * gradU
            
            # full update q with half-updated p
            q, p = self.advance_points(q, p)
            
            # second half-update momenta
            U, gradU = self.get_u(q)
            p = p - 0.5 * self.epsilon * gradU
            
            # print out energy levels
            T = 0.5 * (p @ p)
            H = T + U
            dH = H - H0
            # print(f'U={U:.3f}   T={T:.3f}   H={H:.3f}   ΔH={dH:.3f}')
            # record a trace of the log_post, -U
            self.paths_logP.append(-U)
            # and the chain position
            x = self.q2x(q)
            self.paths.append(x)
        
        return p, q, U, H, H0
    
    
    def q2x(self, q):
        """Transform a vector in the space with diagonal mass to a parameter vector"""
        return self.L @ q + self.x0
    
    def x2q(self, x):
        """Transform a parameter vector to a vector in the space with diagonal mass"""
        return self.Linv @ (x - self.x0)
    
    def extreme_jump_check(self, q, p):
        """
        Check for jumps so extreme that it jumps more than twice the bounds.

        Should be rare for most cases, but when it does happen the reflection will stop working.
        Indicates bad cov estimate or epsilon too large.  Warning below.

        """
        q2 = q + self.epsilon * p
        x = self.q2x(q)
        low = self.limits[:, 0]
        high = self.limits[:, 1]
        upper = 2 * high - low
        lower = 2 * low - high
        return onp.any(x > upper) or onp.any(x < lower)
    

    def random_kick(self):
        """Generate a random momentum vector"""
        mu = onp.zeros(self.ndim)
        I = onp.eye(self.ndim)
        p = onp.random.multivariate_normal(mu, I)
        return p

    
    def sample(self, n, start=None):
        """
        Generate "n" samples from the distribution.

        Start from "start", if supplied.
        """
        if start is None:
            if self.trace:
                start = self.trace[-1]
                print(f"Starting at: {start}")
            else:
                raise ValueError("Must supply a starting point the first call to hmc.sample.")
        else:
            start = onp.array(start)
        
        q = self.x2q(start)
        self.n_accept = 0
        self.n_reject = 0
        for j in range(n):

            # Generate a new random momentum vector
            p = self.random_kick()

            try:
                # Integrate the trajectory along that path
                p, q_new, U, H, H0 = self.integrate(q, p)
            except ExtremeJumpError as err:
                # We catch a specific error - 
                print(f"Extreme jump rejected {err}")
                U = onp.inf
                H = onp.inf
                H0 = 0.0
            # For small enough epsilon these are identical.
            # The acceptance criterion arises from imperfect integration.
            deltaH = H - H0
            log_alpha = -deltaH
            alpha = onp.exp(log_alpha)
            p1 = onp.random.uniform()
            # Standard MH acceptance.
            accept = log_alpha > onp.log(p1)
            # Count and update point, and report
            if accept:
                self.n_accept += 1
                q = q_new
                print(f"Accept {j} alpha={alpha:.2f}  p={p1:.2f}  ΔH={deltaH:.3f}")
            else:
                self.n_reject += 1
                print(f"Reject {j} alpha={alpha:.2f}  p={p1:.2f}  Δh={deltaH:.3f}")
            # keep a trace of things
            x = self.q2x(q)
            self.trace.append(x)
            self.trace_logP.append(-U)
            self.trace_accept.append(accept)


    def first_boundary_crossing(self, q, p, done):
        """
        Locate the first point where the vector connecting
        q to q+p crosses a boundary of our parameter space.

        Report the time to that crossing, the crossing point,
        and which parameter it happened in.

        I better if I was cleverer this would be two lines long
        or something.
        """
        # quick check if we have not crossed any limits
        q_fin = q + self.epsilon * p
        x_fin = self.q2x(q_fin)
        # tmp skip
        ok = (x_fin > self.limits[:,0]) & (x_fin < self.limits[:,1])
        if onp.all(ok):
            return None
        
        # If there is a crossing we have to move to the transformed space
        # to work out where
        lam = onp.zeros(self.ndim)
        for i in range(self.ndim):
            uq = self.upper_bound_planes[i]
            lq = self.lower_bound_planes[i]
            m = self.bound_normals[i]

            # distance along p vector to upper bound plane
            lam[i] = (-(q - uq) @ m) / (p @ m)

            # if this is negative, instead find distance to lower bound plane
            if not lam[i] > 0:
                lam[i] = (-(q - lq) @ m) / (p @ m)

        i = lam.argmin()
        q_new = q + lam[i]* 0.99 * p
        # useful to let the user know where we crossed
        b = self.q2x(q + lam[i] * p)[i]
        print(f"Note: parameter {i} reflecting at boundary {b}")

        if i in done:
            raise ExtremeJumpError("Bounced off the same parameter twice.  Info: "
                                   f"i = {i}, lam[i] = {lam[i]}, b = {b} p = {p}, q = {q}")

        return q_new, lam[i], i

    def reflect(self, p, i):
        """
        Reflect momentum vector p in the boundary plane for parameter i.
        This works whether it's the upper or lower plane as long as they're
        perpendicular.
        """
        m = self.bound_normals[i]
        perpendicular = (m @ p) * m
        parallel = p - perpendicular
        # flip the perpendicular component
        perpendicular *= -1
        return parallel + perpendicular

    def advance_points(self, q, p):
        """
        Update the position vector q using momentum p,
        accounting for any reflections off the boundary of the space.
        """
        if self.extreme_jump_check(q, p):
            raise ExtremeJumpError("Extreme jumps (more than double the range) - check M, epsilon")
        
        p = p.copy()
        q = q.copy()
        
        t0 = 0
        done = {}
        # locate the first crossing of the bounds, if there is one.
        # if not this just returns None
        crossing = self.first_boundary_crossing(q, p, done)
        while (crossing is not None):
            q, t_c, i = crossing
            # Record how much time we have left
            t0 += t_c
            # Reflect momentum in plane i
            p = self.reflect(p, i)
            done[i] = crossing
            # We might cross multiple boundaries
            crossing = self.first_boundary_crossing(q, p, done)

        # any remaining jump, with the newly flipped momentum
        q = q + (self.epsilon - t0) * p

        return q, p
        
