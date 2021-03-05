import numpy as np
import osqp
from scipy import sparse
import matplotlib.pyplot as plt

##################################################################
#                                                                #
#   Based on Multi-Purpose MPC from Mats Steinweg, ZTH           #
#   Github: https://github.com/matssteinweg/Multi-Purpose-MPC    #
#                                                                #
##################################################################

# Colors
PREDICTION = "#BA4A00"


class MPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints, ay_max):

        # parameters
        self.N = N
        self.Q = Q
        self.R = R
        self.QN = QN

        self.model = model
        self.nx = self.model.n_states
        self.nu = 2

        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints

        # maximal lateral acceleration
        self.ay_max = ay_max

        self.current_prediction = None

        self.infeasibility_counter = 0

        self.current_control = np.zeros((self.nu * self.N))

        # solver for QP problem
        self.optimizer = osqp.OSQP()

    def init_problem(self):

        # constraints
        umin = self.input_constraints["umin"]
        umax = self.input_constraints["umax"]
        xmin = self.state_constraints["xmin"]
        xmax = self.state_constraints["xmax"]

        # LTV systems
        # huge matrices that consist of all matrices in a given horizon (N)
        A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
        B = np.zeros((self.nx * (self.N + 1), self.nu * (self.N)))

        x_ref = np.zeros(self.nx * (self.N + 1))
        u_ref = np.zeros(self.nu * self.N)

        # offset for equality constraints
        u_eq = np.zeros(self.N * self.nx)

        # dynamic state and input constraints
        xmin_dyn = np.kron(np.ones(self.N + 1), xmin)
        xmax_dyn = np.kron(np.ones(self.N + 1), xmax)
        umax_dyn = np.kron(np.ones(self.N), umax)

        # derive predicted curvature from last control
        # kappa = tan(delta) / length
        kappa_pred = (
            np.tan(
                np.array(self.current_control[3::] + self.current_control[-1::]))
            / self.model.length
        )

        # fill the information over entire horizon
        for n in range(self.N):

            # extract information from current waypoint
            current_waypoint = self.model.reference_path.get_waypoint(
                self.model.wp_id + n
            )
            next_waypoint = self.model.reference_path.get_waypoint(
                self.model.wp_id + n + 1
            )

            # Previous reference output
            # delta_s = next_waypoint - current_waypoint
            # kappa_ref = current_waypoint.kappa

            v_ref = current_waypoint.v_ref
            psi_ref = current_waypoint.psi
            # TODO: reference values below haven't created yet
            delta_ref = current_waypoint.delta

            # get linearized LTV model
            # TODO: different model, different states
            A_lin, B_lin = self.model.linearize(v_ref, psi_ref, delta_ref)

            A[
                (n + 1) * self.nx: (n + 2) * self.nx, n * self.nx: (n + 1) * self.nx
            ] = A_lin
            B[
                (n + 1) * self.nx: (n + 2) * self.nx, n * self.nu: (n + 1) * self.nu
            ] = B_lin

            # TODO: 2 inputs have changed
            u_ref[n * self.nu: (n + 1) *
                  self.nu] = np.array([v_ref, delta_ref])

            # compute equality constraint offset
            # TODO: dunno what this is
            u_eq[n * self.nx: (n + 1) * self.nx] = B_lin.dot(
                np.array([v_ref, delta_ref])
            )

            # maximum speed limited by predicted car curvature
            vmax_dyn = np.sqrt(self.ay_max / (np.abs(kappa_pred[n] + 1e-12)))
            if vmax_dyn < umax_dyn[self.nu * n]:
                umax_dyn[self.nu * n] = vmax_dyn

        # compute dynamic constraints on states
        upp_b, low_b, _ = self.model.reference_path.update_path_constraints(
            self.model.wp_id + 1,
            self.N,
            2 * self.model.safety_margin,
            self.model.safety_margin,
        )
        xmin_dyn[0] = self.model.temporal_state.x  # TODO
        xmax_dyn[0] = self.model.temporal_state.x  # TODO
        xmin_dyn[self.nx:: self.nx] = low_b
        xmax_dyn[self.nx:: self.nx] = upp_b

        # reference state = middle line of free space
        x_ref[self.nx:: self.nx] = (low_b + upp_b) / 2

        """
        form an QP problem using the format of OSQP
        we need P, q, A, l, u
        for more infromation, visit https://osqp.org/docs/examples/setup-and-solve.html
        """

        ################################
        # P: matrix for quadratic term #
        # q: matrix for linear term    #
        ################################

        P = sparse.block_diag(
            [
                sparse.kron(sparse.eye(self.N), self.Q),
                self.QN,
                sparse.kron(sparse.eye(self.N), self.R),
            ],
            format="csc",
        )

        # TODO: dunno how to construct
        q = np.hstack(
            [
                -np.tile(np.diag(self.Q.A), self.N) * x_ref[: -self.nx],
                -self.QN.dot(x_ref[-self.nx:]),
                -np.tile(np.diag(self.R.A), self.N) * u_ref,
            ]
        )

        #################################
        # A: system matrix in osqp form #
        #################################

        # equality part
        Ax = sparse.kron(
            sparse.eye(self.N + 1), -sparse.eye(self.nx)
        ) + sparse.csc_matrix(A)

        Bu = sparse.csc_matrix(B)
        A_eq = sparse.hstack([Ax, Bu])

        # inequality matrix
        A_ineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)

        A = sparse.vstack([A_eq, A_ineq], format="csc")

        #####################################
        # l, u: lower bound and upper bound #
        #####################################

        # equality part: just for the format
        x0 = np.array(self.model.temporal_state[:])
        low_eq = np.hstack([-x0, u_eq])
        upp_eq = low_eq  # format reason

        # inequality part
        low_ineq = np.hstack([xmin_dyn, np.kron(np.ones(self.N), umin)])
        upp_ineq = np.hstack([xmax_dyn, umax_dyn])

        # results
        l = np.hstack([low_eq, low_ineq])
        u = np.hstack([upp_eq, upp_ineq])

        #####################################
        #       Solve the problem           #
        #####################################

        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)

    def get_control(self):

        self.model.get_current_waypoint()

        # create the optimization problem and solve
        self.init_problem()
        res = self.optimizer.solve()

        # Check solver status
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        try:
            command = np.array(res.x[-self.N * self.nu:])
            command[1::2] = np.arctan(command[1::2] *
                                      self.model.length)
            v, delta = command[0], command[1]
            ctrl = np.array([v, delta])

            # update the control input
            self.current_control = command

            # update prediction (x,y coordinates)
            x = np.reshape(res.x[: (self.N + 1) * self.nx],
                           (self.N + 1, self.nx))
            print("x: ", x)
            self.current_prediction = self.update_prediction(x)

            # reset infeasibility counter if problem is solved
            self.infeasibility_counter = 0

        except:

            print("Infeasible problem, use last predicited control input")
            id = self.nu * (self.infeasibility_counter + 1)
            ctrl = np.array(self.current_control[id: id + 2])

            self.infeasibility_counter += 1

        if self.infeasibility_counter == (self.N - 1):
            print("No control input computed")
            exit(1)

        return ctrl

    def update_prediction(self, state_prediction):
        """
        output lists of coordinate x and y for visualization
        """
        x_pred, y_pred = [], []

        for n in range(2, self.N):
            matched_waypoint = self.model.reference_path.get_waypoint(
                self.model.wp_id + n
            )

            # TODO since we don't need to tranform from spatial to temporal, need change
            x_pred.append(matched_waypoint.x)
            y_pred.append(matched_waypoint.y)

        return x_pred, y_pred

    def show_prediction(self):
        """
        dispaly predicited car trajectory in current axis
        """

        if self.current_prediction is not None:
            plt.scatter(
                self.current_prediction[0],
                self.current_prediction[1],
                c=PREDICTION,
                s=30,
            )
