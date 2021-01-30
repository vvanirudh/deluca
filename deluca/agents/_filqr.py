import jax.numpy as jnp

from deluca.agents.core import Agent

class FiniteLQR(Agent):
    def __init__(self,
                 x_0: jnp.ndarray,
                 H: jnp.int32,
                 A: jnp.ndarray,
                 B: jnp.ndarray,
                 Q: jnp.ndarray = None,
                 R: jnp.ndarray = None,
                 Qf: jnp.ndarray = None) -> None:
        
        self.state_size, self.action_size = B.shape
        self.H = H
        self.x_0 = x_0  # initial state

        self.Q = Q if Q is not None else jnp.identity(self.state_size, dtype=jnp.float32)
        self.R = R if R is not None else jnp.identity(self.action_size, dtype=jnp.float32)
        self.Qf = Qf if Qf is not None else jnp.identity(self.state_size, dtype=jnp.float32)

        self.K, self.traj, self.U = self.solve()

    def solve(self):
        # Backward pass - compute P_t
        H, A, B = self.H, self.A, self.B
        Q, R, Qf = self.Q, self.R, self.Qf
        
        P = [None for _ in range(H+1)]
        P[H] = Qf
        for t in range(H-1, -1, -1):
            P[t] = Q + A.T @ P[t+1] @ A - A.T @ P[t+1] @ B @ jnp.linalg.inv(R + B.T @ P[t+1] @ B) @ (B.T @ P[t+1] @ A)

        # Forward pass - compute K_t
        X = [None for _ in range(H)]
        U = [None for _ in range(H)]
        X[0] = self.x_0
        K = [None for _ in range(H)]
        for t in range(H):
            K[t] = -jnp.linalg.inv(R + B.T @ P[t+1] @ B) @ B.T @ P[t+1] @ A
            U[t] = K[t] @ X[t]
            X[t+1] = A @ X[t] + B @ U[t]

        return K, X, U

    def __call__(self, state, t) -> jnp.ndarray:
        assert t >= 0 and t < self.H, "Invalid time index"
        return -self.K[t] @ state
