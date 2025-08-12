# %%
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import optuna

from sklearn.model_selection import TimeSeriesSplit

SEED = 42
random.seed(SEED)
np.random.seed(SEED)         

# %% [markdown]
# # Dataset generation

# %%
def generate_machen(length, dt=0.01, alpha=3, beta=0.1, gamma=1,noise=0.0, seed=42, initial_state=None):
    
    """
    Generate a trajectory of the Ma-Chen financial system.

    Args:
        length         (int):      Number of time steps to generate.
        dt             (float):    Time step size.
        alpha          (float):    Parameter alpha (savings rate).
        beta           (float):    Parameter beta (investment cost).
        gamma          (float):    Parameter gamma (demand elasticity).
        noise          (float):    Std-dev of additive Gaussian noise on each variable.
        seed           (int/None): Random seed for reproducibility.
        initial_state  (array-like or None): Initial (u1, u2, u3). Random if None.

    Returns:
        t -> time vector
        traj -> the Ma–Chen system trajectory at each time.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize state
    if initial_state is None:
        state = rng.standard_normal(3)
    else:
        state = np.array(initial_state, dtype=float)
    
    t = np.linspace(0, dt*(length-1), length)
    traj = np.zeros((length, 3))
    traj[0] = state

    def machen_rhs(s, alpha=3, beta=0.1, gamma=1):
        u1, u2, u3 = s
        du1 = u3 + (u2 - alpha) * u1
        du2 = 1 - beta * u2 - u1**2
        du3 = -u1 - gamma * u3
        return np.array([du1, du2, du3])

    for i in range(1, length):
        s = traj[i-1]
        k1 = machen_rhs(s)
        k2 = machen_rhs(s + 0.5 * dt * k1)
        k3 = machen_rhs(s + 0.5 * dt * k2)
        k4 = machen_rhs(s + dt * k3)
        traj[i] = s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # Add Gaussian noise if requested on measurements
    if noise > 0:
        traj += rng.normal(scale=noise, size=traj.shape)

    return t, traj


def machen_rhs(state, alpha, beta, gamma):
   
    state = np.asarray(state)
    if state.ndim == 2:
        u1 = state[:, 0]
        u2 = state[:, 1]
        u3 = state[:, 2]
        du1 = u3 + (u2 - alpha) * u1
        du2 = 1 - beta * u2 - u1**2
        du3 = -u1 - gamma * u3
        return np.stack((du1, du2, du3), axis=1)

    u1, u2, u3 = state
    du1 = u3 + (u2 - alpha) * u1
    du2 = 1 - beta * u2 - u1**2
    du3 = -u1 - gamma * u3
    return np.array([du1, du2, du3])


def nrmse(y_true, y_pred):
    
    num = np.linalg.norm(y_true - y_pred)
    den = np.linalg.norm(y_true)
    return num / den


def split_train_val_test(u, y, train_frac=0.6, val_frac=0.2):
    
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0")
    N = len(u)
    i_train = int(train_frac * N)
    i_val   = i_train + int(val_frac * N)

    return ((u[:i_train], y[:i_train]), (u[i_train:i_val], y[i_train:i_val]), (u[i_val:], y[i_val:]))

# %%
class ESN:
    def __init__(
        self,
        in_size: int,
        res_size: int,
        out_size: int,
        spectral_radius: float = 0.95,
        sparsity: float = 0.1,
        input_scaling: float = 1.0,
        leak_rate: float = 1.0,
        ridge_reg: float = 1e-8,
        seed: int = SEED,
        topology: str = "random"
    ):
        """
        Echo State Network (ESN) constructor.
        """
        self.in_size = in_size
        self.res_size = res_size
        self.out_size = out_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.ridge_reg = ridge_reg
        self.topology = topology

        self.rng = np.random.default_rng(seed)
        self.Wout = None  # verrà settato in fit()

        # Genera Win e W
        self._init_weights()

    def _init_weights(self):
        """Crea Win e W, quindi scala W per avere lo spectral radius desiderato."""
        N = self.res_size
        # Win: (res_size, in_size+1)
        self.Win = (np.random.rand(self.res_size, self.in_size + 1) - 0.5) * 2 * self.input_scaling

        if self.topology == "random":
            W = self.rng.random((N, N)) - 0.5
            mask = self.rng.random((N, N)) < self.sparsity
            W[mask] = 0.0
        elif self.topology == "ring":
            W = np.zeros((N, N))
            for i in range(N):
                W[i, (i+1) % N] = self.rng.uniform(-0.5, 0.5)
                W[i, (i-1) % N] = self.rng.uniform(-0.5, 0.5)
        elif self.topology == "double_cycle":
            if N % 2 != 0:
                raise ValueError("double_cycle richiede res_size pari")
            W = np.zeros((N, N))
            half = N // 2
            for i in range(N):
                W[i, (i+1) % N]    = self.rng.uniform(-0.5, 0.5)
                W[i, (i-1) % N]    = self.rng.uniform(-0.5, 0.5)
                W[i, (i+half) % N] = self.rng.uniform(-0.5, 0.5)
                W[i, (i-half) % N] = self.rng.uniform(-0.5, 0.5)
        else:
            raise ValueError(f"Topologia '{self.topology}' non riconosciuta")

        eigs = np.linalg.eigvals(W)
        radius = np.max(np.abs(eigs))
        self.W = W * (self.spectral_radius / radius)

    def _update(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        One-step reservoir update (leaky integrator):
            x' = (1 - α)*x + α * tanh( Win·[1; u] + W·x ).
        `u` è sempre array 1D di lunghezza `in_size`.
        """
        u_aug = np.hstack(([1.0], u))  # (in_size + 1,)
        preact = self.Win.dot(u_aug) + self.W.dot(state)
        x_new = (1 - self.leak_rate) * state + self.leak_rate * np.tanh(preact)
        return x_new

    def fit(self, U: np.ndarray, Y: np.ndarray, washout: int = 100):
        """
        Allena la readout Wout tramite ridge-regression.
        U: array di shape (T, in_size) oppure (T,) se in_size=1
        Y: array di shape (T, out_size)
        """
        T = U.shape[0]
        Nf = self.res_size + self.in_size + 1

        # Prealloca X (da washout in poi) e Y_target
        X = np.zeros((T - washout, Nf))
        Y_target = Y[washout:]  # (T-washout, out_size)

        x = np.zeros(self.res_size)
        for t in range(T):
            # garantiamo che U[t] diventi array 1D di lunghezza in_size
            u = np.atleast_1d(U[t])
            x = self._update(x, u)
            if t >= washout:
                feat = np.hstack(([1.0], u, x))
                X[t - washout] = feat

        # Ridge regression chiude-forma
        XtX = X.T.dot(X)
        reg_mat = self.ridge_reg * np.eye(XtX.shape[0])
        Wtilde = np.linalg.inv(XtX + reg_mat)
        self.Wout = Y_target.T.dot(X).dot(Wtilde)  # (out_size, n_features)

    def predict(self, U: np.ndarray, continuation: bool = False) -> np.ndarray:
        """
        Predice la sequenza di output dato l'input U.
        U: (T, in_size) o (T,) se in_size=1.
        Se continuation=True, parte dallo stato di reservoir salvato.
        """
        T = U.shape[0]
        Y_hat = np.zeros((T, self.out_size))

        if continuation and hasattr(self, 'last_state'):
            x = self.last_state.copy()
        else:
            x = np.zeros(self.res_size)

        for t in range(T):
            u = np.atleast_1d(U[t])
            x = self._update(x, u)
            feat = np.hstack(([1.0], u, x))
            Y_hat[t] = self.Wout.dot(feat)

        self.last_state = x.copy()
        return Y_hat

# %%
import numpy as np
from scipy.optimize import minimize

SEED = 42  # seme fisso per riproducibilità

class PIESN(ESN):
    """
    PI-ESN che utilizza come feature esclusivamente lo stato del reservoir (x).
    """

    def __init__(
        self,
        in_size: int,
        res_size: int,
        out_size: int,
        spectral_radius: float = 0.95,
        sparsity: float = 0.1,
        input_scaling: float = 1.0,
        leak_rate: float = 1.0,
        ridge_reg: float = 1e-8,
        seed: int = SEED,
        topology: str = "random",
        lambda_data: float = 1.0,
        lambda_phys: float = 0.01,
        ode_func=None,
        delta_t: float = 0.01,
        phys_horizon: int = 10,
        full_closed_loop: bool = True,
        curriculum: bool = False,
        curr_tau: float = 10.0
    ):
        super().__init__(
            in_size        = in_size,
            res_size       = res_size,
            out_size       = out_size,
            spectral_radius= spectral_radius,
            sparsity       = sparsity,
            input_scaling  = input_scaling,
            leak_rate      = leak_rate,
            ridge_reg      = ridge_reg,
            seed           = seed,
            topology       = topology
        )
        self.lambda_data      = lambda_data
        self.lambda_phys      = lambda_phys
        self.ode_func         = ode_func
        self.delta_t          = delta_t
        self.phys_horizon     = phys_horizon
        self.full_closed_loop = full_closed_loop
        self.curriculum       = curriculum
        self.curr_tau         = curr_tau
        self.curr_step        = 0

        # Statistiche di normalizzazione (assegnate in fit_physics)
        self._mean_U = None  # array di lunghezza in_size
        self._std_U  = None  # array di lunghezza in_size

        # Stato del reservoir da ereditare se continuation=True
        self.last_state = None

    def _compute_data_loss(
        self,
        W_flat: np.ndarray,
        X: np.ndarray,
        Y_target: np.ndarray
    ) -> float:
        """
        Data‐loss = MSE + ridge_reg * ||Wout||^2
          - W_flat: vettore length = out_size * res_size
          - X: matrice degli stati normalizzati (shape = (T-washout, res_size))
          - Y_target: target normalizzato (shape = (T-washout, out_size))
        """
        Wout = W_flat.reshape(self.out_size, self.res_size)  # (out_size, res_size)
        Y_pred = X.dot(Wout.T)                               # (T-washout, out_size)
        mse    = np.mean((Y_pred - Y_target) ** 2)
        reg    = self.ridge_reg * np.sum(Wout ** 2)
        return mse + reg

    def _compute_physics_loss(
        self,
        W_flat: np.ndarray,
        Y0: np.ndarray,
        U_seq: np.ndarray
    ) -> float:
        """
        Physics‐loss su phys_horizon passi futuri:
          - Ricostruisce Y_pred in closed‐loop (o open‐loop) a partire da Y0 e U_seq
          - dY/dt ≈ (Y_pred[t+1] - Y_pred[t]) / delta_t
          - Confronta con ode_func(Y_pred[t]) e calcola MSE
          - Se curriculum=True, moltiplica per curr_factor
        """
        N    = self.phys_horizon
        Wout = W_flat.reshape(self.out_size, self.res_size)  # (out_size, res_size)
        x    = np.zeros(self.res_size)
        Y_pred = [Y0.copy()]  # lista di vettori (out_size,)

        for n in range(N):
            u = np.atleast_1d(U_seq[n])       # (in_size,)
            feat = x                         # (res_size,) → solo reservoir
            y_next = Wout.dot(feat)          # (out_size,)
            Y_pred.append(y_next)

            if self.full_closed_loop:
                x = self._update(x, y_next)
            else:
                x = self._update(x, u)

        Y_pred = np.stack(Y_pred)             # (N+1, out_size)
        dY     = (Y_pred[1:] - Y_pred[:-1]) / self.delta_t  # (N, out_size)

        try:
            F = self.ode_func(Y_pred[:-1])    # (N, out_size)
        except:
            F = np.vstack([self.ode_func(y) for y in Y_pred[:-1]])

        res = dY - F                          # (N, out_size)
        phys_loss = np.mean(res ** 2)

        if self.curriculum:
            curr_factor = 1 - np.exp(- self.curr_step / self.curr_tau)
            self.curr_step += 1
        else:
            curr_factor = 1.0

        return curr_factor * phys_loss

    def _total_loss(
        self,
        W_flat: np.ndarray,
        X: np.ndarray,
        Y_target: np.ndarray,
        Y0: np.ndarray,
        U_seq: np.ndarray
    ) -> float:
        """
        Combina data_loss e physics_loss:
          L = lambda_data * Ld + lambda_phys * Lp
        """
        Ld = self._compute_data_loss(W_flat, X, Y_target)
        Lp = self._compute_physics_loss(W_flat, Y0, U_seq)
        return self.lambda_data * Ld + self.lambda_phys * Lp

    def fit_physics(
        self,
        U: np.ndarray,
        Y: np.ndarray,
        washout: int = 100
    ):
        """
        Esegue addestramento PI-ESN:
          1. Normalizza (U,Y) con Z‐score
          2. Propaga reservoir su U_norm per raccogliere tutti gli stati (states)
          3. Costruisce X = states[washout:]  e Y_target = Y_norm[washout:]
             ⇒ X ha shape (T-washout, res_size)
          4. Risolve regressione ridge iniziale su (X, Y_target)
             ⇒ W0 piatto di shape (out_size * res_size)
          5. Minimizza L = data_loss + physics_loss con L-BFGS-B
        Ritorna OptimizeResult di SciPy.
        """
        T = U.shape[0]
        N = self.phys_horizon

        # 1) Normalizzazione Z‐score
        mean_U = U.mean(axis=0)          # (in_size,)
        std_U  = U.std(axis=0) + 1e-8
        U_norm = (U - mean_U) / std_U     # (T, in_size)
        Y_norm = (Y - mean_U) / std_U     # (T, out_size)

        self._mean_U = mean_U.copy()
        self._std_U  = std_U.copy()

        # 2) Propago il reservoir e salvo gli stati in “states”
        x_state = np.zeros(self.res_size)
        states = np.zeros((T, self.res_size))  # (T, res_size)
        for t_i in range(T):
            u_i = np.atleast_1d(U_norm[t_i])   # (in_size,)
            x_state = self._update(x_state, u_i)
            states[t_i] = x_state

        # 3) Costruisco X, Y_target dopo washout
        X = states[washout:]           # (T-washout, res_size)
        Y_target = Y_norm[washout:]    # (T-washout, out_size)

        # Verifica condizione physics horizon
        if T < washout + 2 * N:
            raise ValueError("Serve T >= washout + 2 * phys_horizon")

        # 4) Preparo Y0 e U_seq normalizzati per physics_loss
        idx0 = washout + N
        Y0   = Y_norm[idx0]                 # (out_size,)
        U_seq = U_norm[idx0 : idx0 + N]      # (N, in_size)

        # 5) Soluzione iniziale ridge-only (X, Y_target)
        XtX = X.T.dot(X)                     # (res_size, res_size)
        reg_mat = self.ridge_reg * np.eye(self.res_size)
        W_ridge = np.linalg.solve(XtX + reg_mat, X.T.dot(Y_target))
        # W_ridge ha shape (res_size, out_size)
        W0 = W_ridge.T.flatten()             # (out_size * res_size,)

        # 6) Minimizzazione loss totale con L-BFGS-B
        self.curr_step = 0
        result = minimize(
            fun    = self._total_loss,
            x0     = W0,
            args   = (X, Y_target, Y0, U_seq),
            method = 'L-BFGS-B'
        )

        W_opt = result.x.reshape(self.out_size, self.res_size)  # (out_size, res_size)
        self.Wout = W_opt.copy()
        return result

    def predict(
        self,
        U: np.ndarray,
        continuation: bool = False
    ) -> np.ndarray:
        """
        1. Normalizza U usando (mean_U, std_U)
        2. Propaga reservoir e raccoglie gli stati x_t
        3. Calcola Y_hat_norm[t] = Wout ⋅ x_t
        4. De‐normalizza: Y_hat = Y_hat_norm * std_U + mean_U
        5. Se continuation=True, eredita last_state
        """
        U_norm = (U - self._mean_U) / self._std_U  # (T_te, in_size)
        T = U_norm.shape[0]
        Y_hat_norm = np.zeros((T, self.out_size))  # (T_te, out_size)

        if continuation and hasattr(self, "last_state"):
            x = self.last_state.copy()
        else:
            x = np.zeros(self.res_size)

        for t_i in range(T):
            u_i = np.atleast_1d(U_norm[t_i])
            x = self._update(x, u_i)           # (res_size,)
            Y_hat_norm[t_i] = self.Wout.dot(x) # (out_size,)

        self.last_state = x.copy()
        return (Y_hat_norm * self._std_U) + self._mean_U

# %%
def objective(trial):
    
    params = {
        "in_size":         3,
        "out_size":        3,
        "res_size":       trial.suggest_int("res_size",      256, 1024, step=2),
        "spectral_radius":trial.suggest_float("rho",        0.1, 0.3, log=True),
        "sparsity":       trial.suggest_float("sparsity",   0.01, 0.05, log=True),
        "input_scaling":  trial.suggest_float("in_scale",   0.1, 1.0),
        "leak_rate":      trial.suggest_float("alpha",      0.2, 0.5),
        "ridge_reg":      trial.suggest_float("ridge_reg",  1e-6, 1e-3, log=True),
        "topology":       trial.suggest_categorical("topo",   ["random","ring","double_cycle"]),
        "lambda_data":    trial.suggest_float("λ_data",     0.5, 3.0, log=True),
        "lambda_phys":    trial.suggest_float("λ_phys",     1e-2, 0.5, log=True),
        "phys_horizon":   trial.suggest_int("horizon",      3,   7),
    }

    fold_scores = []
    for train_idx, val_idx in tscv.split(U):
        u_tr,  u_val  = U[train_idx],  U[val_idx]
        y_tr,  y_val  = Y[train_idx],  Y[val_idx]

        # 2) Costruisci PIESN con tutti i nuovi flag
        piesn = PIESN(
            **params,
            seed=SEED,
            ode_func=machen_rhs,
            delta_t=dt,
            full_closed_loop=True,
            curriculum=True,
            curr_tau=10.0
        )

        # 3) Allena con washout (normalizzazione interna inclusa)
        piesn.fit_physics(u_tr, y_tr, washout=100)

        # 4) Predici
        y_val_pred = piesn.predict(u_val, continuation=False)

        # 5) Calcola NRMSE (3D)
        score = nrmse_multi(y_val[100:], y_val_pred[100:])
        fold_scores.append(score)

    return float(np.mean(fold_scores))

# %%
import numpy as np
import pandas as pd
from numpy.linalg import norm
from functools import partial   # ← aggiunto


SEED = 42  # seme fisso per riproducibilità

best = {
    'res_size': 960,
    'rho': 0.10009762488352285,
    'sparsity': 0.014635603653829913,
    'in_scale': 0.10012333896882074,
    'alpha': 0.22320271024144345,
    'ridge_reg': 8.160533191764472e-05,
    'topo': 'random',
    'λ_data': 0.736548536623833,
    'λ_phys': 0.13714180027119693,
    'horizon': 7
}
# Best CV NRMSE: 0.006060125647370649

dt = 0.01
length = 10000

alpha = 3
beta = 0.1
gamma = 1

results = []
noise_levels = [0.0, 0.05, 1.0, 1.5, 2.0, 4.0, 5.0]

for noise in noise_levels:
   
    np.random.seed(SEED)
    t, traj_noisy = generate_machen(length=length, dt=dt, noise=noise)

    U_noisy = traj_noisy[:-1]   
    Y_noisy = traj_noisy[1:]    
    (u_tr, y_tr), (u_val, y_val), (u_te, y_te) = split_train_val_test(U_noisy, Y_noisy, train_frac=0.6, val_frac=0.2)


    ode_wrapped = partial(machen_rhs, alpha=alpha, beta=beta, gamma=gamma)

    piesn_final = PIESN(
        in_size=3,
        res_size=best['res_size'],
        out_size=3,
        spectral_radius=best['rho'],
        sparsity=best['sparsity'],
        input_scaling=best['in_scale'],
        leak_rate=best['alpha'],
        ridge_reg=best['ridge_reg'],
        seed=SEED,
        topology=best['topo'],
        lambda_data=best['λ_data'],
        lambda_phys=best['λ_phys'],
        ode_func=ode_wrapped,        # il wrapper che include già alpha,beta,gamma
        delta_t=dt,
        phys_horizon=best['horizon'],
        full_closed_loop=True,
        curriculum=True,
        curr_tau=10.0
    )
    piesn_final.fit_physics(u_tr, y_tr, washout=100)
    y_pred_piesn = piesn_final.predict(u_te, continuation=False)
    nrmse_piesn = norm(y_te[100:] - y_pred_piesn[100:]) / norm(y_te[100:])

    esn_only = ESN(
        in_size=3,
        res_size=best['res_size'],
        out_size=3,
        spectral_radius=best['rho'],
        sparsity=best['sparsity'],
        input_scaling=best['in_scale'],
        leak_rate=best['alpha'],
        ridge_reg=best['ridge_reg'],
        seed=SEED,
        topology=best['topo']
    )
    esn_only.fit(u_tr, y_tr, washout=100)
    y_pred_esn = esn_only.predict(u_te, continuation=False)
    nrmse_esn = norm(y_te[100:] - y_pred_esn[100:]) / norm(y_te[100:])

    # salva risultati
    results.append({
        "noise_level":   noise,
        "nrmse_piesn":   nrmse_piesn,
        "nrmse_esn":     nrmse_esn
    })

df_results = pd.DataFrame(results)
df_results

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
from functools import partial

best = {
    'res_size': 960,
    'rho': 0.10009762488352285,
    'sparsity': 0.014635603653829913,
    'in_scale': 0.10012333896882074,
    'alpha': 0.22320271024144345,
    'ridge_reg': 8.160533191764472e-05,
    'topo': 'random',
    'λ_data': 0.736548536623833,
    'λ_phys': 0.13714180027119693,
    'horizon': 7
}

SEED, dt, length = 42, 0.01, 6000

# Parametri Ma–Chen
alpha_mc = 3.0
beta_mc  = 0.1
gamma_mc = 1.0

# Livelli di rumore “equivalenti” a quelli usati per Lorenz
noise_levels = [0.03, 0.06, 0.09, 0.12, 0.15]
horizons        = [1, 10, 50, 100, 200, 500, 1000]
lambda_phys_grid = [0.005, 0.01, 0.02, 0.05]

def roll(model, u0, steps, σ=0, rng=None):
    y = model.predict(u0.reshape(1, -1), continuation=False)[0]
    out = [y.copy()]
    for _ in range(steps - 1):
        if σ:
            y_in = y + rng.normal(scale=σ, size=y.shape)
        else:
            y_in = y
        y = model.predict(y_in.reshape(1, -1), continuation=True)[0]
        out.append(y.copy())
    return np.stack(out)[:, 1]

def nrmse(true, pred):
    return norm(true - pred) / norm(true)

# 1) Genero una volta la traiettoria pulita Ma–Chen per il test
np.random.seed(SEED)
_, traj_clean = generate_machen(
    length=length, dt=dt,
    alpha=alpha_mc, beta=beta_mc, gamma=gamma_mc,
    noise=0.0, seed=SEED
)
U_clean, Y_clean = traj_clean[:-1], traj_clean[1:]
(_, _), (_, _), (u_te, y_te) = split_train_val_test(
    U_clean, Y_clean, train_frac=0.6, val_frac=0.2
)
y_true = y_te[:, 1]

rows = []

# 2) Colleziono prestazioni per vari σ e λ_phys
for σ in noise_levels:
    np.random.seed(SEED)
    _, traj_noisy = generate_machen(
        length=length, dt=dt,
        alpha=alpha_mc, beta=beta_mc, gamma=gamma_mc,
        noise=σ, seed=SEED
    )
    U_noisy, Y_noisy = traj_noisy[:-1], traj_noisy[1:]
    (u_tr, y_tr), (_, _), _ = split_train_val_test(
        U_noisy, Y_noisy, train_frac=0.6, val_frac=0.2
    )

    # Alleno ESN puro su dati rumorosi
    esn = ESN(
        in_size         = 3,
        res_size        = best['res_size'],
        out_size        = 3,
        spectral_radius = best['rho'],
        sparsity        = best['sparsity'],
        input_scaling   = best['in_scale'],
        leak_rate       = best['alpha'],
        ridge_reg       = best['ridge_reg'],
        seed            = SEED,
        topology        = best['topo']
    )
    esn.fit(u_tr, y_tr, washout=100)
    rng_esn = np.random.default_rng(int(1e6 * σ) + SEED)

    for lp in lambda_phys_grid:
        pi = PIESN(
            in_size          = 3,
            res_size         = best['res_size'],
            out_size         = 3,
            spectral_radius  = best['rho'],
            sparsity         = best['sparsity'],
            input_scaling    = best['in_scale'],
            leak_rate        = best['alpha'],
            ridge_reg        = best['ridge_reg'],
            seed             = SEED,
            topology         = best['topo'],
            lambda_data      = best['λ_data'],
            lambda_phys      = lp,
            ode_func         = partial(machen_rhs, alpha=alpha_mc, beta=beta_mc, gamma=gamma_mc),
            delta_t          = dt,
            phys_horizon     = best['horizon'],
            full_closed_loop = True,
            curriculum       = False
        )
        pi.fit_physics(u_tr, y_tr, washout=100)
        rng_pi = np.random.default_rng(int(1e6 * σ) + SEED + int(lp * 1e3))

        esn_noisy = roll(esn, u_te[0], max(horizons), σ, rng_esn)
        pi_noisy  = roll(pi,  u_te[0], max(horizons), σ, rng_pi)
        esn_clean = roll(esn, u_te[0], max(horizons))
        pi_clean  = roll(pi,  u_te[0], max(horizons))

        for h in horizons:
            rows.append({
                'σ':           σ,
                'λ_phys':      lp,
                'h':           h,
                'ESN_noisy':   nrmse(y_true[:h], esn_noisy[:h]),
                'PI_noisy':    nrmse(y_true[:h], pi_noisy[:h]),
                'ESN_clean':   nrmse(y_true[:h], esn_clean[:h]),
                'PI_clean':    nrmse(y_true[:h], pi_clean[:h])
            })

df = pd.DataFrame(rows)

# 3) Trova λ_phys ottimale per ciascun σ (minimizzando PI_noisy a h = 10)
best_lps = (
    df[df['h'] == 10]
    .groupby('σ')
    .apply(lambda d: d.loc[d['PI_noisy'].idxmin(), 'λ_phys'])
    .to_dict()
)

# 4) Plot NRMSE vs Horizon per ESN e PI-ESN usando λ_phys ottimale
for σ in noise_levels:
    lp_best = best_lps[σ]
    subset = df[(df['σ'] == σ) & (df['λ_phys'] == lp_best)]
    plt.figure(figsize=(6, 4))
    plt.plot(horizons, subset['ESN_noisy'], 'r-o', label='ESN noisy-roll')
    plt.plot(horizons, subset['PI_noisy'],  'b-s', label=f'PI-ESN noisy-roll (λ={lp_best})')
    plt.plot(horizons, subset['ESN_clean'],'m--^', label='ESN clean-roll')
    plt.plot(horizons, subset['PI_clean'], 'c-.*', label=f'PI-ESN clean-roll (λ={lp_best})')
    plt.xlabel('Horizon (h)')
    plt.ylabel('NRMSE')
    plt.title(f'NRMSE vs Horizon at σ = {σ:.2f} (Ma–Chen)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()

# 5) Plot “True vs Predicted” per λ_phys ottimale
hmax = max(horizons)
for σ in noise_levels:
    lp_best = best_lps[σ]

    np.random.seed(SEED)
    _, traj_noisy = generate_machen(
        length=length, dt=dt, noise=σ,
        alpha=alpha_mc, beta=beta_mc, gamma=gamma_mc,
        seed=SEED
    )
    U_noisy, Y_noisy = traj_noisy[:-1], traj_noisy[1:]
    (u_tr, y_tr), (_, _), _ = split_train_val_test(U_noisy, Y_noisy, train_frac=0.6, val_frac=0.2)

    esn = ESN(
        in_size         = 3,
        res_size        = best['res_size'],
        out_size        = 3,
        spectral_radius = best['rho'],
        sparsity        = best['sparsity'],
        input_scaling   = best['in_scale'],
        leak_rate       = best['alpha'],
        ridge_reg       = best['ridge_reg'],
        seed            = SEED,
        topology        = best['topo']
    )
    esn.fit(u_tr, y_tr, washout=100)
    rng_esn = np.random.default_rng(int(1e6 * σ) + SEED)

    pi = PIESN(
        in_size         = 3,
        res_size        = best['res_size'],
        out_size        = 3,
        spectral_radius = best['rho'],
        sparsity        = best['sparsity'],
        input_scaling   = best['in_scale'],
        leak_rate       = best['alpha'],
        ridge_reg       = best['ridge_reg'],
        seed            = SEED,
        topology        = best['topo'],
        lambda_data     = best['λ_data'],
        lambda_phys     = lp_best,
        ode_func        = partial(machen_rhs, alpha=alpha_mc, beta=beta_mc, gamma=gamma_mc),
        delta_t         = dt,
        phys_horizon    = best['horizon'],
        full_closed_loop= True,
        curriculum      = False
    )
    pi.fit_physics(u_tr, y_tr, washout=100)
    rng_pi = np.random.default_rng(int(1e6 * σ) + SEED + int(lp_best * 1e3))

    esn_clean = roll(esn, u_te[0], hmax, σ=0, rng=None)
    pi_clean  = roll(pi,  u_te[0], hmax, σ=0, rng=None)

    esn_noisy = roll(esn, u_te[0], hmax, σ, rng_esn)
    pi_noisy  = roll(pi,  u_te[0], hmax, σ, rng_pi)

    idx = np.arange(hmax)

    plt.figure(figsize=(6, 4))
    plt.plot(idx, y_true[:hmax], 'k-', label='True $u_2$')
    plt.plot(idx, esn_clean[:hmax], 'r--', label='ESN clean-roll')
    plt.plot(idx, pi_clean[:hmax],  'b-.', label=f'PI-ESN clean-roll (λ={lp_best})')
    plt.xlabel('Step')
    plt.ylabel('$u_2$')
    plt.title(f'Clean-roll True vs Predicted at σ_train = {σ:.2f} (Ma–Chen)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(idx, y_true[:hmax], 'k-', label='True $u_2$')
    plt.plot(idx, esn_noisy[:hmax], 'r--', label='ESN noisy-roll')
    plt.plot(idx, pi_noisy[:hmax],  'b-.', label=f'PI-ESN noisy-roll (λ={lp_best})')
    plt.xlabel('Step')
    plt.ylabel('$u_2$')
    plt.title(f'Noisy-roll True vs Predicted at σ_train = {σ:.2f} (Ma–Chen)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# %%
# 4) Costruisci un DataFrame riepilogativo usando il λ_phys ottimale
summary_rows = []
for σ, lp_best in best_lps.items():
    subset = df[(df['σ'] == σ) & (df['λ_phys'] == lp_best)]
    for _, row in subset.iterrows():
        summary_rows.append({
            'σ': σ,
            'h': int(row['h']),
            'λ_phys': lp_best,
            'ESN_clean': row['ESN_clean'],
            'PI_clean': row['PI_clean'],
            'ESN_noisy': row['ESN_noisy'],
            'PI_noisy': row['PI_noisy']
        })

df_summary = pd.DataFrame(summary_rows)
df_summary = df_summary.sort_values(by=['σ', 'h']).reset_index(drop=True)

# Stampa la tabella riepilogativa
print("---------- Riepilogo NRMSE per ciascun σ e h (usando λ_phys ottimale) ----------\n")
print(df_summary.to_string(index=False, float_format="{:.3f}".format))

# %%
#computer the variance of y_true
variance_y_true = np.var(Y_true)
print(f"Varianza di y_true: {variance_y_true:.4f}")

# %%
from scipy.stats import pearsonr
# === 2. Funzione per calcolare la curva di memoria lineare ===
def compute_memory_curve(esn, delay_max=500, washout=100, input_length=10000):
    rng = np.random.default_rng(0)
    # Sequenza di input casuale u(t) ∈ [-1,1]
    U = rng.uniform(-1, 1, size=(input_length, esn.in_size))
    
    # Colleziono stati dopo washout
    states = np.zeros((input_length - washout, esn.res_size))
    x = np.zeros(esn.res_size)
    for t in range(input_length):
        u = np.atleast_1d(U[t])
        x = esn._update(x, u)
        if t >= washout:
            states[t - washout] = x
    U_trim = U[washout:]
    
    # Calcolo r_k^2 per ogni ritardo k
    r2 = np.zeros(delay_max)
    for k in range(1, delay_max + 1):
        if k >= len(U_trim):
            break
        X_k = states[k:]            # (T - washout - k, res_size)
        target = U_trim[:-k, 0]     # Ricostruiamo prima componente di U
        # Disegno la matrice di progetto X_design = [1, x(t)]
        X_design = np.hstack((np.ones((X_k.shape[0], 1)), X_k))
        # Ridge closed-form: w = (X^T X + α I)^{-1} X^T y
        alpha = esn.ridge_reg
        XtX = X_design.T.dot(X_design)
        reg = alpha * np.eye(XtX.shape[0])
        w = np.linalg.inv(XtX + reg).dot(X_design.T).dot(target)
        preds = X_design.dot(w)
        r, _ = pearsonr(preds, target)
        r2[k-1] = r**2
    return r2

# === 3. Funzione per tracciare la curva di memoria e capacità cumulativa ===
def plot_memory_curve(r2):
    k = np.arange(1, len(r2) + 1)
    cum_capacity = np.cumsum(r2)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: r_k^2 vs k
    plt.subplot(1, 2, 1)
    plt.plot(k, r2, label=r"$r_k^2$")
    plt.xlabel("Delay $k$")
    plt.ylabel(r"$r_k^2$")
    plt.title("Linear Memory Curve")
    plt.grid(True)
    
    # Plot 2: capacità cumulativa vs k
    plt.subplot(1, 2, 2)
    plt.plot(k, cum_capacity, label="Cumulative capacity", color='orange')
    plt.xlabel("Delay $k$")
    plt.ylabel(r"Cumulative capacity $\sum_{i=1}^k r_i^2$")
    plt.title("Cumulative Linear Memory Capacity")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# === 4. Funzione per la risposta a impulso del reservoir ===
def compute_impulse_response(esn, time_steps=1000):
    # Definiamo u(0)=1, u(t>0)=0
    U = np.zeros((time_steps, esn.in_size))
    U[0] = np.ones(esn.in_size)
    
    x = np.zeros(esn.res_size)
    norms = np.zeros(time_steps)
    for t in range(time_steps):
        u = U[t]
        x = esn._update(x, u)
        norms[t] = np.linalg.norm(x)
    return norms

def plot_impulse_response(norms):
    t = np.arange(len(norms))
    plt.figure(figsize=(6, 4))
    plt.semilogy(t, norms, label=r"$\|x(t)\|$")
    plt.xlabel("Time step $t$")
    plt.ylabel(r"Reservoir state norm (scala logaritmica)")
    plt.title("Impulse Response Decay")
    plt.grid(True)
    plt.show()

# === 5. Esempio di utilizzo con i parametri “best” forniti ===

best = {
    'res_size':   960,
    'rho':        0.10009762488352285,
    'sparsity':   0.014635603653829913,
    'in_scale':   0.10012333896882074,
    'alpha':      0.22320271024144345,
    'ridge_reg':  8.160533191764472e-05,
    'topo':       'random'
}
# 5.1 Istanzio l’ESN con i parametri “best”
esn = ESN(
    in_size=1,
    res_size=best_params['res_size'],
    out_size=1,
    spectral_radius=best_params['rho'],
    sparsity=best_params['sparsity'],
    input_scaling=best_params['in_scale'],
    leak_rate=best_params['alpha'],
    ridge_reg=best_params['ridge_reg'],
    topology=best_params['topo'],
    seed=42
)

# 5.2 Calcolo la curva di memoria lineare
r2_curve = compute_memory_curve(esn, delay_max=500, washout=100, input_length=10000)

# 5.3 Traccio i grafici di memoria e capacità cumulativa
plot_memory_curve(r2_curve)

# 5.4 Calcolo e traccia la risposta a impulso
norms = compute_impulse_response(esn, time_steps=1000)
plot_impulse_response(norms)

print(f"Ritardo k* (r_k^2 < {epsilon}): {k_star}")
print(f"Capacità lineare totale (∑ r_k^2): {total_MC:.4f}")
print(f"Capacità normalizzata (MC / N): {normalized_MC:.6f}")
print(f"Costante di decadimento stimata τ: {tau_est:.2f} time steps")

# %%
import numpy as np
import matplotlib.pyplot as plt

best = {
    'res_size':   960,
    'rho':        0.10009762488352285,
    'sparsity':   0.014635603653829913,
    'in_scale':   0.10012333896882074,
    'alpha':      0.22320271024144345,
    'ridge_reg':  8.160533191764472e-05,
    'topo':       'random'
}

SEED    = 42
in_size = 1
out_size= 1
N       = best['res_size']

# Istanzia solo l'ESN
esn = ESN(
    in_size         = in_size,
    res_size        = N,
    out_size        = out_size,
    spectral_radius = best['rho'],
    sparsity        = best['sparsity'],
    input_scaling   = best['in_scale'],
    leak_rate       = best['alpha'],
    ridge_reg       = best['ridge_reg'],
    seed            = SEED,
    topology        = best['topo']
)

# ------------------------------------------------------------------
# 3) drive l'ESN con input i.i.d. ±1
# ------------------------------------------------------------------
T_warm = 200
T_eval = 4000
rng = np.random.default_rng(SEED)
u_seq = rng.choice([-1.0, +1.0], size=T_warm + T_eval)

# Warm‐up: chiamata a fit(…) per i primi 200 passi
esn.fit(
    u_seq[:T_warm].reshape(-1,1),
    u_seq[:T_warm].reshape(-1,1),
    washout=0
)

# Raccogliamo gli stati interni per i successivi 4000 passi
states_esn   = np.zeros((T_eval, N))
x_esn = np.zeros(N)

for t in range(T_eval):
    u_in = np.array([u_seq[T_warm + t]])
    x_esn   = esn._update(x_esn, u_in)
    states_esn[t]   = x_esn

# ------------------------------------------------------------------
# 4) calcolo linear memory‐capacity curve MC(k)
# ------------------------------------------------------------------
max_k     = 120
eps_ridge = 1e-8

R2_esn = []

for k in range(1, max_k + 1):
    # target u(t−k) allineato su [T_warm, T_warm+T_eval)
    targets = u_seq[T_warm - k : T_warm + T_eval - k]  # lunghezza = T_eval

    # Ridge closed‐form: X = states_esn (T_eval, N)
    X = states_esn
    XtX = X.T @ X
    reg = eps_ridge * np.eye(N)
    W_ridge_esn = np.linalg.solve(XtX + reg, X.T @ targets)  # (N,)
    pred_esn = X @ W_ridge_esn

    var_t   = np.var(targets)
    R2_k_esn = 0.0 if var_t == 0 else 1 - np.var(targets - pred_esn) / var_t
    R2_esn.append(R2_k_esn)

MC_total_esn   = np.sum(R2_esn)

print(f"Total linear MC (ESN) ≈ {MC_total_esn:.2f}  (upper bound = {N})")

# ------------------------------------------------------------------
# 5) Plotto la memory‐capacity curve
# ------------------------------------------------------------------
ks = np.arange(1, max_k + 1)

plt.figure(figsize=(8,4))
plt.plot(ks, R2_esn,   label="ESN",   lw=1.5)
plt.xlabel("ritardo $k$")
plt.ylabel("$MC(k)$  (coeff. di determinazione $R^2$)")
plt.title("Linear Memory‐Capacity Curve: ESN")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

# %%
import numpy as np

def compute_MC(res_size, rho, alpha, sparsity, max_k=120):
    rng = np.random.default_rng(42)
    u_seq = rng.choice([-1.0, +1.0], size=200 + 4000)

    esn_test = ESN(
        in_size         = 1,
        res_size        = res_size,
        out_size        = 1,
        spectral_radius = rho,
        sparsity        = sparsity,
        input_scaling   = 0.1177,
        leak_rate       = alpha,
        ridge_reg       = 1e-6,
        seed            = 42,
        topology        = 'double_cycle'
    )
    esn_test.fit(u_seq[:200].reshape(-1,1),
                 u_seq[:200].reshape(-1,1),
                 washout=0)

    T_eval = 4000
    states = np.zeros((T_eval, res_size))
    x = np.zeros(res_size)
    for t in range(T_eval):
        x = esn_test._update(x, np.array([u_seq[200 + t]]))
        states[t] = x

    R2 = []
    eps = 1e-8
    for k in range(1, max_k+1):
        targets = u_seq[200 - k : 200 + T_eval - k]
        X = states
        W = np.linalg.solve(X.T.dot(X) + eps*np.eye(res_size),
                            X.T.dot(targets))
        pred = X.dot(W)
        var_t = np.var(targets)
        R2.append(0.0 if var_t == 0 else 1 - np.var(targets - pred)/var_t)

    return np.sum(R2), R2

# esempio di sweep parziale
params_list = [
    (702, 0.95, 0.8, 0.018),
    (702, 0.98, 0.9, 0.018),
    (702, 0.99, 0.9, 0.05),
    (1000, 0.95, 0.8, 0.018),
    (1000, 0.98, 0.9, 0.018),
]

for (res_size, rho, alpha, sparsity) in params_list:
    MC_val, _ = compute_MC(res_size, rho, alpha, sparsity)
    print(f"N={res_size}, ρ={rho:.2f}, α={alpha:.2f}, spar={sparsity:.3f} → MC≈{MC_val:.1f}")

# %%
from scipy.stats import pearsonr
# === 2. Funzione per calcolare la curva di memoria lineare ===
def compute_memory_curve(esn, delay_max=500, washout=100, input_length=10000):
    rng = np.random.default_rng(0)
    # Sequenza di input casuale u(t) ∈ [-1,1]
    U = rng.uniform(-1, 1, size=(input_length, esn.in_size))
    
    # Colleziono stati dopo washout
    states = np.zeros((input_length - washout, esn.res_size))
    x = np.zeros(esn.res_size)
    for t in range(input_length):
        u = np.atleast_1d(U[t])
        x = esn._update(x, u)
        if t >= washout:
            states[t - washout] = x
    U_trim = U[washout:]
    
    # Calcolo r_k^2 per ogni ritardo k
    r2 = np.zeros(delay_max)
    for k in range(1, delay_max + 1):
        if k >= len(U_trim):
            break
        X_k = states[k:]            # (T - washout - k, res_size)
        target = U_trim[:-k, 0]     # Ricostruiamo prima componente di U
        # Disegno la matrice di progetto X_design = [1, x(t)]
        X_design = np.hstack((np.ones((X_k.shape[0], 1)), X_k))
        # Ridge closed-form: w = (X^T X + α I)^{-1} X^T y
        alpha = esn.ridge_reg
        XtX = X_design.T.dot(X_design)
        reg = alpha * np.eye(XtX.shape[0])
        w = np.linalg.inv(XtX + reg).dot(X_design.T).dot(target)
        preds = X_design.dot(w)
        r, _ = pearsonr(preds, target)
        r2[k-1] = r**2
    return r2

# === 3. Funzione per tracciare la curva di memoria e capacità cumulativa ===
def plot_memory_curve(r2):
    k = np.arange(1, len(r2) + 1)
    cum_capacity = np.cumsum(r2)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: r_k^2 vs k
    plt.subplot(1, 2, 1)
    plt.plot(k, r2, label=r"$r_k^2$")
    plt.xlabel("Delay $k$")
    plt.ylabel(r"$r_k^2$")
    plt.title("Linear Memory Curve")
    plt.grid(True)
    
    # Plot 2: capacità cumulativa vs k
    plt.subplot(1, 2, 2)
    plt.plot(k, cum_capacity, label="Cumulative capacity", color='orange')
    plt.xlabel("Delay $k$")
    plt.ylabel(r"Cumulative capacity $\sum_{i=1}^k r_i^2$")
    plt.title("Cumulative Linear Memory Capacity")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# === 4. Funzione per la risposta a impulso del reservoir ===
def compute_impulse_response(esn, time_steps=1000):
    # Definiamo u(0)=1, u(t>0)=0
    U = np.zeros((time_steps, esn.in_size))
    U[0] = np.ones(esn.in_size)
    
    x = np.zeros(esn.res_size)
    norms = np.zeros(time_steps)
    for t in range(time_steps):
        u = U[t]
        x = esn._update(x, u)
        norms[t] = np.linalg.norm(x)
    return norms

def plot_impulse_response(norms):
    t = np.arange(len(norms))
    plt.figure(figsize=(6, 4))
    plt.semilogy(t, norms, label=r"$\|x(t)\|$")
    plt.xlabel("Time step $t$")
    plt.ylabel(r"Reservoir state norm (scala logaritmica)")
    plt.title("Impulse Response Decay")
    plt.grid(True)
    plt.show()

# === 5. Esempio di utilizzo con i parametri “best” forniti ===
best_params  = {
    'res_size':   960,
    'rho':        0.10009762488352285,
    'sparsity':   0.014635603653829913,
    'in_scale':   0.10012333896882074,
    'alpha':      0.22320271024144345,
    'ridge_reg':  8.160533191764472e-05,
    'topo':       'random'
}
# 5.1 Istanzio l’ESN con i parametri “best”
esn = ESN(
    in_size=1,
    res_size=best_params['res_size'],
    out_size=1,
    spectral_radius=best_params['rho'],
    sparsity=best_params['sparsity'],
    input_scaling=best_params['in_scale'],
    leak_rate=best_params['alpha'],
    ridge_reg=best_params['ridge_reg'],
    topology=best_params['topo'],
    seed=42
)

# 5.2 Calcolo la curva di memoria lineare
r2_curve = compute_memory_curve(esn, delay_max=500, washout=100, input_length=10000)

# 5.3 Traccio i grafici di memoria e capacità cumulativa
plot_memory_curve(r2_curve)

# 5.4 Calcolo e traccia la risposta a impulso
norms = compute_impulse_response(esn, time_steps=1000)
plot_impulse_response(norms)

epsilon = 0.01
indices = np.where(r2_curve < epsilon)[0]
k_star = indices[0] + 1 if len(indices) > 0 else None

total_MC = np.sum(r2_curve)
normalized_MC = total_MC / esn.res_size

T_fit = 50
t_fit = np.arange(T_fit)
y_fit = norms[:T_fit]
coeffs = np.polyfit(t_fit, np.log(y_fit), 1)
q, lnA = coeffs
tau_est = -1.0 / q
print(f"Ritardo k* (r_k^2 < {epsilon}): {k_star}")
print(f"Capacità lineare totale (∑ r_k^2): {total_MC:.4f}")
print(f"Capacità normalizzata (MC / N): {normalized_MC:.6f}")
print(f"Costante di decadimento stimata τ: {tau_est:.2f} time steps")


