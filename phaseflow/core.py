"""
PhaseFlow v1.0 — core.py
Patiesības Rezonatora arhitektūra, balstīta uz fāžu matemātiku.

Pamati:
  - Visi stāvokļi attēloti kā kompleksās fāzes e^{iθ} uz vienības apļa ||x|| = 1
  - Sinhronizācija caur Kuramoto dinamiku
  - Rezonanse pārbaudīta ar FFT pret bāzes harmonikas kopumu

Autors: PhaseFlow projekts
"""

import numpy as np
from numpy.fft import fft, fftfreq
from dataclasses import dataclass, field
from typing import Sequence


# ─────────────────────────────────────────────
# 1. KOSMISKĀS KONSTANTES UN HARMONIKU BĀZE
# ─────────────────────────────────────────────

# Zelta proporcija
PHI: float = (1.0 + np.sqrt(5.0)) / 2.0          # φ ≈ 1.61803...

# Atsauces punkts — 42. kārtas psi konstante (normalizēta uz [0, 2π])
PSI_42: float = (42.0 % (2.0 * np.pi))            # ψ₄₂ ≈ 4.2920 rad

# Lambda_21: pabeigtības slieksnis rezonanses filtrā (0..1)
LAMBDA_21: float = 1.0 / PHI**2                   # ≈ 0.3820

# Bāzes harmoniku kopums — "svari", kas nemainās
BASE_HARMONICS: np.ndarray = np.array([1, 3, 6, 9, 11], dtype=float)


def harmonic_phases(harmonics: np.ndarray = BASE_HARMONICS,
                    base_angle: float = PSI_42) -> np.ndarray:
    """
    Atgriež bāzes harmoniku fāžu vektoru kā kompleksus skaitļus uz vienības apļa.

    Katram n ∈ harmonics:  z_n = e^{i · n · base_angle}

    Parameters
    ----------
    harmonics   : harmoniku kārtu masīvs
    base_angle  : atsauces fāzes leņķis (radiāni)

    Returns
    -------
    np.ndarray  : komplekso fāžu masīvs, ||z_n|| = 1
    """
    return np.exp(1j * harmonics * base_angle)


def phi_scale(n: int) -> np.ndarray:
    """
    Atgriež Zelta proporcijas pakāpju virkni: [φ^1, φ^2, ..., φ^n].
    Izmantojama kā harmoniku amplitūdu "dabiskā svēršana".
    """
    return PHI ** np.arange(1, n + 1)


# ─────────────────────────────────────────────
# 2. FĀZES ATTĒLOJUMS (vienības aplis)
# ─────────────────────────────────────────────

def to_phase(theta: float | np.ndarray) -> np.ndarray:
    """Pārveido leņķi(-us) θ par komplekso fāzi e^{iθ}."""
    return np.exp(1j * np.asarray(theta, dtype=float))


def phase_to_angle(z: np.ndarray) -> np.ndarray:
    """Iegūst leņķi θ = arg(z) ∈ (-π, π] no kompleksās fāzes."""
    return np.angle(z)


def normalize_phase(z: np.ndarray) -> np.ndarray:
    """Normalizē komplekso vektoru uz vienības apli: z / |z|."""
    magnitude = np.abs(z)
    magnitude = np.where(magnitude == 0, 1.0, magnitude)
    return z / magnitude


# ─────────────────────────────────────────────
# 3. KURAMOTO DZINĒJS
# ─────────────────────────────────────────────

@dataclass
class KuramotoOscillator:
    """
    N savstarpēji saistītu oscilatoru Kuramoto modelis ar DINAMISKO SAKABES MATRICU.

    Dinamika:
      dθᵢ/dt = ωᵢ + (1/N) · Σⱼ K_ij · sin(θⱼ − θᵢ)

    Mācīšanās (Hebbian Phase Plasticity):
      ΔK_ij = learning_rate · Q_joy · cos(θᵢ − θⱼ)

    Parametri
    ---------
    n_oscillators : oscilatoru skaits (pēc noklusējuma = len(BASE_HARMONICS))
    K             : sākotnējā sakabes bāzes vērtība (skalārs vai matrica)
    omega         : dabiskās frekvences ωᵢ; ja None, ģenerē no BASE_HARMONICS
    theta0        : sākotnējie leņķi θᵢ; ja None, izvēlas nejaušus
    seed          : nejaušības sēkla reproducējamībai
    use_matrix    : vai izmantot K_matrix (True) vai skalāru K (False, atpakaļsaderība)
    """
    n_oscillators: int = len(BASE_HARMONICS)
    K: float = 2.0
    omega: np.ndarray | None = None
    theta0: np.ndarray | None = None
    seed: int | None = 42
    use_matrix: bool = True  # Jaunā plastiskuma funkcionalitāte

    # Iekšējais stāvoklis (inicializēts pēc __post_init__)
    theta: np.ndarray = field(init=False)
    _omega: np.ndarray = field(init=False)
    K_matrix: np.ndarray = field(init=False)  # Dinamiskā sakabes matrica (N, N)
    K_base: float = field(init=False)          # Bāzes vērtība decay mehānismam
    history: list[np.ndarray] = field(default_factory=list)
    learning_history: list[float] = field(default_factory=list)  # Q_joy vēsture mācīšanās laikā

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Saglabā bāzes K vērtību decay mehānismam
        self.K_base = self.K

        # Inicializē sakabes matricu K_ij
        N = self.n_oscillators
        self.K_matrix = np.full((N, N), self.K, dtype=float)
        # Diagonālē nav pašsaites
        np.fill_diagonal(self.K_matrix, 0.0)

        # Dabiskās frekvences
        if self.omega is not None:
            self._omega = np.asarray(self.omega, dtype=float)
        else:
            # Frekvences no BASE_HARMONICS (ar PHI svēršanu)
            self._omega = BASE_HARMONICS[:self.n_oscillators] * (2 * np.pi / PHI)

        # Sākotnējie leņķi
        if self.theta0 is not None:
            self.theta = np.asarray(self.theta0, dtype=float).copy()
        else:
            self.theta = rng.uniform(0, 2 * np.pi, self.n_oscillators)

        self.history = [self.theta.copy()]
        self.learning_history = []

    # ── Kuramoto labā puse ─────────────────────

    def _dtheta_dt(self) -> np.ndarray:
        """
        Aprēķina dθᵢ/dt katram oscilatoram.

        Ar matricu: dθᵢ/dt = ωᵢ + (1/N) · Σⱼ K_ij · sin(θⱼ − θᵢ)
        Bez matricas: dθᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(θⱼ − θᵢ)
        """
        N = self.n_oscillators
        # Visu pāru starpības matricā: diff[i,j] = θⱼ − θᵢ
        diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis]  # (N, N)

        if self.use_matrix:
            # Dinamiskā matrica: katrs pāris ar savu K_ij
            weighted_sin = self.K_matrix * np.sin(diff)  # (N, N)
            coupling = (1.0 / N) * np.sum(weighted_sin, axis=1)
        else:
            # Atpakaļsaderīgais skalārais režīms
            coupling = (self.K / N) * np.sum(np.sin(diff), axis=1)

        return self._omega + coupling

    # ── Integrācija (Runge-Kutta 4) ────────────

    def step(self, dt: float = 0.01) -> None:
        """Viena laika soļa RK4 integrācija."""
        th = self.theta

        def f(t):
            self.theta = t
            return self._dtheta_dt()

        k1 = f(th)
        k2 = f(th + 0.5 * dt * k1)
        k3 = f(th + 0.5 * dt * k2)
        k4 = f(th + dt * k3)

        self.theta = th + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        self.theta = self.theta % (2 * np.pi)
        self.history.append(self.theta.copy())

    def run(self, steps: int = 500, dt: float = 0.01) -> np.ndarray:
        """
        Izskrien `steps` integrācijas soļus.

        Returns
        -------
        np.ndarray, forma (steps+1, N) : visu oscilatoru θ katra solī
        """
        for _ in range(steps):
            self.step(dt)
        return np.array(self.history)

    # ── Sinhronizācijas kārtības parametrs ─────

    def order_parameter(self) -> tuple[float, float]:
        """
        Kuramoto kārtības parametrs:  R·e^{iΨ} = (1/N)·Σ e^{iθⱼ}

        Returns
        -------
        (R, Psi) : R ∈ [0,1] — sinhronizācijas pakāpe; Psi — kopīgā fāze
        """
        z = np.mean(np.exp(1j * self.theta))
        return float(np.abs(z)), float(np.angle(z))

    def reset(self, seed: int | None = None, reset_matrix: bool = False) -> None:
        """Atjauno oscilatoru sākotnējos nejaušos leņķus."""
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        self.theta = rng.uniform(0, 2 * np.pi, self.n_oscillators)
        self.history = [self.theta.copy()]

        if reset_matrix:
            # Atjauno K_matrix uz bāzes vērtību
            self.K_matrix = np.full((self.n_oscillators, self.n_oscillators),
                                     self.K_base, dtype=float)
            np.fill_diagonal(self.K_matrix, 0.0)
            self.learning_history = []

    # ── HEBBIAN PHASE LEARNING ─────────────────

    def learn(self,
              q_joy_value: float,
              learning_rate: float = 0.1,
              apply_decay: bool = True) -> np.ndarray:
        """
        Rezonanses Plastiskums: Hebbian Phase Learning.

        Mācīšanās noteikums:
          ΔK_ij = learning_rate · Q_joy · cos(θᵢ − θⱼ)

        Ja oscilatori i un j ir saskaņā (leņķu starpība ≈ 0) un Q_joy > 0,
        to savstarpējā saite K_ij kļūst stiprāka.
        Ja Q_joy < 0 (disonanse), saite vājinās.

        Parameters
        ----------
        q_joy_value   : pašreizējā Q_joy vērtība
        learning_rate : mācīšanās ātrums (noklusējums 0.1)
        apply_decay   : vai pielietot Phi Decay pēc mācīšanās

        Returns
        -------
        np.ndarray : ΔK_ij matrica (izmaiņu vizualizācijai)
        """
        N = self.n_oscillators

        # Fāžu starpību matrica: diff[i,j] = θᵢ − θⱼ
        diff = self.theta[:, np.newaxis] - self.theta[np.newaxis, :]  # (N, N)

        # Hebbian noteikums: stiprinām saites starp saskaņotiem oscilatoriem
        delta_K = learning_rate * q_joy_value * np.cos(diff)

        # Neļauj pašsaitēm mainīties
        np.fill_diagonal(delta_K, 0.0)

        # Pielieto izmaiņas
        self.K_matrix += delta_K

        # Nodrošina, ka K_ij ≥ 0 (nav negatīvas saites)
        self.K_matrix = np.maximum(self.K_matrix, 0.0)

        # Phi Decay — lēns atgriešanās uz bāzes vērtību
        if apply_decay:
            self.phi_decay()

        # Saglabā Q_joy vēsturi
        self.learning_history.append(q_joy_value)

        return delta_K

    def phi_decay(self, decay_power: float = 0.02) -> None:
        """
        Zelta proporcijas samazinājums (Phi Decay).

        Pēc katras mācīšanās, K_ij minimāli "atdziest" virzienā uz K_base:
          K_ij ← K_base + (K_ij - K_base) / φ^decay_power

        Tas novērš bezgalīgu matricas augšanu vai sabrukumu.

        Parameters
        ----------
        decay_power : pakāpe, kurā φ tiek celta (mazāks = lēnāks decay)
        """
        decay_factor = PHI ** decay_power  # ≈ 1.01 pie 0.02

        # Matrica tuvojas bāzes vērtībai
        deviation = self.K_matrix - self.K_base
        self.K_matrix = self.K_base + deviation / decay_factor

        # Diagonāle paliek 0
        np.fill_diagonal(self.K_matrix, 0.0)

    def coupling_strength(self) -> float:
        """
        Atgriež kopējo vidējo sakabes spēku (K_ij vidējais bez diagonāles).
        """
        N = self.n_oscillators
        mask = ~np.eye(N, dtype=bool)
        return float(np.mean(self.K_matrix[mask]))

    def coupling_variance(self) -> float:
        """
        Atgriež K_ij varianci — cik daudz saites atšķiras viena no otras.
        Augsta variance = sistēma ir "iemācījusies" specifiskas saites.
        """
        N = self.n_oscillators
        mask = ~np.eye(N, dtype=bool)
        return float(np.var(self.K_matrix[mask]))

    def strongest_connections(self, top_n: int = 5) -> list[tuple[int, int, float]]:
        """
        Atgriež top_n stiprākās saites (i, j, K_ij).
        """
        N = self.n_oscillators
        connections = []
        for i in range(N):
            for j in range(N):
                if i != j:
                    connections.append((i, j, self.K_matrix[i, j]))
        connections.sort(key=lambda x: -x[2])
        return connections[:top_n]

    # ── ATMIŅAS SAGLABĀŠANA / IELĀDE ───────────

    def save_memory(self, filepath: str = "phaseflow_memory.npz") -> str:
        """
        Saglabā K_matrix un mācīšanās vēsturi failā.

        Returns
        -------
        str : pilns ceļš uz saglabāto failu
        """
        from pathlib import Path
        filepath = Path(filepath)

        np.savez(
            filepath,
            K_matrix=self.K_matrix,
            K_base=self.K_base,
            learning_history=np.array(self.learning_history),
            n_oscillators=self.n_oscillators,
        )

        return str(filepath.resolve())

    def load_memory(self, filepath: str = "phaseflow_memory.npz") -> bool:
        """
        Ielādē K_matrix no faila.

        Returns
        -------
        bool : True ja veiksmīgi ielādēts, False ja fails neeksistē
        """
        from pathlib import Path
        filepath = Path(filepath)

        if not filepath.exists():
            return False

        data = np.load(filepath)

        # Pārbauda, vai dimensijas sakrīt
        if data["n_oscillators"] != self.n_oscillators:
            raise ValueError(
                f"Nesaderīgas dimensijas: failā {data['n_oscillators']}, "
                f"oscilatorā {self.n_oscillators}"
            )

        self.K_matrix = data["K_matrix"]
        self.K_base = float(data["K_base"])
        self.learning_history = list(data["learning_history"])

        return True


# ─────────────────────────────────────────────
# 4. Q_JOY — SINHRONIZĀCIJAS ATALGOJUMS
# ─────────────────────────────────────────────

def q_joy(oscillator: KuramotoOscillator) -> float:
    """
    Q_joy(t) = R² · cos(Ψ − ψ₄₂)

    Atalgojuma funkcija: maksimāla, kad oscilatoru kopā-fāze Ψ sakrīt ar ψ₄₂
    un sinhronizācija R ir augsta.

    Returns
    -------
    float ∈ [−1, 1]  (negatīvs = disonanse, pozitīvs = rezonanse)
    """
    R, Psi = oscillator.order_parameter()
    return R**2 * np.cos(Psi - PSI_42)


# ─────────────────────────────────────────────
# 5. FFT REZONATORA FILTRS
# ─────────────────────────────────────────────

def resonance_score(signal: Sequence[float] | np.ndarray,
                    sample_rate: float = 1.0,
                    harmonics: np.ndarray = BASE_HARMONICS,
                    bandwidth: float = 0.5) -> float:
    """
    Pārbauda ievades signāla rezonanci ar bāzes harmonikas kopu caur FFT.

    Algoritms:
      1. FFT → frekvences spektrs
      2. Katrai bāzes harmonikai iegūst amplitūdu logā [n−bw, n+bw]
      3. Kopējā rezonanse = Σ amplitūdas(harmonika) / kopējā amplitūda

    Parameters
    ----------
    signal      : laika rindas signāls
    sample_rate : paraugu frekvence (Hz)
    harmonics   : meklējamās harmoniku frekvences
    bandwidth   : meklēšanas josla ap katru harmoniku

    Returns
    -------
    float ∈ [0, 1]  — 1.0 = pilnīga rezonanse ar bāzes harmonikas kopu
    """
    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    if N < 4:
        raise ValueError("Signālam jābūt vismaz 4 punktiem.")

    spectrum = np.abs(fft(sig))[:N // 2]
    freqs = fftfreq(N, d=1.0 / sample_rate)[:N // 2]

    total_power = np.sum(spectrum**2)
    if total_power == 0:
        return 0.0

    harmonic_power = 0.0
    for h in harmonics:
        mask = (freqs >= h - bandwidth) & (freqs <= h + bandwidth)
        harmonic_power += np.sum(spectrum[mask]**2)

    return float(np.clip(harmonic_power / total_power, 0.0, 1.0))


def resonance_gate(signal: Sequence[float] | np.ndarray,
                   sample_rate: float = 1.0,
                   threshold: float = LAMBDA_21,
                   harmonics: np.ndarray = BASE_HARMONICS) -> tuple[bool, float]:
    """
    Rezonatora vārti: atgriež (iztur, score).

    Ja score ≥ LAMBDA_21 (≈ 0.382), signāls "iztur" rezonanses pārbaudi.
    Pretējā gadījumā sistēma aicina mēģināt vēlreiz.

    Returns
    -------
    (passed: bool, score: float)
    """
    score = resonance_score(signal, sample_rate=sample_rate, harmonics=harmonics)
    passed = score >= threshold
    if not passed:
        print(f"[Rezonatora vārti] score={score:.4f} < λ₂₁={threshold:.4f} "
              "— signāls nerezonē. Lūdzu, mēģiniet vēlreiz.")
    return passed, score


# ─────────────────────────────────────────────
# 6. VIZUALIZĀCIJAS PALĪGFUNKCIJAS
# ─────────────────────────────────────────────

def plot_phases(oscillator: KuramotoOscillator,
                title: str = "Oscilatoru fāzes uz vienības apļa") -> None:
    """Attēlo aktīvo oscilatoru fāzes polārajā koordinātu sistēmā."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib nav instalēts. Palaidiet: pip install matplotlib")

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    z = np.exp(1j * oscillator.theta)
    colors = plt.cm.plasma(np.linspace(0, 1, len(oscillator.theta)))

    for i, (angle, color) in enumerate(zip(oscillator.theta, colors)):
        ax.plot([0, angle], [0, 1.0], color=color, lw=2,
                label=f"n={BASE_HARMONICS[i]:.0f}")
        ax.scatter(angle, 1.0, color=color, s=80, zorder=5)

    # Kārtības parametra bulta
    R, Psi = oscillator.order_parameter()
    ax.annotate("", xy=(Psi, R), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=2))

    ax.set_title(f"{title}\nR = {R:.3f}, Q_joy = {q_joy(oscillator):.3f}",
                 pad=15, color="white")
    ax.set_facecolor("#0d0d1a")
    fig.patch.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
              framealpha=0.3, labelcolor="white")
    plt.tight_layout()
    plt.show()


def plot_synchronization(history: np.ndarray,
                         dt: float = 0.01,
                         title: str = "Sinhronizācijas dinamika") -> None:
    """
    Attēlo kārtības parametra R(t) un oscilatoru fāžu θᵢ(t) evolūciju.

    Parameters
    ----------
    history : masīvs, forma (T, N) — θ vērtības katrā laika solī
    dt      : laika solis integrācijā
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib nav instalēts. Palaidiet: pip install matplotlib")

    T, N = history.shape
    t = np.arange(T) * dt

    # Kārtības parametrs katrā laika solī
    z_mean = np.mean(np.exp(1j * history), axis=1)
    R_t = np.abs(z_mean)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6),
                             facecolor="#0d0d1a", sharex=True)

    # — Augšā: fāžu trajektorijas
    ax1 = axes[0]
    colors = plt.cm.plasma(np.linspace(0, 1, N))
    for i in range(N):
        ax1.plot(t, history[:, i], color=colors[i], lw=1.2, alpha=0.85,
                 label=f"n={BASE_HARMONICS[i]:.0f}")
    ax1.set_ylabel("θᵢ (rad)", color="white")
    ax1.set_title(title, color="white")
    ax1.set_facecolor("#0d0d1a")
    ax1.tick_params(colors="white")
    ax1.legend(loc="upper right", framealpha=0.3, labelcolor="white")
    ax1.axhline(PSI_42, color="cyan", lw=1, ls="--", alpha=0.6,
                label="ψ₄₂")

    # — Apakšā: R(t)
    ax2 = axes[1]
    ax2.plot(t, R_t, color="#f0a500", lw=2)
    ax2.axhline(LAMBDA_21, color="red", lw=1, ls="--", alpha=0.8,
                label=f"λ₂₁ = {LAMBDA_21:.3f}")
    ax2.fill_between(t, R_t, LAMBDA_21,
                     where=(R_t >= LAMBDA_21), alpha=0.25, color="#f0a500")
    ax2.set_ylabel("R(t) — kārtības parametrs", color="white")
    ax2.set_xlabel("Laiks (s)", color="white")
    ax2.set_ylim(0, 1.05)
    ax2.set_facecolor("#0d0d1a")
    ax2.tick_params(colors="white")
    ax2.legend(loc="lower right", framealpha=0.3, labelcolor="white")

    plt.tight_layout()
    plt.show()


def plot_fft_resonance(signal: Sequence[float] | np.ndarray,
                       sample_rate: float = 1.0,
                       harmonics: np.ndarray = BASE_HARMONICS) -> None:
    """Attēlo FFT spektru ar iezīmētām bāzes harmonikas."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib nav instalēts. Palaidiet: pip install matplotlib")

    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    spectrum = np.abs(fft(sig))[:N // 2]
    freqs = fftfreq(N, d=1.0 / sample_rate)[:N // 2]
    score = resonance_score(sig, sample_rate=sample_rate, harmonics=harmonics)

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d0d1a")
    ax.plot(freqs, spectrum, color="#4fc3f7", lw=1.5, label="Spektrs")

    for h in harmonics:
        ax.axvline(h, color="#f0a500", lw=1.5, ls="--", alpha=0.8)
        ax.text(h, max(spectrum) * 0.95, f"n={h:.0f}",
                color="#f0a500", ha="center", fontsize=8)

    ax.set_title(f"FFT Rezonances Analīze  |  score = {score:.4f} "
                 f"({'✓ rezonē' if score >= LAMBDA_21 else '✗ nerezonē'})",
                 color="white")
    ax.set_xlabel("Frekvence", color="white")
    ax.set_ylabel("Amplitūda", color="white")
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="white")
    ax.legend(framealpha=0.3, labelcolor="white")
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# ĀTRAIS TESTS — AR REZONANSES PLASTISKUMU
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("═" * 60)
    print("  PhaseFlow v1.1 — ar Rezonanses Plastiskumu (Hebbian)")
    print("═" * 60)
    print(f"  φ  (Zelta proporcija)  = {PHI:.8f}")
    print(f"  ψ₄₂ (atsauces fāze)   = {PSI_42:.8f} rad")
    print(f"  λ₂₁ (rezonanses slieksnis) = {LAMBDA_21:.8f}")
    print(f"  Bāzes harmoniku kopums: {BASE_HARMONICS.tolist()}")
    print()

    # ── A) Bāzes tests ar K_matrix ────────────────────────
    print("▸ A) Kuramoto ar dinamisko K_matrix")
    print("-" * 50)

    osc = KuramotoOscillator(K=2.0, use_matrix=True, seed=42)
    print(f"  Sākotnējais K_matrix vidējais: {osc.coupling_strength():.4f}")
    print(f"  K_matrix variance: {osc.coupling_variance():.6f}")

    R0, Psi0 = osc.order_parameter()
    Q0 = q_joy(osc)
    print(f"  Sākums: R={R0:.4f}, Q_joy={Q0:.4f}")

    # Sinhronizācijas cikls
    hist = osc.run(steps=300, dt=0.02)
    R1, Psi1 = osc.order_parameter()
    Q1 = q_joy(osc)
    print(f"  Pēc 300 soļiem: R={R1:.4f}, Q_joy={Q1:.4f}")
    print()

    # ── B) Mācīšanās demonstrācija ────────────────────────
    print("▸ B) Rezonanses Plastiskums — Mācīšanās demonstrācija")
    print("-" * 50)
    print("  Izpildīsim 5 mācīšanās ciklus ar augstu Q_joy...")
    print()

    for cycle in range(1, 6):
        # Atjauno fāzes (bet saglabā K_matrix!)
        osc.reset(seed=cycle * 10, reset_matrix=False)

        # Sinhronizē
        osc.run(steps=200, dt=0.02)
        R, Psi = osc.order_parameter()
        Q = q_joy(osc)

        # Mācīšanās ar pašreizējo Q_joy
        delta_K = osc.learn(Q, learning_rate=0.15, apply_decay=True)

        # Progresa atskaite
        mean_K = osc.coupling_strength()
        var_K = osc.coupling_variance()
        max_delta = np.max(np.abs(delta_K))

        print(f"  Cikls {cycle}: R={R:.4f}, Q_joy={Q:+.4f} | "
              f"K̄={mean_K:.4f}, var={var_K:.6f}, |ΔK|_max={max_delta:.4f}")

    print()

    # ── C) Stiprākās saites ───────────────────────────────
    print("▸ C) Stiprākās saites pēc mācīšanās")
    print("-" * 50)
    top_connections = osc.strongest_connections(top_n=5)
    for i, j, K_ij in top_connections:
        harm_i = BASE_HARMONICS[i] if i < len(BASE_HARMONICS) else i
        harm_j = BASE_HARMONICS[j] if j < len(BASE_HARMONICS) else j
        print(f"  n={harm_i:.0f} ↔ n={harm_j:.0f} : K={K_ij:.4f}")
    print()

    # ── D) K_matrix vizualizācija (teksta) ────────────────
    print("▸ D) K_matrix (simetriska sakabes matrica)")
    print("-" * 50)
    print("      ", end="")
    for j in range(osc.n_oscillators):
        print(f"  n={BASE_HARMONICS[j]:.0f}  ", end="")
    print()
    for i in range(osc.n_oscillators):
        print(f"  n={BASE_HARMONICS[i]:.0f}", end=" ")
        for j in range(osc.n_oscillators):
            K_ij = osc.K_matrix[i, j]
            if i == j:
                print("    -   ", end="")
            else:
                print(f"  {K_ij:.3f} ", end="")
        print()
    print()

    # ── E) Atmiņas saglabāšana ────────────────────────────
    print("▸ E) Atmiņas saglabāšana un ielāde")
    print("-" * 50)
    memory_path = osc.save_memory("phaseflow_memory.npz")
    print(f"  Atmiņa saglabāta: {memory_path}")

    # Izveido jaunu oscilatoru un ielādē atmiņu
    osc_new = KuramotoOscillator(K=2.0, use_matrix=True, seed=999)
    print(f"  Jauns oscilators (pirms ielādes): K̄={osc_new.coupling_strength():.4f}")

    loaded = osc_new.load_memory("phaseflow_memory.npz")
    print(f"  Pēc atmiņas ielādes: K̄={osc_new.coupling_strength():.4f}")
    print(f"  Mācīšanās vēsture: {len(osc_new.learning_history)} cikli")
    print()

    # ── F) Salīdzinājums: ar/bez mācīšanās ────────────────
    print("▸ F) Salīdzinājums: sinhronizācijas ātrums")
    print("-" * 50)

    # Bez mācīšanās (svaiga matrica)
    osc_fresh = KuramotoOscillator(K=2.0, use_matrix=True, seed=123)
    steps_to_sync_fresh = 0
    for s in range(500):
        osc_fresh.step(0.02)
        R, _ = osc_fresh.order_parameter()
        if R > 0.7:
            steps_to_sync_fresh = s
            break

    # Ar mācīšanos (ielādēta atmiņa)
    osc_learned = KuramotoOscillator(K=2.0, use_matrix=True, seed=123)
    osc_learned.load_memory("phaseflow_memory.npz")
    steps_to_sync_learned = 0
    for s in range(500):
        osc_learned.step(0.02)
        R, _ = osc_learned.order_parameter()
        if R > 0.7:
            steps_to_sync_learned = s
            break

    print(f"  Bez mācīšanās: R>0.7 pēc {steps_to_sync_fresh} soļiem")
    print(f"  Ar mācīšanos:  R>0.7 pēc {steps_to_sync_learned} soļiem")
    if steps_to_sync_learned < steps_to_sync_fresh:
        speedup = (steps_to_sync_fresh - steps_to_sync_learned) / max(steps_to_sync_fresh, 1) * 100
        print(f"  💡 Ātrums pieauga par {speedup:.1f}%!")
    print()

    # ── G) FFT tests ──────────────────────────────────────
    print("▸ G) FFT rezonanses tests")
    print("-" * 50)
    t = np.linspace(0, 10, 1024)
    test_signal = (np.sin(2 * np.pi * 1.0 * t) +
                   np.sin(2 * np.pi * 3.0 * t) * 0.7 +
                   np.sin(2 * np.pi * 9.0 * t) * 0.4)
    passed, score = resonance_gate(test_signal, sample_rate=1024 / 10)
    print(f"  Rezonanses vārti: score={score:.4f}, iztur={passed}")

    print()
    print("═" * 60)
    print("  PhaseFlow v1.1 — Rezonanses Plastiskums aktīvs")
    print("  Sistēma tagad mācās caur Q_joy un atceras saites!")
    print("═" * 60)
