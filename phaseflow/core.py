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
    N savstarpēji saistītu oscilatoru Kuramoto modelis.

    Dinamika:
      dθᵢ/dt = ωᵢ + (K / N) · Σⱼ sin(θⱼ − θᵢ)

    Parametri
    ---------
    n_oscillators : oscilatoru skaits (pēc noklusējuma = len(BASE_HARMONICS))
    K             : sakabes koeficients λ_tH — lielāks K → ātrāka sinhronizācija
    omega         : dabiskās frekvences ωᵢ; ja None, ģenerē no BASE_HARMONICS
    theta0        : sākotnējie leņķi θᵢ; ja None, izvēlas nejaušus
    seed          : nejaušības sēkla reproducējamībai
    """
    n_oscillators: int = len(BASE_HARMONICS)
    K: float = 2.0
    omega: np.ndarray | None = None
    theta0: np.ndarray | None = None
    seed: int | None = 42

    # Iekšējais stāvoklis (inicializēts pēc __post_init__)
    theta: np.ndarray = field(init=False)
    _omega: np.ndarray = field(init=False)
    history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)

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

    # ── Kuramoto labā puse ─────────────────────

    def _dtheta_dt(self) -> np.ndarray:
        """
        Aprēķina dθᵢ/dt katram oscilatoram.

        dθᵢ/dt = ωᵢ + (K/N) · Σⱼ sin(θⱼ − θᵢ)
        """
        N = self.n_oscillators
        # Visu pāru starpības matricā (jxī)
        diff = self.theta[np.newaxis, :] - self.theta[:, np.newaxis]  # (N, N)
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

    def reset(self, seed: int | None = None) -> None:
        """Atjauno oscilatoru sākotnējos nejaušos leņķus."""
        rng = np.random.default_rng(seed if seed is not None else self.seed)
        self.theta = rng.uniform(0, 2 * np.pi, self.n_oscillators)
        self.history = [self.theta.copy()]


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
# ĀTRAIS TESTS
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("── PhaseFlow v1.0 ─────────────────────────")
    print(f"  φ  (Zelta proporcija)  = {PHI:.8f}")
    print(f"  ψ₄₂ (atsauces fāze)   = {PSI_42:.8f} rad")
    print(f"  λ₂₁ (rezonanses slieksnis) = {LAMBDA_21:.8f}")
    print(f"  Bāzes harmoniku kopums: {BASE_HARMONICS.tolist()}")

    phases = harmonic_phases()
    print(f"\nHarmoniku fāzes (e^{{i·n·ψ₄₂}}):")
    for n, z in zip(BASE_HARMONICS, phases):
        print(f"  n={int(n):2d}  →  {z.real:+.4f} + {z.imag:+.4f}i  "
              f"(|z|={abs(z):.4f}, θ={np.angle(z):.4f} rad)")

    print("\nKuramoto oscilatoru inicializācija (K=2.0) ...")
    osc = KuramotoOscillator(K=2.0)
    R0, Psi0 = osc.order_parameter()
    print(f"  Sākums: R={R0:.3f}, Q_joy={q_joy(osc):.3f}")

    hist = osc.run(steps=500, dt=0.02)
    R1, Psi1 = osc.order_parameter()
    print(f"  Pēc 500 soļiem: R={R1:.3f}, Q_joy={q_joy(osc):.3f}")

    # FFT rezonances tests
    t = np.linspace(0, 10, 1024)
    test_signal = (np.sin(2 * np.pi * 1.0 * t) +
                   np.sin(2 * np.pi * 3.0 * t) * 0.7 +
                   np.sin(2 * np.pi * 9.0 * t) * 0.4)
    passed, score = resonance_gate(test_signal, sample_rate=1024 / 10)
    print(f"\nRezonanses vārti: score={score:.4f}, iztur={passed}")
    print("───────────────────────────────────────────")
