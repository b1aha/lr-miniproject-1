# ~/profiles.py
# Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
# École Polytechnique Fédérale de Lausanne,
# CH-1015 Lausanne,
# Switzerland
# ================================================
import numpy as np


class FootForceProfile:
    def __init__(self, f0: float, f1: float, Fx: float, Fy: float, Fz: float):
        self.theta = 0.0
        self.f0 = float(f0)
        self.f1 = float(f1)
        self.F = np.array([Fx, Fy, Fz], dtype=float)

    def step(self, dt: float):
        fi = self.f0 if np.sin(self.theta) < 0.0 else self.f1
        self.theta += 2.0 * np.pi * fi * dt
        self.theta %= 2.0 * np.pi

    def phase(self) -> float:
        return float(self.theta)

    def force(self) -> np.ndarray:
        s = np.sin(self.theta)
        if s < 0.0:
            return self.F * s
        return np.zeros(3)

    def impulse_duration(self) -> float:
        return 1.0 / (2.0 * self.f0)

    def idle_duration(self) -> float:
        return 1.0 / (2.0 * self.f1)
