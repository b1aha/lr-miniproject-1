"""
~/jump_params_opt.py
Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
École Polytechnique Fédérale de Lausanne,
CH-1015 Lausanne,
Switzerland
"""

DECISION_VARS = {
    "forward": {
        "f0": {"low": 0.8, "high": 6.0},
        "f1": {"low": 0.2, "high": 3.0},
        "Fx": {"low": 0.0, "high": 400.0},
        "Fy": {"low": 0.0, "high": 0.0},
        "Fz": {"low": 200.0, "high": 700.0}
    },
    "lateral_left": {
        "f0": {"low": 0.8, "high": 6.0},
        "f1": {"low": 0.2, "high": 3.0},
        "Fx": {"low": 0.0, "high": 0.0},
        "Fy": {"low": 0.0, "high": 100.0},
        "Fz": {"low": 200.0, "high": 300.0}
    },
    "lateral_right": {
        "f0": {"low": 0.8, "high": 6.0},
        "f1": {"low": 0.2, "high": 3.0},
        "Fx": {"low": 0.0, "high": 0.0},
        "Fy": {"low": -100.0, "high": 0.0},
        "Fz": {"low": 200.0, "high": 300.0}
    },
    "twist_ccw": {
        "f0": {"low": 0.8, "high": 6.0},
        "f1": {"low": 0.2, "high": 3.0},
        "Fx": {"low": 0.0, "high": 250.0},
        "Fy": {"low": 0.0, "high": 250.0},
        "Fz": {"low": 200.0, "high": 650.0}
    },
    "twist_cw": {
        "f0": {"low": 0.8, "high": 6.0},
        "f1": {"low": 0.2, "high": 3.0},
        "Fx": {"low": 0.0, "high": 250.0},
        "Fy": {"low": 0.0, "high": 250.0},
        "Fz": {"low": 200.0, "high": 650.0}
    },
    "hopping": {
        "f0": {"low": 0.8, "high": 6.0},
        "f1": {"low": 0.2, "high": 3.0},
        "Fx": {"low": 0.0, "high": 400.0},
        "Fy": {"low": 0.0, "high": 0.0},
        "Fz": {"low": 200.0, "high": 700.0}
    }
}
