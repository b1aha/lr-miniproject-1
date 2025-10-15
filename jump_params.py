"""
~/jump_params.py
Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
École Polytechnique Fédérale de Lausanne,
CH-1015 Lausanne,
Switzerland
"""

JUMP_PARAMS = {
    "forward": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 140.0,
        "FORCE_FY": 0.0,
        "FORCE_FZ": 370.0,
        "N_JUMPS": 15,
        "K_VMC": 0.0,
        "Z_OFFSET": -0.220,
        "Y_OFFSET": 0.0838,
        "X_OFFSET": 0.0011,
    },
    "lateral_left": {
        "IMPULSE_F0": 1.2,
        "IDLE_F1": 0.8,
        "FORCE_FX": 0.0,
        "FORCE_FY": 40.0,
        "FORCE_FZ": 230.0,
        "N_JUMPS": 15,
        "K_VMC": 8.0,
        "Z_OFFSET": -0.230,
        "Y_OFFSET": 0.0900,
        "X_OFFSET": 0.0000,
    },
    "lateral_right": {
        "IMPULSE_F0": 1.2,
        "IDLE_F1": 0.8,
        "FORCE_FX": 0.0,
        "FORCE_FY": -40.0,
        "FORCE_FZ": 230.0,
        "N_JUMPS": 15,
        "K_VMC": 10.0,
        # "K_VMC": 100.0, # FOR VMC COMPARISON
        "Z_OFFSET": -0.230,
        "Y_OFFSET": 0.1050,
        "X_OFFSET": 0.0000,
    },
    "twist_ccw": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 80.0,
        "FORCE_FY": 80.0,
        "FORCE_FZ": 320.0,
        "N_JUMPS": 15,
        "K_VMC": 10.0,
        "Z_OFFSET": -0.220,
        "Y_OFFSET": 0.0838,
        "X_OFFSET": 0.0041,
    },
    "twist_cw": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 80.0,
        "FORCE_FY": 80.0,
        "FORCE_FZ": 320.0,
        "N_JUMPS": 15,
        "K_VMC": 10.0,
        "Z_OFFSET": -0.220,
        "Y_OFFSET": 0.0838,
        "X_OFFSET": 0.0041,
    },
}
