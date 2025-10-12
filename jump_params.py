# ~/jump_params.py
# Oct 2025; Jiri Blaha, Giulia Cortelazzo, Semih Zaman
# École Polytechnique Fédérale de Lausanne,
# CH-1015 Lausanne,
# Switzerland
# ================================================
JUMP_PARAMS = {
    "forward": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 140.0,
        "FORCE_FY": 0.0,
        "FORCE_FZ": 370.0,
        "N_JUMPS": 10,
        "K_VMC": 0.0,
    },
    "lateral_left": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 0.0,
        "FORCE_FY": 140.0,
        "FORCE_FZ": 320.0,
        "N_JUMPS": 10,
        "K_VMC": 0.0,
    },
    "lateral_right": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 0.0,
        "FORCE_FY": -140.0,
        "FORCE_FZ": 320.0,
        "N_JUMPS": 10,
        "K_VMC": 0.0,
    },
    "twist_ccw": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 80.0,
        "FORCE_FY": 80.0,
        "FORCE_FZ": 320.0,
        "N_JUMPS": 20,
        "K_VMC": 0.0,
    },
    "twist_cw": {
        "IMPULSE_F0": 3.8,
        "IDLE_F1": 1.0,
        "FORCE_FX": 80.0,
        "FORCE_FY": 80.0,
        "FORCE_FZ": 320.0,
        "N_JUMPS": 20,
        "K_VMC": 0.0,
    },
}
