"""
Shared plotting styles keyed by method label.

Keep all figure styles consistent across plotting functions by importing STYLE_MAP.
"""

STYLE_MAP = {
    # solid black with star markers for classical baseline
    "Classical ALS": {"color": "black", "linestyle": "-", "marker": "*", "label": "C-ALS"},
    # blue dash-dot with dot markers for quantum/VQLS variant
    "Quantum ALS": {"color": "C0", "linestyle": "-.", "marker": "o", "label": "Q-ALS (ours)"},
}


def get_style(label: str):
    """
    Return the preferred style dict for a given label, or a minimal fallback.
    """
    return STYLE_MAP.get(label, {"label": label})
