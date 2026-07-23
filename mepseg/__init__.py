"""mepseg: segmentazione di nuvole di punti per impianti MEP (scan-to-BIM)."""
from .classi import ClasseMEP, COLORI_CLASSI, NOMI_CLASSI
from .io_nuvole import Nuvola, carica_nuvola
from .pipeline import RisultatoSegmentazione, segmenta

__version__ = "0.1.0"
__all__ = [
    "ClasseMEP",
    "COLORI_CLASSI",
    "NOMI_CLASSI",
    "Nuvola",
    "carica_nuvola",
    "RisultatoSegmentazione",
    "segmenta",
    "__version__",
]
