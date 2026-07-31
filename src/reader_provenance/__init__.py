"""READER: dynamic LLM provenance from query-varying interactions."""

from reader_provenance.features.dct import encode_padded_trajectories
from reader_provenance.training.probe import LinearSourceProbe, fit_source_probe

__all__ = [
    "LinearSourceProbe",
    "encode_padded_trajectories",
    "fit_source_probe",
]

__version__ = "1.0.0"
