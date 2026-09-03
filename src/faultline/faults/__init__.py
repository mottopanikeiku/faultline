"""Latent fault descriptors and injection helpers."""

from faultline.faults.base import FaultKind, LatentFault, inject_fault
from faultline.faults.machine import DownstreamBackpressure, FailedProcessor
from faultline.faults.transport import BlockedEdge

__all__ = [
    "BlockedEdge",
    "DownstreamBackpressure",
    "FailedProcessor",
    "FaultKind",
    "LatentFault",
    "inject_fault",
]
