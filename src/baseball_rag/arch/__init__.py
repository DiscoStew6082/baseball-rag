"""Baseball RAG — Architecture Explorer module.

Exports the public API for the diagram component registry and tracing primitives.
"""

from baseball_rag.arch.components import (
    ComponentTestStatus,
    DiagramComponent,
    Layer,
    get_components_by_layer,
    get_registry,
    get_source_snippet,
)
from baseball_rag.arch.diagram import ArchitectureDiagram
from baseball_rag.arch.tracing import PipelineStage, PipelineTrace

# Public alias
TestStatus = ComponentTestStatus

__all__ = [
    "ArchitectureDiagram",
    "DiagramComponent",
    "Layer",
    "PipelineStage",
    "PipelineTrace",
    "TestStatus",  # alias for backward compat
    "get_registry",
    "get_components_by_layer",
    "get_source_snippet",
]
