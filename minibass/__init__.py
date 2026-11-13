"""MiniBass: Physical acoustics, discrete optimization, and active low-frequency sound field synthesis.

Exports:
- physics: rectangular_transfer, evaluate_source, ActiveFilter, Driver, Design, Port, Environment
- data_generation: create_10x12_room_instance, create_driver_catalog, create_enclosure_designs, precompute_atom_tensors
- algorithms: VectorizedGeometricScanner, MultiSeatMinimaxOptimizer, RLTActiveBilinearSolver, EvaluationResult
- dryrun_10x12: run_10x12_dryrun, evaluate_system_configuration
"""

from .algorithms import (
    EvaluationResult,
    LayeredBeamSearchOptimizer,
    MultiSeatMinimaxOptimizer,
    RLTActiveBilinearSolver,
    VectorizedGeometricScanner,
    prune_nondominated,
    search_instance,
    select_beam,
)
from .data_generation import (
    active_filter_from_record,
    create_10x12_room_instance,
    create_control_records,
    create_driver_catalog,
    create_enclosure_designs,
    precompute_atom_tensors,
)
from .dryrun_10x12 import (
    evaluate_system_configuration,
    run_10x12_dryrun,
)
from .physics import (
    DEFAULT_ENVIRONMENT,
    ActiveFilter,
    Design,
    Driver,
    DriverMount,
    Environment,
    Port,
    SourceResult,
    evaluate_source,
    net_volume_m3,
    rectangular_transfer,
    validate_design,
)

__version__ = "0.2.0"

__all__ = [
    "ActiveFilter",
    "DEFAULT_ENVIRONMENT",
    "Design",
    "Driver",
    "DriverMount",
    "Environment",
    "EvaluationResult",
    "LayeredBeamSearchOptimizer",
    "MultiSeatMinimaxOptimizer",
    "Port",
    "RLTActiveBilinearSolver",
    "SourceResult",
    "VectorizedGeometricScanner",
    "active_filter_from_record",
    "create_10x12_room_instance",
    "create_control_records",
    "create_driver_catalog",
    "create_enclosure_designs",
    "evaluate_source",
    "evaluate_system_configuration",
    "net_volume_m3",
    "precompute_atom_tensors",
    "prune_nondominated",
    "rectangular_transfer",
    "run_10x12_dryrun",
    "search_instance",
    "select_beam",
    "validate_design",
]

