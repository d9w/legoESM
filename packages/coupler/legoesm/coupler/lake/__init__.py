"""Lake models: the two-layer lake and the AutoSim multi-level column.

The scheme is selected by the config type passed to :func:`step_lake_tile`:
:class:`LakeConfig` runs the two-layer ``step_lake`` (the default everywhere),
:class:`AutoSimColumnLakeConfig` runs ``step_autosim_column_lake``.
"""

from legoesm.coupler.lake.autosim_column import (
    init_autosim_column_lake_state,
    step_autosim_column_lake,
)
from legoesm.coupler.lake.config import AutoSimColumnLakeConfig, LakeConfig
from legoesm.coupler.lake.dispatch import VALID_LAKE_SCHEMES, lake_scheme_of, step_lake_tile
from legoesm.coupler.lake.state import ColumnLakeState, LakeState
from legoesm.coupler.lake.two_layer_lake import step_lake

__all__ = [
    "AutoSimColumnLakeConfig", "ColumnLakeState", "LakeConfig", "LakeState",
    "VALID_LAKE_SCHEMES", "init_autosim_column_lake_state", "lake_scheme_of",
    "step_autosim_column_lake", "step_lake", "step_lake_tile",
]
