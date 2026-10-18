"""Lake scheme selection by config type.

``LakeConfig`` selects the two-layer lake (``"two_layer"``), the default of every
coupled driver; ``AutoSimColumnLakeConfig`` selects the AutoSim multi-level
column (``"autosim_column"``).  The selection is made on the static Python
config before tracing, and an unknown config type raises.
"""

from __future__ import annotations

from legoesm.core.coupling_fields import AtmToSurface
from legoesm.coupler.lake.autosim_column import step_autosim_column_lake
from legoesm.coupler.lake.config import AutoSimColumnLakeConfig, LakeConfig
from legoesm.coupler.lake.two_layer_lake import step_lake

VALID_LAKE_SCHEMES = ("two_layer", "autosim_column")


def lake_scheme_of(config) -> str:
    """Scheme name selected by ``config``; raises ``ValueError`` for an unknown type."""
    if isinstance(config, AutoSimColumnLakeConfig):
        return "autosim_column"
    if isinstance(config, LakeConfig):
        return "two_layer"
    raise ValueError(
        f"Unknown lake scheme config {type(config).__name__!r}; expected a config of "
        f"one of {VALID_LAKE_SCHEMES}.")


def step_lake_tile(state, forcing: AtmToSurface, config, U_min: float, dt: float):
    """Step the lake tile with the scheme selected by ``config``.

    ``LakeConfig`` calls ``step_lake`` unchanged (state ``LakeState``);
    ``AutoSimColumnLakeConfig`` calls ``step_autosim_column_lake`` (state
    ``ColumnLakeState``).  Returns ``(new_state, TileResponse)``.
    """
    scheme = lake_scheme_of(config)
    if scheme == "autosim_column":
        return step_autosim_column_lake(state, forcing, config, U_min, dt)
    return step_lake(state, forcing, config, U_min, dt)
