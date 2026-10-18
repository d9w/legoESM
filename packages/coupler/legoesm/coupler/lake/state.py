"""Lake model state container."""

from __future__ import annotations

from typing import NamedTuple

from legoesm.core.field import Field


class LakeState(NamedTuple):
    """Two-layer lake state.

    All fields have shape (6, n, n).
    """
    T_epi: Field         # Epilimnion temperature [K]
    T_hypo: Field        # Hypolimnion temperature [K]
    Q_freeze: Field | None = None  # Diagnosed latent freezing flux [W/m²]


class ColumnLakeState(NamedTuple):
    """Multi-level lake column state of the ``autosim_column`` scheme.

    ``T_lake`` and ``T_sediment`` have shape ``(..., n_levels)``: the spatial
    shape of the tile followed by the lake levels, level 0 at the surface.
    ``T_sediment`` is the bed-sediment slab under each level (the bed area a
    level touches).  ``wind_power_memory`` has the spatial shape.  The scheme
    defines no ice; water temperature is bounded below by ``T_freeze`` before
    the sediment exchange, which can take it slightly below.
    """
    T_lake: Field        # Lake water temperature at the level centres [K]
    T_sediment: Field    # Bed-sediment slab temperature under each level [K]
    wind_power_memory: Field  # Relaxing memory E of the wind power rho_a Cd U^3 [W/m2]
