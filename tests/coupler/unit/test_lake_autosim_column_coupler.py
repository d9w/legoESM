"""The ``autosim_column`` lake through the coupler and the run_coupled --params bundle.

Covers the selection wiring: ``init_surface_state(lake_config=...)``, ``make_coupler``'s
lake tile (``step_lake_tile``) and ``run_coupled --lake-scheme`` with ``--params``.
Runs in float64 (``jax_enable_x64`` set below).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface
from legoesm.coupler.config import CouplerConfig, TileConfig
from legoesm.coupler.coupler import init_surface_state, make_coupler
from legoesm.coupler.lake import AutoSimColumnLakeConfig, ColumnLakeState, LakeConfig, LakeState
from legoesm.ice.config import SeaIceConfig
from legoesm.land.config import LandConfig

jax.config.update("jax_enable_x64", True)

REPO = Path(__file__).resolve().parents[3]
SHAPE = (6, 2, 2)
DT = 3600.0


def _forcing(shape=SHAPE):
    full = lambda v: jnp.full(shape, v)  # noqa: E731
    return AtmToSurface(
        sw_down=full(200.0), lw_down=full(300.0), precip_total=full(1e-5),
        precip_snow=jnp.zeros(shape), T_lowest=full(285.0), q_lowest=full(5e-3),
        u_lowest=full(5.0), v_lowest=full(-3.0), p_lowest=full(95000.0),
        p_surface=full(1e5), rho_lowest=full(1.15), cos_zenith=full(0.6),
        co2_ppmv=jnp.array(400.0), has_radiation=jnp.array(1.0),
        has_precipitation=jnp.array(1.0))


def test_init_surface_state_builds_the_column():
    cfg = AutoSimColumnLakeConfig()
    state = init_surface_state(SHAPE, lake_config=cfg, T_epi_init=285.0, T_hypo_init=278.0)
    assert isinstance(state.lake, ColumnLakeState)
    T = np.asarray(state.lake.T_lake.data)
    assert T.shape == SHAPE + (len(cfg.depths_m),)
    shallow = np.asarray(cfg.depths_m) <= LakeConfig().h_epi
    assert np.all(T[..., shallow] == 285.0) and np.all(T[..., ~shallow] == 278.0)
    np.testing.assert_array_equal(np.asarray(state.lake.T_sediment.data), T)
    assert state.lake.wind_power_memory.data.shape == SHAPE


def test_default_lake_state_unchanged():
    a = init_surface_state(SHAPE)
    b = init_surface_state(SHAPE, lake_config=LakeConfig())
    assert isinstance(a.lake, LakeState) and isinstance(b.lake, LakeState)
    for x, y in zip(jax.tree.leaves(a.lake), jax.tree.leaves(b.lake)):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_coupler_steps_the_column_lake():
    cfg = AutoSimColumnLakeConfig()
    step_fn = make_coupler(CouplerConfig(), LandConfig(), SeaIceConfig(), cfg)
    state = init_surface_state(SHAPE, lake_config=cfg)
    tile = TileConfig(f_land=jnp.full(SHAPE, 0.3), f_lake=jnp.full(SHAPE, 0.05))
    sst, zero = jnp.full(SHAPE, 290.0), jnp.zeros(SHAPE)
    T0 = state.lake.T_lake.data
    for _ in range(6):
        state, blended = step_fn(state, _forcing(), tile, sst, zero, zero, DT)
    assert isinstance(state.lake, ColumnLakeState)
    assert jnp.all(jnp.isfinite(state.lake.T_lake.data))
    assert float(jnp.max(jnp.abs(state.lake.T_lake.data - T0))) > 0.0
    assert jnp.all(state.lake.T_lake.data >= constants.T_freeze)
    for name, val in blended._asdict().items():
        if val is not None:
            assert jnp.all(jnp.isfinite(val)), name


# --- run_coupled --lake-scheme and --params ---------------------------------------------

_spec = importlib.util.spec_from_file_location("_run_coupled_lake", REPO / "scripts/run/run_coupled.py")
run_coupled = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_coupled)


def _coupled_inputs():
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.driver.coupled_config import CoupledConfig
    from legoesm.land.config import MultiLayerLandConfig
    atm = ExperimentConfig(
        grid=GridConfig(grid_type="cubed_sphere", resolution=4, nlev=5),
        dycore=DycoreConfig(model_type="hydrostatic", discretization="cdgrid"))
    return atm, CoupledConfig(land_config=MultiLayerLandConfig())


def test_lake_scheme_flag():
    dests = {a.dest: a for a in run_coupled.build_parser()._actions}
    assert dests["lake_scheme"].default == "two_layer"
    assert tuple(dests["lake_scheme"].choices) == ("two_layer", "autosim_column")


def test_lake_scheme_flag_reaches_the_driver(tmp_path):
    common = ["--minimal-physics", "--resolution", "8", "--nlev", "10", "--output",
              str(tmp_path)]
    column = run_coupled.main(["--lake-scheme", "autosim_column", *common], build_only=True)
    assert column._lake_config == AutoSimColumnLakeConfig()
    default = run_coupled.main(common, build_only=True)
    assert default._lake_config is None


def test_params_reach_the_selected_column_config(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("coupler.lake_autosim_column.q_correction_W_m2: 20.0\n")
    atm, coupled = _coupled_inputs()
    *_, lake = run_coupled.apply_coupled_params(str(p), "analytical", atm, coupled, None,
                                                lake_scheme="autosim_column")
    assert isinstance(lake, AutoSimColumnLakeConfig) and lake.q_correction_W_m2 == 20.0


def test_params_for_the_unselected_scheme_raise(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_text("coupler.lake_autosim_column.q_correction_W_m2: 20.0\n")
    atm, coupled = _coupled_inputs()
    with pytest.raises(ValueError, match="no effect"):
        run_coupled.apply_coupled_params(str(p), "analytical", atm, coupled, None,
                                         lake_scheme="two_layer")
    with pytest.raises(ValueError, match="Unknown lake scheme"):
        run_coupled.select_lake_config(run_coupled.build_params_bundle(coupled), "flake")
