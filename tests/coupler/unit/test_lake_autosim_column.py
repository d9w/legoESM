"""AutoSim multi-level lake column (``coupler/lake/autosim_column.py``) and the lake dispatch.

Runs in float64 (``jax_enable_x64`` set below): the energy-budget and gradient checks compare
at 1e-10 and finite-difference tolerances that float32 cannot resolve.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
from legoesm.coupler.lake import (
    VALID_LAKE_SCHEMES,
    AutoSimColumnLakeConfig,
    ColumnLakeState,
    LakeConfig,
    LakeState,
    init_autosim_column_lake_state,
    lake_scheme_of,
    step_autosim_column_lake,
    step_lake,
    step_lake_tile,
)
from legoesm.coupler.lake.autosim_column import (
    column_background_diffusivity,
    column_geometry,
    column_lake_step_C,
    column_sediment_offset_C,
    column_wind_memory_init,
    column_wind_power,
    freshwater_density_chen_millero,
)

jax.config.update("jax_enable_x64", True)

CONFIG = AutoSimColumnLakeConfig()
N_LEV = len(CONFIG.depths_m)
SHAPE = (3,)
DT = 10800.0
U_MIN = 1.0


def make_forcing(shape=SHAPE, sw_down=200.0, lw_down=320.0, T_lowest=286.0, u=4.0, v=1.0,
                 q=0.007, p=101000.0, rho=1.22, precip=1e-5):
    f = jnp.float64
    full = lambda x: jnp.full(shape, x, f)  # noqa: E731
    return AtmToSurface(
        sw_down=full(sw_down), lw_down=full(lw_down), precip_total=full(precip),
        precip_snow=full(0.0), T_lowest=full(T_lowest), q_lowest=full(q), u_lowest=full(u),
        v_lowest=full(v), p_lowest=full(p), p_surface=full(p), rho_lowest=full(rho),
        cos_zenith=full(0.5), co2_ppmv=full(415.0), has_radiation=full(1.0),
        has_precipitation=full(1.0))


def stratified_profile_K(shape=SHAPE):
    z = jnp.asarray(CONFIG.depths_m)
    prof = constants.T_freeze + 6.0 + 10.0 / (1.0 + jnp.exp((z - 10.0) / 2.0))
    return jnp.broadcast_to(prof, tuple(shape) + (N_LEV,))


def make_state(T_K=None, shape=SHAPE, config=CONFIG):
    T_K = stratified_profile_K(shape) if T_K is None else T_K
    return init_autosim_column_lake_state(T_K, shape, config, dims=("ncol",))


def state_C(T_K=None, shape=SHAPE, S_offset_C=0.0, E=None):
    """``(T, S, E)`` in degC and W/m2 for ``column_lake_step_C``."""
    T_C = (stratified_profile_K(shape) if T_K is None else T_K) - constants.T_freeze
    E = jnp.full(shape, CONFIG.wind_power_ref_W_m2 if E is None else E)
    return T_C, T_C + S_offset_C, E


# 16 steps of 10800 s of the numpy program legoesm_feeagh_obs-s016 (its step with the recorded
# fit, legoESM g and saturation humidity put in) from the profile of `stratified_profile_K`, the
# sediment 1.5 K colder and E = 0.6 W/m2: water, sediment [degC] and memory [W/m2].
PROGRAM_REFERENCE = {
    "warm": (
        (
            15.293241197743868, 15.293261159496636, 15.28712104435123, 13.546344093809003,
            9.997011243227949, 7.568674686845348, 6.760593899844895, 6.3206873217096975,
            6.170952976206275, 6.111046888358994, 6.03257450492788, 6.003963077852579,
            5.99233171702269,
        ),
        (
            14.397235101433093, 14.272241985234938, 13.744233766302553, 11.813343192699815,
            8.278159941155021, 5.694940255222175, 4.977084476581793, 4.682536923184857,
            4.569584812618748, 4.527362928015924, 4.504615946449036, 4.502726359634716,
            4.502551865416689,
        ),
        0.5753626197972802,
    ),
    "cooling": (
        (
            13.83759033174709, 13.837625739655174, 13.83764389747539, 13.02153158675344,
            9.944363559028337, 7.561618586946134, 6.757711776088198, 6.3182545806227495,
            6.170628887468426, 6.110906111688449, 6.0325506729711655, 6.003960069985898,
            5.992331499289948,
        ),
        (
            14.395874069941422, 14.270881034655025, 13.742891731395119, 11.812906892881207,
            8.278123621079988, 5.694935849316313, 4.977082602921689, 4.682535505682456,
            4.569584625657894, 4.527362858603695, 4.504615936581564, 4.502726358557529,
            4.502551865347484,
        ),
        0.5600481248658831,
    ),
}


# The s004 configuration on this scheme: every term added after s004 at its neutral value.
S004 = CONFIG._replace(
    q_correction_W_m2=43.75, sw_absorption_scale=0.825, sw_extinction_m=1.225,
    hostetler_wind_scale=3.0, background_diffusivity_scale=30.0, background_n2_exponent=0.43,
    weak_stratification_n2_s2=0.0, shelter_inverse_time_per_day=0.0,
    lake_number_gate_amplitude=0.0, penetrative_convection_coef=0.0,
    sediment_conductance_W_m2_K=0.0, wind_memory_exponent=0.0)


class TestGeometry:
    def test_volumes_match_the_piecewise_linear_hypsograph(self):
        geo = column_geometry(CONFIG, dtype=jnp.float64)
        d = np.asarray(CONFIG.depths_m)
        zp = np.concatenate([[0.0], d, [CONFIG.lake_depth_m]])
        ap = np.concatenate([[1.0], CONFIG.area_fraction, [0.0]])
        zb = np.asarray(geo["zb"])
        exact = []
        for lo, hi in zip(zb[:-1], zb[1:]):
            z = np.unique(np.concatenate([[lo, hi], zp[(zp > lo) & (zp < hi)]]))
            a = np.interp(z, zp, ap)
            exact.append(np.sum(0.5 * (a[1:] + a[:-1]) * np.diff(z)))
        np.testing.assert_allclose(np.asarray(geo["vol"]), exact, rtol=1e-4)
        assert float(geo["a_b"][0]) == 1.0 and float(geo["a_b"][-1]) == 0.0

    def test_density_maximum_near_4C(self):
        T = jnp.linspace(0.0, 10.0, 10001)
        rho = freshwater_density_chen_millero(T)
        assert abs(float(T[jnp.argmax(rho)]) - 3.9863) < 2e-3


class TestStep:
    def test_returns_finite_state_and_response(self):
        new, resp = step_autosim_column_lake(make_state(), make_forcing(), CONFIG, U_MIN, DT)
        assert isinstance(new, ColumnLakeState) and isinstance(resp, TileResponse)
        assert new.T_lake.data.shape == SHAPE + (N_LEV,)
        assert new.T_lake.units == "K" and new.T_lake.dims == ("ncol", "lake_level")
        assert new.T_sediment.data.shape == SHAPE + (N_LEV,)
        assert new.T_sediment.units == "K" and new.T_sediment.dims == ("ncol", "lake_level")
        assert new.wind_power_memory.data.shape == SHAPE
        assert new.wind_power_memory.units == "W/m2"
        for name, val in resp._asdict().items():
            if val is not None:          # optional TileResponse channels the lake leaves unset
                assert jnp.all(jnp.isfinite(val)), name
        np.testing.assert_array_equal(np.asarray(resp.T_sfc), np.asarray(new.T_lake.data[..., 0]))
        # The reported albedo carries the absorbed-shortwave scale (energy seen by the air).
        np.testing.assert_allclose(
            np.asarray(resp.albedo),
            1.0 - CONFIG.sw_absorption_scale * (1.0 - CONFIG.albedo_lake), rtol=1e-15)
        assert CONFIG.sw_absorption_scale < 1.0 and float(resp.albedo[0]) > CONFIG.albedo_lake

    def test_energy_budget_closes(self):
        """Water plus sediment heat change = dt (SW_net + LW_net - SH - LH + q_correction)."""
        geo = column_geometry(CONFIG, dtype=jnp.float64)
        st = state_C(S_offset_C=-1.5)
        forcing = make_forcing()
        (T_new, S_new, _), fl = column_lake_step_C(st, forcing, CONFIG, geo, U_MIN, DT)
        sw_net, lw_net, _, shflx, lhflx = fl[:5]
        rho_c = CONFIG.rho_water * CONFIG.c_water_mass
        c_sed = (CONFIG.sediment_conductance_W_m2_K * CONFIG.sediment_relaxation_days * 86400.0
                 * geo["bed"])
        stored = (rho_c * jnp.sum(geo["vol"] * (T_new - st[0]), axis=-1)
                  + jnp.sum(c_sed * (S_new - st[1]), axis=-1))
        supplied = DT * (sw_net + lw_net - shflx - lhflx + CONFIG.q_correction_W_m2)
        np.testing.assert_allclose(np.asarray(stored), np.asarray(supplied), rtol=1e-10)
        # The sediment took heat from the water (it started 1.5 K colder).
        assert float(jnp.min(S_new - st[1])) > 0.0

    def test_convective_mixing_removes_inversion(self):
        cold_top = jnp.broadcast_to(
            constants.T_freeze + jnp.linspace(10.0, 14.0, N_LEV), SHAPE + (N_LEV,))
        new, _ = step_autosim_column_lake(make_state(cold_top), make_forcing(sw_down=0.0),
                                          CONFIG, U_MIN, DT)
        spread_before = float(jnp.max(cold_top) - jnp.min(cold_top))
        spread_after = float(jnp.max(new.T_lake.data) - jnp.min(new.T_lake.data))
        assert spread_after < 0.5 * spread_before

    def test_stronger_wind_mixes_heat_deeper(self):
        geo = column_geometry(CONFIG, dtype=jnp.float64)
        T_C = stratified_profile_K() - constants.T_freeze
        st = state_C()
        (calm, _, _), _ = column_lake_step_C(st, make_forcing(u=1.0, v=0.0), CONFIG, geo, U_MIN,
                                             DT)
        (windy, _, _), _ = column_lake_step_C(st, make_forcing(u=12.0, v=0.0), CONFIG, geo,
                                              U_MIN, DT)
        # level 3 (8 m) sits below the warm layer: wind mixing warms it more.
        assert float(windy[0, 3] - T_C[0, 3]) > float(calm[0, 3] - T_C[0, 3])

    def test_water_bounded_below_by_freezing(self):
        state = make_state(jnp.full(SHAPE + (N_LEV,), constants.T_freeze + 0.05))
        forcing = make_forcing(sw_down=0.0, lw_down=150.0, T_lowest=245.0, u=15.0, q=1e-4)
        for _ in range(20):
            state, resp = step_autosim_column_lake(state, forcing, CONFIG, U_MIN, DT)
        assert float(jnp.min(state.T_lake.data)) >= constants.T_freeze

    def test_jit_parity(self):
        state, forcing = make_state(), make_forcing()
        eager = step_autosim_column_lake(state, forcing, CONFIG, U_MIN, DT)
        jitted = jax.jit(lambda s, f: step_autosim_column_lake(s, f, CONFIG, U_MIN, DT))(
            state, forcing)
        for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(jitted)):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=1e-9)

    def test_check_grads_order_two(self):
        forcing = make_forcing(shape=(1,))
        T0 = stratified_profile_K(shape=(1,))

        def one_step(T_K, q_corr, kd, gamma_w, h_sed):
            cfg = CONFIG._replace(q_correction_W_m2=q_corr, sw_extinction_m=kd,
                                  wind_memory_exponent=gamma_w, sediment_conductance_W_m2_K=h_sed)
            state = init_autosim_column_lake_state(T_K, (1,), cfg, dims=("ncol",),
                                                   T_sediment_K=T_K - 1.0,
                                                   wind_power_memory=0.6)
            new, _ = step_autosim_column_lake(state, forcing, cfg, U_MIN, DT)
            return new.T_lake.data, new.T_sediment.data

        # Reverse mode only: thomas_solve is a custom_vjp.  The finite-difference
        # truncation error of the T derivative scales as eps^2 (measured relative
        # mismatch 0.15, 1.6e-3, 1.7e-5 at eps 1e-3, 1e-4, 1e-5), so eps is 1e-6.
        check_grads(one_step, (T0, CONFIG.q_correction_W_m2, CONFIG.sw_extinction_m,
                               CONFIG.wind_memory_exponent, CONFIG.sediment_conductance_W_m2_K),
                    order=2, modes=["rev"], atol=1e-6, rtol=1e-5, eps=1e-6)


class TestS016Terms:
    def test_wind_memory_relaxes_exactly(self):
        geo = column_geometry(CONFIG, dtype=jnp.float64)
        forcing = make_forcing()
        st = state_C(E=2.0)
        (_, _, E1), _ = column_lake_step_C(st, forcing, CONFIG, geo, U_MIN, DT)
        p_w = column_wind_power(forcing, CONFIG, U_MIN)
        tau = CONFIG.wind_memory_days * 86400.0
        np.testing.assert_allclose(np.asarray(E1), np.asarray(p_w + (2.0 - p_w)
                                                             * np.exp(-DT / tau)), rtol=1e-14)
        # Two half steps equal one full step (exact in time).
        half, _ = column_lake_step_C(st, forcing, CONFIG, geo, U_MIN, DT / 2)
        (_, _, E2), _ = column_lake_step_C(half, forcing, CONFIG, geo, U_MIN, DT / 2)
        np.testing.assert_allclose(np.asarray(E2), np.asarray(E1), rtol=1e-13)

    def test_memory_exponent_zero_removes_the_memory(self):
        geo = column_geometry(CONFIG, dtype=jnp.float64)
        cfg = CONFIG._replace(wind_memory_exponent=0.0)
        forcing = make_forcing()
        (a, _, _), _ = column_lake_step_C(state_C(E=0.05), forcing, cfg, geo, U_MIN, DT)
        (b, _, _), _ = column_lake_step_C(state_C(E=3.0), forcing, cfg, geo, U_MIN, DT)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        (c, _, _), _ = column_lake_step_C(state_C(E=3.0), forcing, CONFIG, geo, U_MIN, DT)
        assert float(jnp.max(jnp.abs(c - a))) > 0.0

    def test_sediment_off_leaves_the_slab(self):
        geo = column_geometry(S004, dtype=jnp.float64)
        st = state_C(S_offset_C=-2.0)
        (T_new, S_new, _), _ = column_lake_step_C(st, make_forcing(), S004, geo, U_MIN, DT)
        np.testing.assert_array_equal(np.asarray(S_new), np.asarray(st[1]))
        assert bool(jnp.all(jnp.isfinite(T_new)))

    def test_penetrative_convection_mixes_under_cooling(self):
        """A weakly stratified column (0.01 K/m) under strong surface cooling: the plumes
        overturn the weak faces below the surface layer, so 0.9 m to 14 m mixes more."""
        geo = column_geometry(CONFIG, dtype=jnp.float64)
        cooling = make_forcing(shape=(1,), sw_down=0.0, lw_down=220.0, T_lowest=268.0,
                               u=2.0, v=0.0, q=0.002)
        z = jnp.asarray(CONFIG.depths_m)
        weak = constants.T_freeze + 10.0 + 0.01 * (CONFIG.lake_depth_m - z)[None, :]
        st = state_C(T_K=weak, shape=(1,))
        (on, _, _), _ = column_lake_step_C(st, cooling, CONFIG, geo, U_MIN, DT)
        (off, _, _), _ = column_lake_step_C(
            st, cooling, CONFIG._replace(penetrative_convection_coef=0.0), geo, U_MIN, DT)
        spread = lambda T: float(T[0, 0] - T[0, 5])  # noqa: E731  (0.9 m minus 14 m)
        assert abs(spread(on)) < 0.5 * abs(spread(off))

    def test_sediment_offset_matches_the_complex_form(self):
        """The real closed form equals Re(W_k (-i x)/(1 + i x) e^{i k w}) of the program."""
        doy = jnp.asarray([1.0, 95.5, 200.0, 365.0])
        got = np.asarray(column_sediment_offset_C(doy, CONFIG, dtype=jnp.float64))
        coef = np.asarray(CONFIG.temperature_climatology_C)
        omega = 2.0 * np.pi / 365.25
        want = np.zeros((doy.size, N_LEV))
        for k in (1, 2):
            w_hat = coef[:, 2 * k - 1] - 1j * coef[:, 2 * k]
            x = k * omega * CONFIG.sediment_relaxation_days
            want += np.real(w_hat * (-1j * x / (1.0 + 1j * x))
                            * np.exp(1j * k * omega * np.asarray(doy)[:, None]))
        np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
        assert np.all(np.asarray(column_sediment_offset_C(doy, S004)) == 0.0)

    def test_wind_memory_init_is_the_climatology(self):
        doy = jnp.asarray([10.0, 180.0])
        got = np.asarray(column_wind_memory_init(doy, CONFIG, dtype=jnp.float64))
        c = np.asarray(CONFIG.wind_power_climatology_W_m2)
        w = 2.0 * np.pi / 365.25 * np.asarray(doy)
        want = c[0] + c[1] * np.cos(w) + c[2] * np.sin(w) + c[3] * np.cos(2 * w) + c[4] * np.sin(
            2 * w)
        np.testing.assert_allclose(got, want, rtol=1e-14)
        empty = CONFIG._replace(wind_power_climatology_W_m2=())
        assert np.all(np.asarray(column_wind_memory_init(doy, empty))
                      == CONFIG.wind_power_ref_W_m2)

    @pytest.mark.parametrize("case", ["warm", "cooling"])
    def test_reference_numbers_of_the_program(self, case):
        forcing = {
            "warm": make_forcing(shape=(1,), sw_down=200.0, lw_down=320.0, T_lowest=286.0,
                                 u=5.0, v=0.0, q=0.007, p=101000.0, rho=1.22),
            "cooling": make_forcing(shape=(1,), sw_down=0.0, lw_down=250.0, T_lowest=275.0,
                                    u=2.0, v=0.0, q=0.003, p=101000.0, rho=1.22),
        }[case]
        geo = column_geometry(CONFIG, dtype=jnp.float64)
        st = state_C(shape=(1,), S_offset_C=-1.5, E=0.6)
        for _ in range(16):
            st, _ = column_lake_step_C(st, forcing, CONFIG, geo, U_MIN, DT)
        T_ref, S_ref, E_ref = PROGRAM_REFERENCE[case]
        np.testing.assert_allclose(np.asarray(st[0][0]), T_ref, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(st[1][0]), S_ref, rtol=0.0, atol=1e-9)
        np.testing.assert_allclose(float(st[2][0]), E_ref, rtol=1e-13)

    def test_cold_start_state(self):
        state = make_state()
        np.testing.assert_array_equal(np.asarray(state.T_sediment.data),
                                      np.asarray(state.T_lake.data))
        assert np.all(np.asarray(state.wind_power_memory.data) == CONFIG.wind_power_ref_W_m2)


class TestWeakFloorClip:
    """The program clips the weak-branch floor to [1e-9, max(threshold, 1e-9)] (make_params)."""

    N2P = jnp.asarray([0.0, 5e-5, 2e-4])

    def _k(self, floor):
        cfg = CONFIG._replace(weak_stratification_n2_s2=1e-4,
                              weak_stratification_n2_floor_s2=floor)
        return np.asarray(column_background_diffusivity(self.N2P, cfg, jnp.asarray(0.5)))

    def test_floor_above_threshold_matches_the_program(self):
        # numpy program, make_params(fitted={..., n2_low=1e-4, n2_low_floor=1e-3}), mem = 0.5
        want = [1.5215495407853994e-05, 1.5215495407853994e-05, 6.39732777238689e-06]
        np.testing.assert_allclose(self._k(1e-3), want, rtol=1e-12)
        np.testing.assert_array_equal(self._k(1e-3), self._k(1e-4))

    def test_zero_floor_gives_a_finite_diffusivity(self):
        k = self._k(0.0)
        assert np.all(np.isfinite(k))
        # numpy program: the floor clips to 1e-9
        np.testing.assert_allclose(k[0], 0.08556301851080285, rtol=1e-12)


class TestFloatingInputs:
    """The initial-state helpers reject non-floating inputs and keep floating dtypes."""

    def test_state_init_rejects_integer_and_bool(self):
        with pytest.raises(TypeError, match="floating dtype"):
            init_autosim_column_lake_state(280, SHAPE, CONFIG, dims=("ncol",))
        with pytest.raises(TypeError, match="floating dtype"):
            init_autosim_column_lake_state(280.0, SHAPE, CONFIG, dims=("ncol",),
                                           wind_power_memory=jnp.asarray(True))
        state = init_autosim_column_lake_state(jnp.float32(280.0), SHAPE, CONFIG, dims=("ncol",))
        assert state.T_lake.data.dtype == jnp.float32
        assert state.wind_power_memory.data.dtype == jnp.float32

    def test_sediment_offset_rejects_integer_day(self):
        with pytest.raises(TypeError, match="floating dtype"):
            column_sediment_offset_C(180, CONFIG)
        got = jax.jit(lambda d: column_sediment_offset_C(d, CONFIG))(jnp.float32(180.0))
        assert got.dtype == jnp.float32 and got.shape == (N_LEV,)

    def test_wind_memory_init_rejects_integer_day(self):
        with pytest.raises(TypeError, match="floating dtype"):
            column_wind_memory_init(jnp.asarray(180), CONFIG)
        got = jax.jit(lambda d: column_wind_memory_init(d, CONFIG))(jnp.float32(180.0))
        assert got.dtype == jnp.float32
        np.testing.assert_allclose(float(got), 0.2257001, rtol=1e-6)


class TestDispatch:
    def test_scheme_names(self):
        assert VALID_LAKE_SCHEMES == ("two_layer", "autosim_column")
        assert lake_scheme_of(LakeConfig()) == "two_layer"
        assert lake_scheme_of(CONFIG) == "autosim_column"

    def test_unknown_config_raises(self):
        with pytest.raises(ValueError, match="Unknown lake scheme"):
            lake_scheme_of(object())
        with pytest.raises(ValueError, match="Unknown lake scheme"):
            step_lake_tile(None, make_forcing(), {"scheme": "two_layer"}, U_MIN, DT)

    def test_two_layer_path_is_step_lake(self):
        field = lambda v, n, u: Field(jnp.full(SHAPE, v), n, ("ncol",), u)  # noqa: E731
        state = LakeState(T_epi=field(290.0, "T_epi", "K"), T_hypo=field(280.0, "T_hypo", "K"),
                          Q_freeze=field(0.0, "Q_freeze", "J/m2"))
        forcing = make_forcing()
        a = step_lake_tile(state, forcing, LakeConfig(), U_MIN, 3600.0)
        b = step_lake(state, forcing, LakeConfig(), U_MIN, 3600.0)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_column_path_through_public_config(self):
        state, forcing = make_state(), make_forcing()
        a = step_lake_tile(state, forcing, CONFIG, U_MIN, DT)
        b = step_autosim_column_lake(state, forcing, CONFIG, U_MIN, DT)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
