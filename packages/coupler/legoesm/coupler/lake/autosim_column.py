"""Multi-level hypsographic lake column ported from an AutoSim program (``autosim_column``).

Source: AutoSim run 295835 on the ``legoesm_feeagh_obs`` problem (Lough Feeagh
daily temperature profiles at 13 depths), candidate ``legoesm_feeagh_obs-s016``
(numpy program, sha256 ``fa7eaeac...``, validation RMSE 0.612 degC, held-out
test 0.347 degC), against 1.011 and 0.980 for
:func:`~legoesm.coupler.lake.two_layer_lake.step_lake`.  s016 descends from the
pilot port ``s004`` (sha256 ``3fa75a10...``) through s006 to s014; each
increment is a term below that is off at its neutral value, so s004 runs on
this scheme with those terms off (see
:class:`~legoesm.coupler.lake.config.AutoSimColumnLakeConfig`).  The programs
were written as a pure ``step(state, forcing, params, dt)``; this module is
their ``jax.numpy`` translation with the fitted parameters in the config.

Geometry (z positive down from the surface).  Level ``i`` is the control volume
centred on observed depth ``d_i`` between the face depths ``zb_i`` and
``zb_{i+1}`` (mid-points between depths, 0 at the top, ``lake_depth_m`` at the
bottom).  ``A(z)`` is the hypsograph as a fraction of the surface area, linear
between the tabulated depths; ``V_i = int A dz`` over the level, per unit
surface area, with centroid ``zc_i``; ``bed_i = A(zb_i) - A(zb_{i+1})`` is the
bed area the level touches (sums to 1).

State: water temperature ``T_i``, bed-sediment slab temperature ``S_i`` under
each level and the wind-power memory ``E``.  One step of ``dt`` seconds, all
fluxes per unit lake surface area:

0. Wind-power memory: ``P = rho_a Cd_lake U^2 |U|`` with ``U^2 = |u|^2 +
   U_min^2``; ``E <- P + (E - P) exp(-dt / tau_w)``;
   ``M = clip(E / P_ref, 0.1, 10)^gamma_w``.
1. Surface energy balance at the start-of-step level-0 temperature ``T_0``
   (explicit, as in ``step_lake``; ``shflx`` and ``lhflx`` positive upward):
   ``Q_ns = LW_net - shflx - lhflx + q_correction`` enters level 0;
   shortwave ``SW_abs = s_SW (1 - albedo) SW_down`` is absorbed in level ``i``
   with the fraction
   ``A(zb_i) exp(-K_d zb_i) - A(zb_{i+1}) exp(-K_d zb_{i+1})`` (light reaching
   the sloping bed heats the water above it).
   ``T*_i = T_i + dt (SW_i + Q_ns delta_i0) / (rho_w c_w V_i)``.
2. Diffusivity on the interior faces from ``T*``: ``N^2 = g (rho_{i+1} - rho_i)
   / (rho_w dz)`` with the Chen-Millero freshwater density.
   Wind term: Hostetler ``kappa w* z exp(-k* z) / (1 + 37 Ri^2)`` with
   ``w* = 0.0012 u_2`` and ``k* = c_k sqrt(|sin lat|) u_2^-1.84``, times the
   sheltering ``exp(-PE_f / (rho_w u*^3 T_e))`` (``PE_f`` the potential energy
   needed to homogenise the volumes above and below face ``f``, ``rho_w u*^2 =
   rho_a Cd_lake U^2``) and the Lake Number gate ``1 + beta exp(-L_N / L_c)``
   (``L_N = S_t (1 - z_T/H) / (rho_w u*^2 sqrt(A_s) (1 - z_g/H))``, ``S_t`` the
   Schmidt stability, ``z_T`` the face of largest ``N^2``, ``z_g`` the centre
   of volume).  Background term: ``K_s = s_E 1.04e-8 N_f^-0.86 (max(N^2,
   N_f^2) / N_f^2)^-q``, times ``M`` on faces with ``N^2 >= N_t^2``; below
   ``N_t^2`` the larger of ``M K_s`` and ``K_s(N_t^2) (max(N^2, N_l^2) /
   N_t^2)^-q_l``.  ``K = K_mol + s_K K_wind + K_bg``; ``K = K_conv`` where
   ``N^2 < 0``, and on faces of the convective layer under net surface cooling
   (connected to the surface, each with ``N^2 z^2 < c_pen w_c^2``, ``w_c =
   (B_0 z)^(1/3)``, ``B_0 = g alpha (-Q) / (rho_w c_w)``).
3. Backward-Euler diffusion between the volumes (tridiagonal; conserves
   ``sum_i V_i T_i``).
4. ``T >= T_freeze``: the program defines no ice; this bound is the only
   cold-side treatment and is not energy conserving.  It applies before step 5,
   so exchange with colder sediment can leave water slightly below ``T_freeze``
   (as in the program).
5. Sediment exchange, exact in time: each level and its slab (capacity
   ``h tau_sed bed_i``, conductance ``h bed_i``) relax toward their common
   mean temperature; ``rho_w c_w V_i T_i + h tau_sed bed_i S_i`` is conserved.

``q_correction`` is a fitted constant heat source (unresolved terms) that is not
part of the fluxes reported to the atmosphere, so a coupled run with a nonzero
value does not close the surface energy budget.  The shortwave scale ``s_SW`` is
reported to the atmosphere as the albedo ``1 - s_SW (1 - albedo_lake)``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.core.bulk_flux import simple_bulk_fluxes
from legoesm.core.coupling_fields import AtmToSurface, TileResponse
from legoesm.core.field import Field
from legoesm.core.surface_energy import surface_radiation_fluxes
from legoesm.coupler.lake.config import AutoSimColumnLakeConfig
from legoesm.coupler.lake.state import ColumnLakeState
from legoesm.thermo import latent_heat_vaporization, saturation_specific_humidity
from legoesm.timestepping.tridiagonal import thomas_solve

__physics_contract__ = {
    "summary": "hypsographic multi-level lake column: bulk surface balance, sheltered and "
               "Lake-Number-gated Hostetler wind diffusivity, wind-memory-scaled background "
               "diffusivity, convection, backward-Euler diffusion, bed-sediment slab",
    "inputs": {"T_lake": "K", "T_sediment": "K", "wind_power_memory": "W/m^2",
               "sw_down": "W/m^2", "lw_down": "W/m^2", "T_lowest": "K",
               "q_lowest": "kg/kg", "u_lowest": "m/s", "v_lowest": "m/s",
               "p_surface": "Pa", "rho_lowest": "kg/m^3", "dt": "s"},
    "outputs": {"T_lake": "K", "T_sediment": "K", "wind_power_memory": "W/m^2",
                "T_sfc": "K", "shflx": "W/m^2", "lhflx": "W/m^2"},
    "sign_convention": "z positive down; shflx/lhflx positive upward (lake to air); SW_net, "
                       "LW_net and q_correction positive into the lake",
    # Diffusion and sediment steps. The T_freeze bound adds heat, and q_correction_W_m2 is a
    # source the atmosphere does not see, so the coupled surface budget does not close.
    "conserves": ["energy"],
    "differentiable": True,
    "reference": "AutoSim run 295835 legoesm_feeagh_obs-s016; Hostetler and Bartlein (1990); "
                 "Fang and Stefan (1996); Chen and Millero (1986); Imberger and Patterson "
                 "(1990)",
    "idealized_test": "no forcing exchange and zero correction: sum V_i T_i plus the sediment "
                      "heat unchanged by the diffusion and sediment steps; isothermal column "
                      "at rest stays isothermal",
}

# --- Chen and Millero (1986) freshwater density, T in degC ---
_CHEN_MILLERO_RHO_REF_KG_M3 = 1000.0
_CHEN_MILLERO_A_C = 288.9414
_CHEN_MILLERO_B = 508929.2
_CHEN_MILLERO_C_C = 68.12963
_CHEN_MILLERO_T_MAX_C = 3.9863

# --- Hostetler and Bartlein (1990) wind-driven eddy diffusivity ---
_HOSTETLER_USTAR_PER_U2 = 0.0012    # surface friction velocity / 2 m wind
_HOSTETLER_KSTAR_EXPONENT = -1.84   # k* ~ u_2^-1.84
_HOSTETLER_RI_COEF = 40.0           # Ri = (sqrt(1 + 40 N^2 kappa^2 z^2 / (w* e^-k*z)^2) - 1) / 20
_HOSTETLER_RI_DIVISOR = 20.0
_HOSTETLER_RI_DAMPING = 37.0        # K_wind / (1 + 37 Ri^2)

# --- Fang and Stefan (1996) background diffusivity K_bg = a (N^2)^b ---
# The background law anchors K_s at the N^2 floor with this value and exponent.
_FANG_STEFAN_COEF = 1.04e-8
_FANG_STEFAN_EXPONENT = -0.43

# --- wind-power memory factor M = clip(E / P_ref, min, max)^gamma (AutoSim s016) ---
_WIND_MEMORY_RATIO_MIN = 0.1
_WIND_MEMORY_RATIO_MAX = 10.0

# --- penetrative convection: thermal expansion from a centred density difference ---
_EXPANSION_HALF_STEP_C = 0.05

# --- annual harmonics of the initial-state climatologies (phase 2 pi doy / year) ---
_DAYS_PER_YEAR = 365.25
_SECONDS_PER_DAY = 86400.0

# --- numerics ---
_VOLUME_QUADRATURE_POINTS = 401     # trapezoid points per level for V_i = int A dz
_SHEAR2_FLOOR = 1e-30               # (w* e^-k*z)^2 floor in the Richardson number
_VOLUME_FLOOR_M = 1e-12             # level volume floor in the centroid
_EXP_ARG_CAP = 700.0                # exp(-min(x, cap)) overflow guard
_WIND_WORK_FLOOR_J_M2 = 1e-12       # floor of the wind work in the sheltering
_STRESS_FLOOR_PA = 1e-9             # floor of the stress in the Lake Number
_CENTRE_DEPTH_RATIO_FLOOR = 1e-3    # floor of 1 - z_g / H in the Lake Number


def column_geometry(config: AutoSimColumnLakeConfig, dtype=None) -> dict:
    """Control volumes of the hypsographic column, per unit lake surface area.

    Returns ``zb`` (face depths, n+1), ``a_b`` (area fraction at the faces),
    ``vol`` (level volumes, n, m), ``zc`` (level centroids, m), ``bed`` (bed
    area each level touches), ``z_int``, ``dz_int`` and ``a_int`` (depth,
    centre spacing and area fraction of the n-1 interior faces), ``z_g`` (centre
    of volume) and ``lake_depth``.
    """
    d = jnp.asarray(config.depths_m, dtype=dtype)
    frac = jnp.asarray(config.area_fraction, dtype=d.dtype)
    bottom = jnp.asarray(config.lake_depth_m, dtype=d.dtype)
    zp = jnp.concatenate([jnp.zeros((1,), d.dtype), d, bottom[None]])
    ap = jnp.concatenate([jnp.ones((1,), d.dtype), frac, jnp.zeros((1,), d.dtype)])
    zb = jnp.concatenate([jnp.zeros((1,), d.dtype), 0.5 * (d[1:] + d[:-1]), bottom[None]])
    zz = jnp.linspace(zb[:-1], zb[1:], _VOLUME_QUADRATURE_POINTS, axis=-1)
    aa = jnp.interp(zz, zp, ap)
    w = 0.5 * (aa[..., 1:] + aa[..., :-1]) * jnp.diff(zz, axis=-1)
    vol = jnp.sum(w, axis=-1)
    zc = jnp.sum(w * (0.5 * (zz[..., 1:] + zz[..., :-1])), axis=-1) / jnp.maximum(
        vol, _VOLUME_FLOOR_M)
    a_b = jnp.interp(zb, zp, ap)
    return {"zb": zb, "a_b": a_b, "vol": vol, "zc": zc, "z_int": zb[1:-1],
            "dz_int": jnp.diff(d), "a_int": jnp.interp(zb[1:-1], zp, ap),
            "bed": jnp.maximum(a_b[:-1] - a_b[1:], 0.0),
            "z_g": jnp.sum(vol * zc) / jnp.sum(vol), "lake_depth": bottom}


def freshwater_density_chen_millero(T_C: jnp.ndarray) -> jnp.ndarray:
    """Freshwater density [kg/m3] at ``T_C`` [degC] (Chen and Millero 1986)."""
    return _CHEN_MILLERO_RHO_REF_KG_M3 * (
        1.0 - (T_C + _CHEN_MILLERO_A_C) / (_CHEN_MILLERO_B * (T_C + _CHEN_MILLERO_C_C))
        * (T_C - _CHEN_MILLERO_T_MAX_C) ** 2)


def freshwater_density_anomaly_chen_millero(T_C: jnp.ndarray) -> jnp.ndarray:
    """Freshwater density minus 1000 kg/m3 at ``T_C`` [degC] (same form, no cancellation)."""
    return -_CHEN_MILLERO_RHO_REF_KG_M3 * (T_C + _CHEN_MILLERO_A_C) / (
        _CHEN_MILLERO_B * (T_C + _CHEN_MILLERO_C_C)) * (T_C - _CHEN_MILLERO_T_MAX_C) ** 2


def _wind_magnitude_sq(forcing: AtmToSurface) -> jnp.ndarray:
    return forcing.u_lowest * forcing.u_lowest + forcing.v_lowest * forcing.v_lowest


def column_wind_stress(forcing: AtmToSurface, config: AutoSimColumnLakeConfig,
                       U_min: float) -> jnp.ndarray:
    """Surface stress ``rho_w u*^2 = rho_a Cd_lake (|u|^2 + U_min^2)`` [Pa]."""
    rho_air = jnp.maximum(forcing.rho_lowest, 0.0)
    return rho_air * config.Cd_lake * (_wind_magnitude_sq(forcing) + U_min ** 2)


def column_wind_power(forcing: AtmToSurface, config: AutoSimColumnLakeConfig,
                      U_min: float) -> jnp.ndarray:
    """Wind energy flux ``rho_a Cd_lake U^3`` [W/m2], ``U^2 = |u|^2 + U_min^2``."""
    return column_wind_stress(forcing, config, U_min) * jnp.sqrt(
        _wind_magnitude_sq(forcing) + U_min ** 2)


def column_mixing_cost(T_C: jnp.ndarray, geometry: dict) -> jnp.ndarray:
    """Potential energy [J/m2 of lake] to homogenise the top k volumes, k = 1..n.

    ``g sum_{i<k} rho_i V_i (zc_i - zbar_k)`` on the last axis; the last entry is
    the Schmidt stability of the column.
    """
    vol, zc = geometry["vol"], geometry["zc"]
    m = freshwater_density_anomaly_chen_millero(T_C) * vol
    zbar = jnp.cumsum(vol * zc, axis=-1) / jnp.cumsum(vol, axis=-1)
    return constants.g * (jnp.cumsum(m * zc, axis=-1) - zbar * jnp.cumsum(m, axis=-1))


def _shelter_factor(cost, stress, config: AutoSimColumnLakeConfig):
    """``exp(-PE_f / (rho_w u*^3 T_e))`` on the interior faces; 1 when the term is off."""
    inv_te = config.shelter_inverse_time_per_day
    on = inv_te > 0.0
    ustar3 = (stress / config.rho_water) ** 1.5
    work = config.rho_water * ustar3 * _SECONDS_PER_DAY / jnp.where(on, inv_te, 1.0)
    work = jnp.maximum(work, _WIND_WORK_FLOOR_J_M2)
    factor = jnp.exp(-jnp.minimum(cost[..., 1:] / work[..., None], _EXP_ARG_CAP))
    return jnp.where(on, factor, 1.0)


def _lake_number_gate(cost, n2, stress, config: AutoSimColumnLakeConfig, geometry: dict):
    """Wind-mixing gate ``1 + beta exp(-L_N / L_c)`` of the column (1 when beta is 0)."""
    h = geometry["lake_depth"]
    z_t = geometry["z_int"][jnp.argmax(n2, axis=-1)]
    num = cost[..., -1] * jnp.clip(1.0 - z_t / h, 0.0, 1.0)
    den = (jnp.maximum(stress, _STRESS_FLOOR_PA) * jnp.sqrt(config.lake_area_m2)
           * jnp.maximum(1.0 - geometry["z_g"] / h, _CENTRE_DEPTH_RATIO_FLOOR))
    lake_number = num / den
    return 1.0 + config.lake_number_gate_amplitude * jnp.exp(
        -jnp.minimum(lake_number / config.lake_number_critical, _EXP_ARG_CAP))


def column_background_diffusivity(n2p, config: AutoSimColumnLakeConfig, memory_factor=1.0):
    """Background diffusivity [m2/s] on the faces from ``N^2 >= 0`` and the factor ``M``.

    ``K_s = s_E a N_f^(2b) (max(N^2, N_f^2) / N_f^2)^-q`` (Fang and Stefan ``a``,
    ``b`` at the floor ``N_f^2``) times ``M`` where ``N^2 >= N_t^2``; below the
    threshold the larger of ``M K_s`` and ``K_s(N_t^2) (max(N^2, N_l^2) /
    N_t^2)^-q_l``.  A zero threshold switches the weak branch off.
    """
    floor = config.n2_floor_s2
    q_bg = config.background_n2_exponent
    k_floor = (config.background_diffusivity_scale * _FANG_STEFAN_COEF
               * floor ** _FANG_STEFAN_EXPONENT)
    k_s = k_floor * (jnp.maximum(n2p, floor) / floor) ** (-q_bg)
    k_m = k_s * jnp.asarray(memory_factor)[..., None]
    n_t = config.weak_stratification_n2_s2
    on = n_t > 0.0
    n_t_safe = jnp.where(on, n_t, 1.0)
    k_t = k_floor * (jnp.maximum(n_t, floor) / floor) ** (-q_bg)
    k_low = k_t * (jnp.maximum(n2p, config.weak_stratification_n2_floor_s2) / n_t_safe) ** (
        -config.weak_stratification_n2_exponent)
    weak = jnp.where(n2p >= n_t, k_m, jnp.maximum(k_low, k_m))
    return jnp.where(on, weak, k_m)


def _penetrative_faces(T1, n2, q_total, config: AutoSimColumnLakeConfig, geometry: dict):
    """Faces of the surface-connected convective layer under net cooling (boolean).

    A face is in the layer when ``N^2 < 0`` or ``N^2 z^2 < c_pen w_c^2`` with
    ``w_c^2 = (B_0 z)^(2/3)``, ``B_0 = g alpha (-Q) / (rho_w c_w)``, and every
    face above it is too.  No face is selected during net heating or below the
    temperature of maximum density.  The selection is a switch and carries no
    gradient.
    """
    T1, n2, q_total = jax.lax.stop_gradient((T1, n2, q_total))
    z_int = geometry["z_int"]
    ts = T1[..., 0]
    alpha = (freshwater_density_chen_millero(ts - _EXPANSION_HALF_STEP_C)
             - freshwater_density_chen_millero(ts + _EXPANSION_HALF_STEP_C)) / (
        (2.0 * _EXPANSION_HALF_STEP_C) * config.rho_water)
    b0 = (constants.g * jnp.maximum(alpha, 0.0) * jnp.maximum(-q_total, 0.0)
          / (config.rho_water * config.c_water_mass))
    wc2 = jnp.cbrt(b0[..., None] * z_int) ** 2
    weak = (n2 < 0.0) | (jnp.maximum(n2, 0.0) * z_int * z_int
                         < config.penetrative_convection_coef * wc2)
    connected = jnp.cumprod(weak.astype(jnp.int32), axis=-1) > 0
    return connected & (b0[..., None] > 0.0)


def _sediment_exchange(T_w, T_s, config: AutoSimColumnLakeConfig, geometry: dict, dt):
    """Exact pairwise heat exchange of each level with its bed slab over ``dt`` s.

    Water capacity ``rho_w c_w V_i``, slab capacity ``h tau bed_i``, conductance
    ``h bed_i``; the sum of the two heat contents is conserved.  Unchanged when
    the conductance is 0.
    """
    rho_c = config.rho_water * config.c_water_mass
    vol, bed = geometry["vol"], geometry["bed"]
    h = config.sediment_conductance_W_m2_K
    tau_s = config.sediment_relaxation_days * _SECONDS_PER_DAY
    c_w = rho_c * vol
    c_s = h * tau_s * bed
    lam = h * bed / c_w + 1.0 / tau_s
    t_m = (c_w * T_w + c_s * T_s) / (c_w + c_s)
    delta = (T_w - T_s) * jnp.exp(-lam * dt)
    w_new = t_m + (c_s / (c_w + c_s)) * delta
    s_new = t_m - (c_w / (c_w + c_s)) * delta
    on = h > 0.0
    return jnp.where(on, w_new, T_w), jnp.where(on, s_new, T_s)


def column_lake_surface_fluxes(T_sfc_K, forcing: AtmToSurface,
                               config: AutoSimColumnLakeConfig, U_min: float):
    """Bulk and radiative surface fluxes at the lake surface temperature ``T_sfc_K``.

    Returns ``(sw_abs, lw_net, lw_up, shflx, lhflx, tau_x, tau_y, L_latent)`` with
    the legoESM constant bulk law (``Ch_lake``, wind with the ``U_min`` floor),
    the Kirchhoff latent heat of vaporization and saturation specific humidity
    over water.  ``sw_abs`` is the open-water net shortwave times
    ``sw_absorption_scale``.  ``shflx``/``lhflx`` are positive upward.
    """
    wind_speed = jnp.sqrt(_wind_magnitude_sq(forcing) + U_min ** 2)
    q_sfc = saturation_specific_humidity(T_sfc_K, forcing.p_surface)
    L_latent = latent_heat_vaporization(T_sfc_K)
    tau_x, tau_y, shflx, lhflx = simple_bulk_fluxes(
        forcing.u_lowest, forcing.v_lowest, forcing.T_lowest, forcing.q_lowest,
        T_sfc_K, q_sfc, forcing.rho_lowest, wind_speed,
        config.Cd_lake, config.Ch_lake, L_latent=L_latent)
    sw_net, lw_net, lw_up = surface_radiation_fluxes(
        jnp.maximum(forcing.sw_down, 0.0), forcing.lw_down, T_sfc_K,
        config.albedo_lake, config.emissivity_lake)
    return (config.sw_absorption_scale * sw_net, lw_net, lw_up, shflx, lhflx, tau_x, tau_y,
            L_latent)


def column_lake_step_C(state_C, forcing: AtmToSurface, config: AutoSimColumnLakeConfig,
                       geometry: dict, U_min: float, dt: float, lat_deg=None):
    """One step of the column in degC.

    ``state_C = (T_C, S_C, E)``: water and sediment temperatures [degC] of shape
    ``(..., n_levels)`` and the wind-power memory [W/m2] of shape ``(...)``.
    Returns ``(state_C_new, fluxes)`` with ``fluxes`` the start-of-step tuple of
    :func:`column_lake_surface_fluxes`.  Pure; ``dt`` in seconds.
    """
    T_C, S_C, E = state_C
    vol, zb, a_b = geometry["vol"], geometry["zb"], geometry["a_b"]
    z_int, dz_int, a_int = geometry["z_int"], geometry["dz_int"], geometry["a_int"]
    rho_c = config.rho_water * config.c_water_mass
    n = vol.shape[-1]

    # 0. Wind-power memory: exact relaxation toward the current wind power.
    p_w = column_wind_power(forcing, config, U_min)
    E_new = p_w + (E - p_w) * jnp.exp(-dt / (config.wind_memory_days * _SECONDS_PER_DAY))
    memory_factor = jnp.clip(E_new / config.wind_power_ref_W_m2, _WIND_MEMORY_RATIO_MIN,
                             _WIND_MEMORY_RATIO_MAX) ** config.wind_memory_exponent

    # 1. Surface energy balance from the start-of-step surface temperature.
    fluxes = column_lake_surface_fluxes(T_C[..., 0] + constants.T_freeze, forcing,
                                        config, U_min)
    sw_net, lw_net, _, shflx, lhflx = fluxes[:5]
    att = a_b * jnp.exp(-config.sw_extinction_m * zb)
    sw_frac = att[:-1] - att[1:]
    top = (jnp.arange(n) == 0).astype(T_C.dtype)
    q_nonsolar = lw_net - shflx - lhflx + config.q_correction_W_m2
    heat = sw_net[..., None] * sw_frac + q_nonsolar[..., None] * top
    T1 = T_C + dt * heat / (rho_c * vol)

    # 2. Face diffusivities from the post-flux profile.
    rho = freshwater_density_chen_millero(T1)
    n2 = constants.g * (rho[..., 1:] - rho[..., :-1]) / (config.rho_water * dz_int)
    n2p = jnp.maximum(n2, 0.0)
    w2 = _wind_magnitude_sq(forcing)
    pos = w2 > 0.0
    wind = jnp.where(pos, jnp.sqrt(jnp.where(pos, w2, 1.0)), 0.0)
    u2 = jnp.maximum(config.wind_2m_ratio * wind, config.wind_2m_min_m_s)
    wstar = (_HOSTETLER_USTAR_PER_U2 * u2)[..., None]
    lat = config.latitude_deg if lat_deg is None else lat_deg
    sin_lat = jnp.sqrt(jnp.abs(jnp.sin(jnp.deg2rad(lat))))
    kstar = (config.hostetler_kstar_coef * sin_lat * u2 ** _HOSTETLER_KSTAR_EXPONENT)[..., None]
    ez = jnp.exp(-kstar * z_int)
    shear2 = jnp.maximum((wstar * ez) ** 2, _SHEAR2_FLOOR)
    kappa = constants.kappa_von_karman
    ri = (jnp.sqrt(1.0 + _HOSTETLER_RI_COEF * n2p * (kappa * z_int) ** 2 / shear2)
          - 1.0) / _HOSTETLER_RI_DIVISOR
    k_wind = kappa * wstar * z_int * ez / (1.0 + _HOSTETLER_RI_DAMPING * ri * ri)
    cost = jnp.maximum(column_mixing_cost(T1, geometry), 0.0)
    stress = column_wind_stress(forcing, config, U_min)
    k_wind = k_wind * _shelter_factor(cost, stress, config)
    k_wind = k_wind * _lake_number_gate(cost, n2, stress, config, geometry)[..., None]
    k_face = (config.k_molecular_m2_s + config.hostetler_wind_scale * k_wind
              + column_background_diffusivity(n2p, config, memory_factor))
    k_face = jnp.where(n2 < 0.0, config.k_convective_m2_s, k_face)
    q_total = sw_net * jnp.sum(sw_frac) + q_nonsolar
    k_face = jnp.where(_penetrative_faces(T1, n2, q_total, config, geometry),
                       config.k_convective_m2_s, k_face)

    # 3. Backward-Euler diffusion between the volumes (conserves sum V_i T_i).
    g = dt * a_int * k_face / dz_int
    zero = jnp.zeros_like(g[..., :1])
    g_up = jnp.concatenate([zero, g], axis=-1)
    g_dn = jnp.concatenate([g, zero], axis=-1)
    T2 = thomas_solve(-g_up / vol, 1.0 + (g_up + g_dn) / vol, -g_dn / vol, T1)

    # 4. No ice in this scheme: water is bounded below by the freezing point (before step 5).
    T_w = jnp.maximum(T2, 0.0)

    # 5. Bed-sediment heat exchange (pairwise, exact in time, energy conserving).
    T_w, S_new = _sediment_exchange(T_w, S_C, config, geometry, dt)
    return (T_w, S_new, E_new), fluxes


def column_sediment_offset_C(day_of_year, config: AutoSimColumnLakeConfig, dtype=None):
    """Initial ``S - T`` [degC] per level at ``day_of_year`` (shape ``(...)`` -> ``(..., n)``).

    The periodic response of the slab ``dS/dt = (T - S) / tau_sed`` to the
    harmonic climatology ``T = mean + sum_k a_k cos(k w) + b_k sin(k w)``:
    ``S - T = sum_k [-(a_k x_k^2 + b_k x_k) cos(k w) + (a_k x_k - b_k x_k^2)
    sin(k w)] / (1 + x_k^2)`` with ``x_k = k omega tau_sed``.  Zero when the
    sediment exchange is off or the climatology is empty.
    """
    doy = jnp.asarray(day_of_year, dtype=dtype)
    n_levels = len(config.depths_m)
    coef = jnp.asarray(config.temperature_climatology_C, dtype=doy.dtype)
    off = jnp.zeros(doy.shape + (n_levels,), doy.dtype)
    if coef.size == 0:
        return off
    omega = 2.0 * math.pi / _DAYS_PER_YEAR
    w = omega * doy[..., None]
    for k in range(1, (coef.shape[1] - 1) // 2 + 1):
        a, b = coef[:, 2 * k - 1], coef[:, 2 * k]
        x = k * omega * config.sediment_relaxation_days
        off = off + (-(a * x * x + b * x) * jnp.cos(k * w)
                     + (a * x - b * x * x) * jnp.sin(k * w)) / (1.0 + x * x)
    return jnp.where(config.sediment_conductance_W_m2_K > 0.0, off, 0.0)


def column_wind_memory_init(day_of_year, config: AutoSimColumnLakeConfig, dtype=None):
    """Initial wind-power memory [W/m2]: the training climatology at ``day_of_year``.

    Falls back to ``wind_power_ref_W_m2`` where the climatology is empty,
    non-finite or not positive.
    """
    doy = jnp.asarray(day_of_year, dtype=dtype)
    coef = jnp.asarray(config.wind_power_climatology_W_m2, dtype=doy.dtype)
    fallback = jnp.full(doy.shape, config.wind_power_ref_W_m2, doy.dtype)
    if coef.size == 0:
        return fallback
    w = (2.0 * math.pi / _DAYS_PER_YEAR) * doy
    cols = [jnp.ones_like(w)]
    for k in range(1, (coef.size - 1) // 2 + 1):
        cols += [jnp.cos(k * w), jnp.sin(k * w)]
    value = jnp.stack(cols, axis=-1) @ coef
    return jnp.where(jnp.isfinite(value) & (value > 0.0), value, fallback)


def step_autosim_column_lake(
    state: ColumnLakeState,
    forcing: AtmToSurface,
    config: AutoSimColumnLakeConfig,
    U_min: float,
    dt: float,
    lat_deg=None,
) -> tuple[ColumnLakeState, TileResponse]:
    """Step the ``autosim_column`` lake by ``dt`` seconds (coupler seat of ``step_lake``).

    ``state.T_lake`` and ``state.T_sediment`` [K] have shape ``spatial +
    (n_levels,)``; ``state.wind_power_memory`` and the forcing fields have the
    spatial shape.  The response reports the level-0 temperature as ``T_sfc``,
    the albedo of the absorbed shortwave (the scheme has no ice) and the
    start-of-step bulk fluxes, as ``step_lake`` does.
    """
    geometry = column_geometry(config, dtype=state.T_lake.data.dtype)
    state_C = (state.T_lake.data - constants.T_freeze,
               state.T_sediment.data - constants.T_freeze,
               state.wind_power_memory.data)
    (T_C_new, S_C_new, E_new), fluxes = column_lake_step_C(
        state_C, forcing, config, geometry, U_min, dt, lat_deg=lat_deg)
    _, _, _, shflx, lhflx, tau_x, tau_y, L_latent = fluxes
    T_new = T_C_new + constants.T_freeze
    T_sfc = T_new[..., 0]
    dtype = T_sfc.dtype
    evap_rate = lhflx / L_latent
    lw_up = (config.emissivity_lake * constants.sigma_sb * T_sfc ** 4
             + (1.0 - config.emissivity_lake) * forcing.lw_down)
    zeros = jnp.zeros_like(T_sfc)
    response = TileResponse(
        T_sfc=T_sfc,
        albedo=jnp.full(T_sfc.shape,
                        1.0 - config.sw_absorption_scale * (1.0 - config.albedo_lake),
                        dtype=dtype),
        emissivity=jnp.full(T_sfc.shape, config.emissivity_lake, dtype=dtype),
        z0=jnp.full(T_sfc.shape, config.z0_lake, dtype=dtype),
        q_surface=saturation_specific_humidity(T_sfc, forcing.p_surface),
        shflx=shflx,
        lhflx=lhflx,
        tau_x=tau_x,
        tau_y=tau_y,
        lw_up=lw_up,
        u_ocean_sfc=zeros,
        v_ocean_sfc=zeros,
        co2_flux=zeros,
        freshwater_flux=forcing.precip_total - evap_rate,
        ocean_heat_extraction=zeros,
        ocean_stress_x=zeros,
        ocean_stress_y=zeros,
        surface_mass_flux=evap_rate,
        salt_flux=zeros,
    )
    new_state = ColumnLakeState(
        T_lake=state.T_lake.replace(data=T_new),
        T_sediment=state.T_sediment.replace(data=S_C_new + constants.T_freeze),
        wind_power_memory=state.wind_power_memory.replace(
            data=E_new.astype(state.wind_power_memory.data.dtype)),
    )
    return new_state, response


def init_autosim_column_lake_state(
    T_init_K,
    shape: tuple[int, ...],
    config: AutoSimColumnLakeConfig,
    dims: tuple[str, ...] | None = None,
    dtype=None,
    T_sediment_K=None,
    wind_power_memory=None,
) -> ColumnLakeState:
    """Column state of spatial ``shape`` from a profile or a scalar ``T_init_K`` [K].

    ``T_init_K`` and ``T_sediment_K`` broadcast against ``shape + (n_levels,)``,
    ``wind_power_memory`` [W/m2] against ``shape``.  Defaults (cold start): the
    sediment at the water temperature and the memory at
    ``wind_power_ref_W_m2`` (memory factor 1).  For a start on a known day,
    :func:`column_sediment_offset_C` and :func:`column_wind_memory_init` give
    the climatological values.  ``dims`` names the spatial axes (default the
    coupler's ``("face", "x", "y")`` for 3-d shapes).
    """
    n_levels = len(config.depths_m)
    full = tuple(shape) + (n_levels,)
    data = jnp.broadcast_to(jnp.asarray(T_init_K, dtype=dtype), full)
    sed = data if T_sediment_K is None else jnp.broadcast_to(
        jnp.asarray(T_sediment_K, dtype=data.dtype), full)
    mem = jnp.broadcast_to(jnp.asarray(
        config.wind_power_ref_W_m2 if wind_power_memory is None else wind_power_memory,
        dtype=data.dtype), tuple(shape))
    if dims is None:
        dims = ("face", "x", "y") if len(shape) == 3 else tuple(f"dim_{i}" for i in range(len(shape)))
    level_dims = tuple(dims) + ("lake_level",)
    return ColumnLakeState(
        T_lake=Field(data=data, name="T_lake", dims=level_dims, units="K"),
        T_sediment=Field(data=sed, name="T_sediment", dims=level_dims, units="K"),
        wind_power_memory=Field(data=mem, name="wind_power_memory", dims=tuple(dims),
                                units="W/m2"),
    )
