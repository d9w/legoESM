"""Lake model configuration."""

from __future__ import annotations

from typing import NamedTuple

from legoesm import constants

__param_spec__ = {
    "LakeConfig": {
        "scheme_key": "coupler.lake",
        "excluded": {"z_ref": "convention: reference height [m]"},
        "params": {
            "h_epi": {
                "units": "m", "bounds": (1.0, 20.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "two-layer lake epilimnion depth", "shape": None,
            },
            "h_hypo": {
                "units": "m", "bounds": (5.0, 100.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "two-layer lake hypolimnion depth", "shape": None,
            },
            "k_mix": {
                "units": "m^2/s", "bounds": (1.0e-3, 1.0e-1), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "lake epi-hypo turbulent mixing coefficient", "shape": None,
            },
            "wind_mix_alpha": {
                "units": "1", "bounds": (0.01, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "lake wind-driven mixing efficiency", "shape": None,
            },
            "Cd_lake": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "lake-atmosphere momentum drag coefficient", "shape": None,
            },
            "Ch_lake": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "lake-atmosphere heat transfer coefficient", "shape": None,
            },
            "albedo_lake": {
                "units": "1", "bounds": (0.03, 0.2), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "lake surface broadband albedo", "shape": None,
            },
            "albedo_lake_ice": {
                "units": "1", "bounds": (0.3, 0.7), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "frozen-lake (ice/snow) broadband albedo", "shape": None,
            },
            "emissivity_lake": {
                "units": "1", "bounds": (0.9, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "lake surface longwave emissivity", "shape": None,
            },
            "z0_lake": {
                "units": "m", "bounds": (1.0e-5, 1.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "lake aerodynamic roughness length", "shape": None,
            },
        },
    },
    "AutoSimColumnLakeConfig": {
        "scheme_key": "coupler.lake_autosim_column",
        "excluded": {
            "z0_lake": "reported roughness only; the constant bulk law does not use it",
            "wind_2m_ratio": "convention: 2 m / 10 m wind of a log profile, z0 = 1 mm",
            "wind_2m_min_m_s": "numerics: floor of the 2 m wind in the Hostetler terms",
            "n2_floor_s2": "numerics: N^2 floor of the Fang-Stefan background",
            "lake_depth_m": "site geometry: maximum depth of the hypsograph",
            "latitude_deg": "site geometry: latitude used when no column latitude is passed",
            "lake_area_m2": "site geometry: surface area in the Lake Number",
            "wind_power_ref_W_m2": "site climatology: mean training wind power (fit statistic)",
        },
        "params": {
            "albedo_lake": {
                "units": "1", "bounds": (0.03, 0.2), "tunable_tier": 1,
                "transform": "sigmoid", "category": "radiation",
                "reference": "lake surface broadband albedo (LakeConfig value)", "shape": None,
            },
            "emissivity_lake": {
                "units": "1", "bounds": (0.9, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "lake surface longwave emissivity (LakeConfig value)", "shape": None,
            },
            "Cd_lake": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "lake-atmosphere drag coefficient (LakeConfig value): stress, "
                             "wind power and Lake Number",
                "shape": None,
            },
            "Ch_lake": {
                "units": "1", "bounds": (5.0e-4, 5.0e-3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "surface",
                "reference": "LakeConfig constant transfer coefficient times the fitted "
                             "AutoSim scale (ch_scale)", "shape": None,
            },
            "q_correction_W_m2": {
                "units": "W/m^2", "bounds": (-60.0, 60.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s004 and s016 fitted constant surface heat "
                             "flux (unresolved heat terms); not exchanged with the atmosphere",
                "shape": None,
            },
            "sw_absorption_scale": {
                "units": "1", "bounds": (0.5, 1.2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "AutoSim run 295835 s016 fitted scale of the absorbed shortwave "
                             "(sw_scale); the reported albedo carries it", "shape": None,
            },
            "sw_extinction_m": {
                "units": "1/m", "bounds": (0.1, 5.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "radiation",
                "reference": "AutoSim run 295835 s016 fitted shortwave extinction K_d",
                "shape": None,
            },
            "hostetler_kstar_coef": {
                "units": "1", "bounds": (0.05, 10.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "Hostetler and Bartlein (1990) Ekman decay coefficient (6.6); "
                             "AutoSim run 295835 s002 fitted value", "shape": None,
            },
            "hostetler_wind_scale": {
                "units": "1", "bounds": (0.1, 100.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s016 fitted scale of the Hostetler wind "
                             "diffusivity (sK)", "shape": None,
            },
            "background_diffusivity_scale": {
                "units": "1", "bounds": (1.0, 300.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s016 fitted scale of the background "
                             "diffusivity (sEd), anchored at the Fang and Stefan (1996) value "
                             "at n2_floor_s2", "shape": None,
            },
            "k_convective_m2_s": {
                "units": "m^2/s", "bounds": (0.01, 10.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "diffusivity on statically unstable faces (AutoSim run 295835 "
                             "s002)", "shape": None,
            },
            "background_n2_exponent": {
                "units": "1", "bounds": (0.2, 1.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s006 and s016 fitted N^2 exponent of the "
                             "background diffusivity (q_bg); 0.43 in Fang and Stefan (1996)",
                "shape": None,
            },
            "weak_stratification_n2_s2": {
                "units": "1/s^2", "bounds": (0.0, 1.0e-2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s014 threshold of the weak-stratification "
                             "branch (n2_low)", "shape": None,
            },
            "weak_stratification_n2_exponent": {
                "units": "1", "bounds": (0.0, 2.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s014 N^2 exponent of the weak-stratification "
                             "branch (q_low)", "shape": None,
            },
            "weak_stratification_n2_floor_s2": {
                "units": "1/s^2", "bounds": (1.0e-9, 1.0e-2), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s014 N^2 floor of the weak-stratification "
                             "branch (n2_low_floor), caps its diffusivity", "shape": None,
            },
            "shelter_inverse_time_per_day": {
                "units": "1/day", "bounds": (0.0, 100.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s007 fitted (1 day)/T_e of the wind work "
                             "that sets the sheltering exp(-PE / (rho_w u*^3 T_e)) (inv_te)",
                "shape": None,
            },
            "lake_number_gate_amplitude": {
                "units": "1", "bounds": (0.0, 50.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s008 fitted amplitude of the Lake Number "
                             "gate 1 + beta exp(-L_N / L_c) (beta); Imberger and Patterson "
                             "(1990) Lake Number", "shape": None,
            },
            "lake_number_critical": {
                "units": "1", "bounds": (1.0e-3, 100.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s008 fitted critical Lake Number (lc)",
                "shape": None,
            },
            "penetrative_convection_coef": {
                "units": "1", "bounds": (0.0, 1.0e3), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s012 fitted criterion N^2 z^2 < c w_c^2 of "
                             "penetrative convection under net surface cooling (c_pen)",
                "shape": None,
            },
            "wind_memory_exponent": {
                "units": "1", "bounds": (0.0, 3.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s016 fitted exponent of the wind-power "
                             "memory factor (gamma_w)", "shape": None,
            },
            "wind_memory_days": {
                "units": "day", "bounds": (0.5, 120.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s016 fitted e-folding time of the wind-power "
                             "memory (tau_w)", "shape": None,
            },
            "sediment_conductance_W_m2_K": {
                "units": "W/(m^2 K)", "bounds": (0.0, 20.0), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s010 fitted water-sediment heat exchange "
                             "per m2 of bed (h_sed)", "shape": None,
            },
            "sediment_relaxation_days": {
                "units": "day", "bounds": (1.0, 1.0e4), "tunable_tier": 2,
                "transform": "sigmoid", "category": "closure",
                "reference": "AutoSim run 295835 s010 fitted relaxation time of the bed "
                             "sediment slab (tau_sed)", "shape": None,
            },
        },
    },
}


class LakeConfig(NamedTuple):
    """Two-layer lake (epilimnion + hypolimnion) configuration."""
    h_epi: float = 5.0              # Epilimnion depth [m]
    h_hypo: float = 20.0            # Hypolimnion depth [m]
    rho_water: float = constants.rho_water
    c_water_mass: float = constants.c_pw
    k_mix: float = 1.0e-2           # Vertical mixing coefficient [m2/s]
    wind_mix_alpha: float = 0.1     # Wind-driven mixing enhancement factor
    albedo_lake: float = 0.08
    albedo_lake_ice: float = 0.6    # frozen-lake ice/snow broadband albedo
    emissivity_lake: float = 0.97
    z0_lake: float = 1e-4           # Roughness length [m]
    Cd_lake: float = 1.5e-3         # Drag coefficient (constant scheme)
    Ch_lake: float = 1.5e-3         # Heat transfer coefficient (constant)
    T_freeze: float = constants.T_freeze
    # Freshwater EOS parameters (T_freshwater_max_density and
    # rho_freshwater_curvature) live in legoesm.constants and are not
    # repeated here — physical constants belong in the constants
    # module per CLAUDE.md.
    bulk_scheme: str = "constant"   # "constant" or "most"
    z_ref: float = 10.0             # Reference height for MOST [m]
    bulk_n_iter: int = 5            # MOST iterations


# --- Lough Feeagh hypsography (AutoSim run 295835 s002) ---
# Observed temperature depths of the LakeEnsemblR Feeagh profiles [m] and the
# lake area at each depth as a fraction of the surface area, as carried by the
# AutoSim program; the area is 1 at the surface and 0 at ``lake_depth_m``.
FEEAGH_DEPTHS_M = (0.9, 2.5, 5.0, 8.0, 11.0, 14.0, 16.0, 18.0, 20.0, 22.0, 27.0, 32.0, 42.0)
FEEAGH_AREA_FRACTION = (0.944, 0.863, 0.797, 0.709, 0.617, 0.523, 0.466, 0.401, 0.346,
                        0.298, 0.190, 0.104, 0.019)
# Train climatology of the water temperature at each depth [degC], columns
# (mean, cos 1, sin 1, cos 2, sin 2) of the annual harmonics in 2 pi doy / 365.25,
# and of the wind power rho_a Cd U^3 [W/m2], same columns (AutoSim run 295835
# s016 ``fit`` return on 2533 distinct training days).  They set the initial
# sediment temperatures and the initial wind-power memory of a window.
FEEAGH_TEMPERATURE_CLIMATOLOGY_C = (
    (11.30214, -4.74282, -3.15378, 0.10137, -0.19441),
    (11.18014, -4.64697, -3.17256, 0.10039, -0.19383),
    (11.0397, -4.50455, -3.208, 0.02911, -0.1525),
    (10.9438, -4.32276, -3.24287, -0.0552, -0.11243),
    (10.82381, -4.16631, -3.29897, -0.14283, -0.06624),
    (10.62389, -3.85959, -3.3451, -0.28161, -0.00128),
    (10.54095, -3.68022, -3.3505, -0.4026, 0.01234),
    (10.41494, -3.48477, -3.35639, -0.51001, -0.01801),
    (10.27294, -3.25909, -3.32782, -0.6361, -0.08559),
    (10.12233, -3.04483, -3.25655, -0.75055, -0.1952),
    (9.82726, -2.59316, -2.98351, -0.85325, -0.55433),
    (9.61424, -2.42195, -2.78641, -0.76011, -0.66818),
    (9.40756, -2.32269, -2.66284, -0.57243, -0.68861),
)
FEEAGH_WIND_POWER_CLIMATOLOGY_W_M2 = (0.334293, 0.12438, -0.0184999, 0.0142102, -0.0259803)


class AutoSimColumnLakeConfig(NamedTuple):
    """Multi-level hypsographic lake column ported from AutoSim (``autosim_column``).

    Defaults are the values of AutoSim run 295835 candidate s016 (its ``fit``
    return, fitted on 160 of the 465 training windows).  Earlier candidates of
    the same lineage are this scheme with terms switched off: s004 sets
    ``background_n2_exponent=0.43`` (Fang and Stefan) and zero
    ``weak_stratification_n2_s2``, ``shelter_inverse_time_per_day``,
    ``lake_number_gate_amplitude``, ``penetrative_convection_coef``,
    ``sediment_conductance_W_m2_K`` and ``wind_memory_exponent``, with its own
    fitted values (``q_correction_W_m2=43.75``, ``sw_absorption_scale=0.825``,
    ``sw_extinction_m=1.225``, ``hostetler_wind_scale=3``,
    ``background_diffusivity_scale=30``).  See ``autosim_column.py`` for the
    equations.  Selecting this config instead of :class:`LakeConfig` selects
    the scheme (``step_lake_tile``).
    """
    # --- surface energy balance (legoESM constant bulk law) ---
    albedo_lake: float = 0.08
    emissivity_lake: float = 0.97
    Cd_lake: float = 1.5e-3         # momentum drag: stress, wind power, Lake Number
    Ch_lake: float = 1.5e-3         # heat and moisture transfer coefficient
    z0_lake: float = 1e-4           # reported roughness length [m]
    q_correction_W_m2: float = 43.75  # constant downward heat flux into level 0 [W/m2]
    # --- shortwave absorption and penetration ---
    sw_absorption_scale: float = 0.8125  # fraction of the open-water net shortwave absorbed
    sw_extinction_m: float = 0.9    # extinction coefficient K_d [1/m]
    # --- wind mixing (Hostetler term, sheltering, Lake Number gate) ---
    hostetler_kstar_coef: float = 0.5
    hostetler_wind_scale: float = 5.0
    shelter_inverse_time_per_day: float = 1.5  # (1 day) / T_e of the wind work; 0 = off
    lake_number_gate_amplitude: float = 4.0    # beta of 1 + beta exp(-L_N / L_c); 0 = off
    lake_number_critical: float = 0.1          # L_c
    # --- background diffusivity (stratification law, wind-power memory) ---
    background_diffusivity_scale: float = 26.4706
    background_n2_exponent: float = 0.25       # q of K_s ~ N^-2q; 0.43 = Fang and Stefan
    weak_stratification_n2_s2: float = 1.0e-4  # N^2 below which the weak branch applies; 0 = off
    weak_stratification_n2_exponent: float = 0.75
    weak_stratification_n2_floor_s2: float = 1.0e-5
    wind_memory_exponent: float = 0.5          # gamma of (E / P_ref)^gamma; 0 = off
    wind_memory_days: float = 28.0             # e-folding time of the wind-power memory E
    wind_power_ref_W_m2: float = 0.329282      # P_ref, mean train wind power rho_a Cd U^3
    # --- convection ---
    k_molecular_m2_s: float = constants.kappa_T_ocean_molecular
    k_convective_m2_s: float = 1.0
    penetrative_convection_coef: float = 48.0  # c of N^2 z^2 < c w_c^2; 0 = off
    wind_2m_ratio: float = 0.83
    wind_2m_min_m_s: float = 0.5
    n2_floor_s2: float = 7.5e-5
    # --- bed sediment ---
    sediment_conductance_W_m2_K: float = 0.625  # water-slab exchange per m2 of bed; 0 = off
    sediment_relaxation_days: float = 1171.88   # slab time scale; capacity = conductance x time
    # --- water properties ---
    rho_water: float = constants.rho_water
    c_water_mass: float = constants.c_pw
    # --- initial-state climatologies (fitted on the training years) ---
    temperature_climatology_C: tuple = FEEAGH_TEMPERATURE_CLIMATOLOGY_C
    wind_power_climatology_W_m2: tuple = FEEAGH_WIND_POWER_CLIMATOLOGY_W_M2
    # --- site geometry (static) ---
    depths_m: tuple = FEEAGH_DEPTHS_M
    area_fraction: tuple = FEEAGH_AREA_FRACTION
    lake_depth_m: float = 46.8
    lake_area_m2: float = 3.931e6
    latitude_deg: float = 53.9
