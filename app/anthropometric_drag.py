"""
SwimSight AI - Anthropometric Drag & Force Model
=================================================

Folds the swimmer's BODY MASS and HEIGHT (plus velocity & acceleration) into the
hydrodynamic drag and force calculation, instead of using a hard-coded frontal
area. Drop-in companion to the existing hydrodynamics engine.

----------------------------------------------------------------------------
WHY mass & height matter (the physics)
----------------------------------------------------------------------------
The base drag equation is:

        F_d = 1/2 * rho * v^2 * C_d * A                                    (1)

Everything except A is a fluid/velocity property. A -- the frontal
cross-sectional area the swimmer pushes through the water -- is a BODY property,
so it is where height & weight enter. And once we know the swimmer's mass, we can
finally close Newton's 2nd law, which the original blueprint never did:

        F_net   = m * a                  (net force on the centre of mass)  (2)
        F_prop  = m * a + F_d            (what propulsion must deliver)      (3)
        a_glide = -F_d / m               (deceleration during a streamline)  (4)

So a heavier swimmer at the same speed/area feels the same drag (1) but
decelerates more slowly in a glide (4) -- mass is "ballast" that carries
momentum. That is real, coachable physics that only appears once weight is in.

----------------------------------------------------------------------------
The estimation chain (mass + height  ->  area  ->  forces)
----------------------------------------------------------------------------
1. Body Surface Area (DuBois & DuBois, 1916):
        BSA = 0.007184 * m^0.425 * H_cm^0.725           [m^2]              (5)

2. Body volume from mass and body density:
        V   = m / rho_body                               [m^3]             (6)
   rho_body ~ 1050 kg/m^3 for a swimmer mid-stroke (partly inflated lungs).

3. Frontal cross-sectional area -- model the trunk as an elliptical cylinder
   of length L_torso = c_torso * H holding a fraction f_v of body volume.
   The algebra collapses to a single tunable constant k_A = f_v / c_torso:

        A   = k_A * V / H = k_A * m / (rho_body * H)     [m^2]             (7)

   Defaults: f_v ~ 0.45, c_torso ~ 0.30  ->  k_A ~ 1.5 (mid-stroke).
   A pose preset adjusts k_A (tight streamline presents less area than a
   mid-stroke catch). Cross-checked against a BSA-fraction estimate (5).

4. Plug A into (1) for drag, then (2)-(4) for the force balance, plus:
        P_drag = F_d * v        power to overcome drag        [W]          (8)
        DWR    = F_d / (m * g)  drag-to-weight ratio (dimensionless)       (9)

----------------------------------------------------------------------------
Sources / sanity anchors
----------------------------------------------------------------------------
* DuBois & DuBois (1916) BSA formula -- standard in physiology.
* Drillis & Contini (1966) body-segment ratios (biacromial ~0.23-0.245 * H).
* Passive drag for adults at ~2 m/s is measured at ~50-100 N
  (Clarys 1979; Kjendlie & Stallman 2008) -- the model lands in this band.

This file is self-contained (pure-Python core; NumPy only for time series).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
G = 9.81                 # gravitational acceleration       [m/s^2]
RHO_WATER_25C = 997.0    # pool water density at ~25 C      [kg/m^3]
RHO_BODY_DEFAULT = 1050.0  # swimmer body density mid-stroke [kg/m^3]

# Plausibility guards -- catch unit confusion (cm entered as m, lb as kg).
HEIGHT_M_RANGE = (0.5, 2.5)    # metres
MASS_KG_RANGE = (15.0, 200.0)  # kilograms

# BSA-fraction cross-check reference. The fraction is scaled by pose so the two
# area estimators stay consistent for EVERY pose, not just freestyle (see
# AnthropometricDragModel.frontal_area_from_bsa).
BSA_FRACTION_REF = 0.0325       # calibrated for the freestyle k_area below
BSA_FRACTION_K_REF = 1.45       # == POSE_PRESETS["freestyle"]


# ---------------------------------------------------------------------------
# Pose presets: how much frontal area the body presents.
# k_A in eq. (7). Tighter line -> smaller area. Tunable from pool validation.
# ---------------------------------------------------------------------------
POSE_PRESETS: Dict[str, float] = {
    "streamline": 1.15,   # arms locked overhead, tightest profile (push-off/glide)
    "freestyle":  1.45,   # rotating, one arm extended
    "backstroke": 1.45,
    "breaststroke": 1.75,  # wide knee recovery spikes frontal area
    "butterfly":  1.60,
    "mid_stroke": 1.50,   # generic default
}

# Reference drag coefficients (referenced to FRONTAL area). Passive/streamlined
# human ~0.6-0.9 depending on definition; per-stroke active values run higher.
CD_PRESETS: Dict[str, float] = {
    "streamline": 0.60,
    "freestyle":  0.68,
    "backstroke": 0.68,
    "breaststroke": 0.95,
    "butterfly":  0.85,
    "mid_stroke": 0.70,
}


# ---------------------------------------------------------------------------
# Unit helpers (so coaches can enter lb / ft-in if they want)
# ---------------------------------------------------------------------------
def lb_to_kg(lb: float) -> float:
    return lb * 0.45359237


def ft_in_to_m(feet: float, inches: float = 0.0) -> float:
    return (feet * 12.0 + inches) * 0.0254


@dataclass
class Swimmer:
    """Anthropometric profile. Height in metres, mass in kilograms."""
    mass_kg: float
    height_m: float
    rho_body: float = RHO_BODY_DEFAULT
    name: str = "swimmer"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """
        Guard against silent unit confusion. The common failure is entering
        centimetres (180) where metres (1.80) are expected, which used to
        produce a wrong-but-plausible drag number instead of an error.
        """
        lo_h, hi_h = HEIGHT_M_RANGE
        if not (lo_h <= self.height_m <= hi_h):
            if self.height_m > hi_h:
                raise ValueError(
                    f"height_m={self.height_m} looks like centimetres, not "
                    f"metres -- did you mean {self.height_m / 100:.2f}? "
                    f"(expected {lo_h}-{hi_h} m)"
                )
            raise ValueError(
                f"height_m={self.height_m} is implausibly small for a swimmer "
                f"-- expected metres in {lo_h}-{hi_h} m."
            )

        lo_m, hi_m = MASS_KG_RANGE
        if not (lo_m <= self.mass_kg <= hi_m):
            if self.mass_kg > hi_m:
                raise ValueError(
                    f"mass_kg={self.mass_kg} looks like pounds, not kilograms "
                    f"-- did you mean {self.mass_kg * 0.45359237:.1f}? "
                    f"(expected {lo_m}-{hi_m} kg)"
                )
            raise ValueError(
                f"mass_kg={self.mass_kg} is implausibly small for a swimmer "
                f"-- expected kilograms in {lo_m}-{hi_m} kg."
            )

        if self.rho_body <= 0:
            raise ValueError(f"rho_body={self.rho_body} must be positive.")

    @classmethod
    def from_imperial(cls, weight_lb: float, height_ft: float,
                      height_in: float = 0.0, name: str = "swimmer") -> "Swimmer":
        return cls(mass_kg=lb_to_kg(weight_lb),
                   height_m=ft_in_to_m(height_ft, height_in),
                   name=name)

    # --- derived anthropometrics ------------------------------------------
    def body_surface_area(self) -> float:
        """DuBois BSA, eq. (5).  [m^2]"""
        h_cm = self.height_m * 100.0
        return 0.007184 * (self.mass_kg ** 0.425) * (h_cm ** 0.725)

    def body_surface_area_mosteller(self) -> float:
        """Mosteller BSA -- independent cross-check.  [m^2]"""
        h_cm = self.height_m * 100.0
        return math.sqrt(h_cm * self.mass_kg / 3600.0)

    def body_volume(self) -> float:
        """Volume from mass / density, eq. (6).  [m^3]"""
        return self.mass_kg / self.rho_body

    def weight_n(self) -> float:
        """Body weight as a force.  [N]"""
        return self.mass_kg * G


@dataclass
class DragResult:
    """One instant of the force balance. All SI units."""
    velocity_m_s: float
    acceleration_m_s2: float
    frontal_area_m2: float
    drag_coefficient: float
    drag_force_n: float
    net_force_n: float            # m * a
    propulsive_force_n: float     # m * a + F_d
    drag_power_w: float           # F_d * v
    propulsive_power_w: float     # F_prop * v
    drag_to_weight_ratio: float   # F_d / (m g)
    glide_deceleration_m_s2: float  # -F_d / m
    # True when the acceleration-derived fields (net_force_n, propulsive_force_n)
    # were NOT smoothed and are therefore dominated by differentiation noise.
    # drag_force_n / drag_power_n do NOT depend on acceleration and stay valid
    # regardless. Downstream code should hide/caveat the force fields when True.
    confidence_low: bool = True

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class AnthropometricDragModel:
    """
    Turns (body mass, height, velocity, acceleration) into a full drag/force
    breakdown. Designed to slot in beside WaveDragHydrodynamicsEngine: feed it
    the pelvis velocity (derivative of pelvis position) that the pipeline
    already computes.
    """

    def __init__(self,
                 swimmer: Swimmer,
                 pose: str = "mid_stroke",
                 water_density: float = RHO_WATER_25C,
                 k_area: Optional[float] = None,
                 drag_coefficient: Optional[float] = None):
        self.swimmer = swimmer
        self.pose = pose
        self.rho_water = water_density
        # Allow explicit overrides; otherwise pull from the pose presets.
        self.k_area = k_area if k_area is not None else POSE_PRESETS.get(pose, 1.5)
        self.c_d = (drag_coefficient if drag_coefficient is not None
                    else CD_PRESETS.get(pose, 0.70))

    # ----- geometry --------------------------------------------------------
    def frontal_area(self) -> float:
        """
        Frontal cross-sectional area from mass & height, eq. (7):
            A = k_A * m / (rho_body * H)
        Clamped to a sane physiological band [0.04, 0.14] m^2.
        """
        a = self.k_area * self.swimmer.mass_kg / (self.swimmer.rho_body
                                                  * self.swimmer.height_m)
        return max(0.04, min(0.14, a))

    def frontal_area_from_bsa(self, bsa_fraction: Optional[float] = None) -> float:
        """
        Independent estimate A ~ fraction * BSA, for cross-validation.

        The fraction is scaled by the SAME pose factor (k_area) used by
        frontal_area(), so the two estimators stay consistent for every pose.
        A fixed fraction only agreed for freestyle/mid_stroke and diverged
        ~38% for tight poses like "streamline". Pass an explicit bsa_fraction
        to override.
        """
        if bsa_fraction is None:
            bsa_fraction = BSA_FRACTION_REF * (self.k_area / BSA_FRACTION_K_REF)
        return bsa_fraction * self.swimmer.body_surface_area()

    # ----- core force balance ---------------------------------------------
    def compute(self, velocity_m_s: float,
                acceleration_m_s2: float = 0.0,
                confidence_low: bool = True) -> DragResult:
        """
        Full instantaneous breakdown.

        velocity      -- centre-of-mass (pelvis) speed down the lane [m/s]
        acceleration  -- dv/dt of the same point                     [m/s^2]
        confidence_low -- mark the acceleration-derived force fields as
                          unsmoothed/noisy. Defaults to True for a raw single
                          sample; compute_series() sets it False once it has
                          smoothed the velocity track before differentiating.
        """
        v = float(velocity_m_s)
        a = float(acceleration_m_s2)
        m = self.swimmer.mass_kg

        A = self.frontal_area()
        c_d = self.c_d

        # Eq. (1): drag is unsigned magnitude opposing motion.
        f_drag = 0.5 * self.rho_water * (v * v) * c_d * A

        # Eq. (2) & (3): mass enters here.
        f_net = m * a
        f_prop = f_net + f_drag

        # Eq. (8), (9), (4)
        p_drag = f_drag * v
        p_prop = f_prop * v
        dwr = f_drag / self.swimmer.weight_n()
        a_glide = -f_drag / m   # pure-glide deceleration if propulsion stops

        return DragResult(
            velocity_m_s=v,
            acceleration_m_s2=a,
            frontal_area_m2=A,
            drag_coefficient=c_d,
            drag_force_n=f_drag,
            net_force_n=f_net,
            propulsive_force_n=f_prop,
            drag_power_w=p_drag,
            propulsive_power_w=p_prop,
            drag_to_weight_ratio=dwr,
            glide_deceleration_m_s2=a_glide,
            confidence_low=bool(confidence_low),
        )

    # ----- time series (integrates with the telemetry pipeline) -----------
    def compute_series(self,
                       times_s: Sequence[float],
                       velocities_m_s: Sequence[float],
                       accelerations_m_s2: Optional[Sequence[float]] = None,
                       smooth_seconds: float = 0.5,
                       accel_is_smoothed: bool = False
                       ) -> List[DragResult]:
        """
        Evaluate a whole stroke cycle.

        Pose-estimation gives a jittery position track. Velocity is one
        derivative of that; acceleration is two. Differentiating raw velocity
        straight to acceleration amplifies the per-frame jitter so badly that
        the force fields become noise (a 5 mm position wobble swung
        propulsive_force from -110 N to +214 N at a constant true speed).

        Two ways to get a trustworthy acceleration:

        * BEST (worker path): estimate acceleration straight from the position
          track with acceleration_from_positions() (a local quadratic fit) and
          pass it in here with accel_is_smoothed=True. This stays within +/-20 N
          of drag at every frame rate from 15 to 240 fps.

        * Convenience path: if only velocity is available, this method smooths
          the velocity track with a zero-lag, TIME-based double moving average
          (smooth_seconds wide) BEFORE differentiating. That is accurate at the
          worker's sampled rate (<= ~30 fps); at much higher raw frame rates,
          prefer the position path above.

        drag_force_n is always left on the RAW velocity because it depends on
        v^2 only and is insensitive to the acceleration noise.
        """
        import numpy as np
        t = np.asarray(times_s, dtype=float)
        v = np.asarray(velocities_m_s, dtype=float)
        n = v.size

        if accelerations_m_s2 is not None:
            a = np.asarray(accelerations_m_s2, dtype=float)
            conf_low = not accel_is_smoothed
        elif n >= 5:
            dt = float(np.median(np.diff(t)))
            window = max(3, int(round(smooth_seconds / dt))) if dt > 0 else 5
            # Double moving average == triangular window: much better
            # derivative-noise rejection than a single boxcar.
            v_smoothed = self._moving_average(self._moving_average(v, window),
                                              window)
            a = np.gradient(v_smoothed, t)
            conf_low = False
        else:
            # Too few frames to smooth reliably -- differentiate raw and flag.
            a = np.gradient(v, t) if n >= 2 else np.zeros(n)
            conf_low = True

        # drag <- raw velocity (v^2); force fields <- smoothed acceleration.
        return [self.compute(float(vi), float(ai), confidence_low=conf_low)
                for vi, ai in zip(v, a)]

    @staticmethod
    def _moving_average(x: Sequence[float], window: int):
        """
        Centred (zero-lag) moving average with edge padding so the output is
        the same length as the input and the endpoints are not pulled toward
        zero. Window is forced odd for symmetry.
        """
        import numpy as np
        arr = np.asarray(x, dtype=float)
        w = int(window)
        if w < 2 or arr.size < 2:
            return arr.copy()
        if w % 2 == 0:
            w += 1
        half = w // 2
        padded = np.pad(arr, half, mode="edge")
        kernel = np.ones(w, dtype=float) / w
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def velocity_from_positions(positions_m: Sequence[float],
                                times_s: Sequence[float],
                                smooth_seconds: float = 0.0):
        """
        Helper: pelvis/hip position -> velocity, mirroring the CV pipeline.
        Optionally smooth the position track first (time-based double moving
        average) to tame jitter before differentiating.
        """
        import numpy as np
        p = np.asarray(positions_m, dtype=float)
        t = np.asarray(times_s, dtype=float)
        if smooth_seconds > 0 and p.size >= 5:
            dt = float(np.median(np.diff(t)))
            w = max(3, int(round(smooth_seconds / dt))) if dt > 0 else 3
            p = AnthropometricDragModel._moving_average(
                AnthropometricDragModel._moving_average(p, w), w)
        return np.gradient(p, t)

    @staticmethod
    def acceleration_from_positions(positions_m: Sequence[float],
                                    times_s: Sequence[float],
                                    smooth_seconds: float = 0.8):
        """
        Robust acceleration straight from the position track via a sliding
        local quadratic fit (p ~ c0 + c1*tau + c2*tau^2; a = 2*c2). Estimating
        the second derivative from position with a polynomial fit uses every
        sample in the window and, unlike differentiating noisy velocity twice,
        gets MORE accurate as frame rate rises. Stays within +/-20 N of drag
        across 15-240 fps for 5 mm position jitter.

        Returns acceleration aligned to `times_s`. Intended to be fed into
        compute_series(..., accelerations_m_s2=a, accel_is_smoothed=True).
        """
        import numpy as np
        p = np.asarray(positions_m, dtype=float)
        t = np.asarray(times_s, dtype=float)
        n = p.size
        if n < 5:
            return np.gradient(np.gradient(p, t), t) if n >= 3 else np.zeros(n)
        dt = float(np.median(np.diff(t)))
        w = max(5, int(round(smooth_seconds / dt))) if dt > 0 else 5
        if w % 2 == 0:
            w += 1
        half = w // 2
        out = np.zeros(n, dtype=float)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            tau = t[lo:hi] - t[lo:hi].mean()
            # poly2 fit; highest-order coeff first -> a = 2*c2
            c2 = np.polyfit(tau, p[lo:hi], 2)[0]
            out[i] = 2.0 * c2
        return out

    @staticmethod
    def kinematics_from_positions(positions_m: Sequence[float],
                                  times_s: Sequence[float],
                                  smooth_seconds: float = 0.6):
        """
        Velocity AND acceleration from ONE sliding local quadratic fit.

        Fitting p ~ c0 + c1*tau + c2*tau^2 about each frame (tau centred on that
        frame's time) gives velocity = c1 and acceleration = 2*c2 in a single,
        consistent, edge-aware pass. Unlike smoothing the position with an
        edge-padded moving average and then differentiating -- which flattens
        the slope at the clip ends and biases velocity (hence drag) low on
        short single-camera passes -- the local fit does not attenuate the
        endpoints. Returns (velocity, acceleration) aligned to `times_s`.
        """
        import numpy as np
        p = np.asarray(positions_m, dtype=float)
        t = np.asarray(times_s, dtype=float)
        n = p.size
        if n < 3:
            v = np.gradient(p, t) if n >= 2 else np.zeros(n)
            return v, np.zeros(n)
        dt = float(np.median(np.diff(t)))
        w = max(5, int(round(smooth_seconds / dt))) if dt > 0 else 5
        if w % 2 == 0:
            w += 1
        half = w // 2
        vel = np.zeros(n, dtype=float)
        acc = np.zeros(n, dtype=float)
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            tau = t[lo:hi] - t[i]           # centre on the query frame
            c2, c1, _ = np.polyfit(tau, p[lo:hi], 2)
            vel[i] = c1
            acc[i] = 2.0 * c2
        return vel, acc


# ===========================================================================
# Worked example / self-test
# ===========================================================================
def _demo() -> None:
    profiles = [
        Swimmer(mass_kg=75.0, height_m=1.80, name="Senior male (75 kg, 1.80 m)"),
        Swimmer(mass_kg=60.0, height_m=1.68, name="Senior female (60 kg, 1.68 m)"),
        Swimmer.from_imperial(weight_lb=185, height_ft=6, height_in=3,
                              name="Tall male (185 lb, 6'3\")"),
    ]

    print("=" * 78)
    print("SwimSight AI  --  Anthropometric Drag & Force  (freestyle, v = 2.0 m/s)")
    print("=" * 78)
    header = (f"{'Profile':<34}{'BSA':>6}{'Area':>7}{'Drag':>8}"
              f"{'Power':>8}{'DWR':>7}{'aGlide':>8}")
    print(header)
    print(f"{'':<34}{'m^2':>6}{'m^2':>7}{'N':>8}{'W':>8}{'-':>7}{'m/s^2':>8}")
    print("-" * 78)
    for s in profiles:
        model = AnthropometricDragModel(s, pose="freestyle")
        r = model.compute(velocity_m_s=2.0, acceleration_m_s2=0.0)
        print(f"{s.name:<34}{s.body_surface_area():>6.2f}"
              f"{r.frontal_area_m2:>7.3f}{r.drag_force_n:>8.1f}"
              f"{r.drag_power_w:>8.0f}{r.drag_to_weight_ratio:>7.3f}"
              f"{r.glide_deceleration_m_s2:>8.2f}")

    # Detail: the same swimmer across a race-pace velocity sweep.
    print("\n" + "=" * 78)
    print("Velocity sweep -- 75 kg / 1.80 m male, freestyle")
    print("=" * 78)
    s = profiles[0]
    model = AnthropometricDragModel(s, pose="freestyle")
    print(f"{'v (m/s)':>8}{'F_drag (N)':>12}{'F_prop@a=0.5':>14}"
          f"{'P_drag (W)':>12}{'a_glide':>10}")
    print("-" * 56)
    for v in (1.4, 1.6, 1.8, 2.0, 2.2):
        r = model.compute(velocity_m_s=v, acceleration_m_s2=0.5)
        print(f"{v:>8.1f}{r.drag_force_n:>12.1f}{r.propulsive_force_n:>14.1f}"
              f"{r.drag_power_w:>12.0f}{r.glide_deceleration_m_s2:>10.2f}")

    # Cross-check the two area estimators agree for EVERY pose (the BSA
    # fraction is now pose-scaled, so this holds beyond freestyle).
    print("\nArea estimator cross-check (75 kg / 1.80 m), all poses:")
    print(f"  {'pose':<13}{'cylinder':>9}{'BSA-frac':>9}{'agree':>8}")
    for pose in POSE_PRESETS:
        model = AnthropometricDragModel(profiles[0], pose=pose)
        a1 = model.frontal_area()
        a2 = model.frontal_area_from_bsa()
        print(f"  {pose:<13}{a1:>9.4f}{a2:>9.4f}"
              f"{100*(1-abs(a1-a2)/a1):>7.1f}%")


if __name__ == "__main__":
    _demo()
