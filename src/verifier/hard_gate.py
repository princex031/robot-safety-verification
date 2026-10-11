"""Documented physical limits. Torque limits must be supplied explicitly."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SafetyLimits:
    max_reach_m: float = 0.72
    min_clearance_m: float = 0.020
    max_required_force_N: float = 40.0
    min_com_margin_m: float = 0.020
    max_slip_probability: float = 0.10


def check_hard_gates(
    *,
    reach_m: float,
    clearance_m: float,
    required_force_N: float,
    com_margin_m: float,
    slip_probability: float,
    joint_torques_Nm: list[float],
    joint_torque_limits_Nm: list[float],
    limits: SafetyLimits = SafetyLimits(),
) -> tuple[bool, list[str]]:
    physical_values = {
        "reach_m": reach_m,
        "clearance_m": clearance_m,
        "required_force_N": required_force_N,
        "com_margin_m": com_margin_m,
        "slip_probability": slip_probability,
    }
    for name, value in physical_values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if len(joint_torques_Nm) != len(joint_torque_limits_Nm):
        raise ValueError("Torque values and limits must have equal lengths")
    if not joint_torque_limits_Nm:
        raise ValueError("Verified joint torque limits are required")
    if any(not math.isfinite(x) for x in joint_torques_Nm + joint_torque_limits_Nm):
        raise ValueError("Torque values and limits must be finite")
    if any(x <= 0 for x in joint_torque_limits_Nm):
        raise ValueError("Torque limits must be positive")

    reasons = []
    if reach_m > limits.max_reach_m:
        reasons.append("reach_limit_exceeded")
    if clearance_m < limits.min_clearance_m:
        reasons.append("collision_clearance_too_small")
    if required_force_N > limits.max_required_force_N:
        reasons.append("required_force_exceeded")
    if com_margin_m < limits.min_com_margin_m:
        reasons.append("com_margin_too_small")
    if slip_probability > limits.max_slip_probability:
        reasons.append("slip_probability_too_high")

    for i, (torque, limit) in enumerate(
        zip(joint_torques_Nm, joint_torque_limits_Nm), start=1
    ):
        if abs(torque) > limit:
            reasons.append(f"joint_{i}_torque_exceeded")

    return not reasons, reasons
