"""Seven-joint input validation, not a full robot kinematics simulator."""


def validate_seven(values: list[float], label: str = "values") -> None:
    if len(values) != 7:
        raise ValueError(f"{label} must contain exactly 7 values")
    if any(not isinstance(v, (int, float)) for v in values):
        raise ValueError(f"{label} must be numeric")


def within_joint_limits(
    joint_values: list[float],
    lower_limits: list[float],
    upper_limits: list[float],
) -> bool:
    validate_seven(joint_values, "joint_values")
    validate_seven(lower_limits, "lower_limits")
    validate_seven(upper_limits, "upper_limits")
    if any(lo > hi for lo, hi in zip(lower_limits, upper_limits)):
        raise ValueError("Lower joint limit exceeds upper limit")
    return all(
        lo <= value <= hi
        for value, lo, hi in zip(joint_values, lower_limits, upper_limits)
    )
