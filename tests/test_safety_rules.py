import unittest
from src.verifier.hard_gate import check_hard_gates
from src.models.seven_dof_arm import within_joint_limits


class SafetyTests(unittest.TestCase):
    def kwargs(self):
        return dict(
            reach_m=0.70, clearance_m=0.025, required_force_N=20,
            com_margin_m=0.025, slip_probability=0.05,
            joint_torques_Nm=[1.0] * 7,
            joint_torque_limits_Nm=[2.0] * 7,
        )

    def test_safe_example(self):
        safe, reasons = check_hard_gates(**self.kwargs())
        self.assertTrue(safe)
        self.assertEqual(reasons, [])

    def test_clearance_failure(self):
        args = self.kwargs()
        args["clearance_m"] = 0.010
        safe, reasons = check_hard_gates(**args)
        self.assertFalse(safe)
        self.assertIn("collision_clearance_too_small", reasons)

    def test_torque_limit_required(self):
        args = self.kwargs()
        args["joint_torque_limits_Nm"] = []
        with self.assertRaises(ValueError):
            check_hard_gates(**args)

    def test_seven_joint_limits(self):
        self.assertTrue(within_joint_limits([0] * 7, [-1] * 7, [1] * 7))


if __name__ == "__main__":
    unittest.main()
