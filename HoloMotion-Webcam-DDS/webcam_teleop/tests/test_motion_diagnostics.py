import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_diagnostics import MotionDiagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_log_preserves_actual_target_and_error_without_qualifying_arrival(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'motion.jsonl'
            writer = MotionDiagnostics(path)
            actual = [.1, .2]
            transport = SimpleNamespace(_state=SimpleNamespace(q=actual, dq=[.01, .02]),
                names=['left_shoulder_pitch_joint', 'left_elbow_joint'],
                _last_command=[.3, .4], state_seq=42, state_age_ms=12.,
                _state_time=time.monotonic() - .012,
                control_mode='arm_sdk', robot_state=4)
            alignment = SimpleNamespace(target=[.5, .6], error=.4, speed=.02, status='moving')
            writer.record(transport, 'alignment', reference=[.5, .6], alignment=alignment)
            actual[0] = 99.  # Asynchronous writes must preserve the sampled values.
            writer.close()
            record = json.loads(path.read_text())
            self.assertEqual(record['actual_q_rad'], [.1, .2])
            self.assertEqual(record['command_q_rad'], [.3, .4])
            self.assertEqual(record['frozen_target_q_rad'], [.5, .6])
            self.assertEqual(record['max_alignment_error_rad'], .4)
            self.assertFalse(record['observed_settled'])
            self.assertEqual(record['state_seq'], 42)


if __name__ == '__main__':
    unittest.main()
