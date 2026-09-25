import os
import tempfile
import unittest
from pathlib import Path

from evaluations.load_helpers import find_seed_eval_npz


class TestFindSeedEvalNpz(unittest.TestCase):
    def test_supports_flat_and_timestamped_run_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            mode_dir = Path(tmp)
            flat = mode_dir / "seed_0" / "eval" / "evaluations.npz"
            older = mode_dir / "seed_1" / "run_old" / "eval" / "evaluations.npz"
            newer = mode_dir / "seed_1" / "run_new" / "eval" / "evaluations.npz"
            for path in (flat, older, newer):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            os.utime(older, (1, 1))
            os.utime(newer, (2, 2))

            found = find_seed_eval_npz(mode_dir)

            self.assertEqual(found["0"], flat)
            self.assertEqual(found["1"], newer)


if __name__ == "__main__":
    unittest.main()
