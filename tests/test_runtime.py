import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_diffusion.runtime import StorageGuard, atomic_checkpoint


class RuntimeTests(unittest.TestCase):
    def test_low_space_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            old = root/'last.pt'
            old.write_bytes(b'previous valid checkpoint')
            guard = StorageGuard(root, root, {'min_free_gib': 8, 'project_cap_gib': 26, 'run_cap_gib': 1,
                                             'tensorboard_cap_mib': 256, 'warning_free_gib': 12})
            with patch('conditional_diffusion.runtime.shutil.disk_usage') as usage:
                usage.return_value.free = 1024
                with self.assertRaises(RuntimeError):
                    atomic_checkpoint(old, {'model': {'a': torch.ones(1)}}, guard)
            self.assertEqual(old.read_bytes(), b'previous valid checkpoint')
            self.assertFalse(old.with_suffix('.pt.tmp').exists())

    def test_failed_atomic_write_preserves_last(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d)/'last.pt'
            old.write_bytes(b'old')
            class Guard:
                def check(self, size):
                    pass
            def failing_save(value, f):
                f.write(b'partial')
                raise OSError('simulated disk failure')
            with patch('conditional_diffusion.runtime.torch.save', failing_save):
                with self.assertRaises(OSError):
                    atomic_checkpoint(old, {'model': {'a': torch.ones(1)}}, Guard())
            self.assertEqual(old.read_bytes(), b'old')
            self.assertFalse(old.with_suffix('.pt.tmp').exists())


if __name__ == '__main__':
    unittest.main()
