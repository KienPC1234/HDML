"""Run with Python + NumPy; no CUDA emulation or fake model results."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('episode_preprocessing', Path(__file__).resolve().parents[1] / 'hdml/data/episodes.py')
ep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ep)


def trajectory(value, n=4):
    return {'observations': np.full((n, 2), value, dtype=np.float32),
            'actions': np.arange(n, dtype=np.float32)[:, None],
            'rewards': np.ones(n, dtype=np.float32)}


class EpisodeRegressionTests(unittest.TestCase):
    def test_windows_do_not_cross_reset_and_include_last_window(self):
        out = ep.pack_episodes([trajectory(1), trajectory(99)], 3)
        np.testing.assert_array_equal(out['valid_starts'], [0, 1, 4, 5])
        for start in out['valid_starts']:
            self.assertEqual(len(np.unique(out['states'][start:start+3, 0])), 1)

    def test_returns_and_timesteps_reset_per_episode(self):
        out = ep.pack_episodes([trajectory(1), trajectory(2)], 2, gamma=1)
        np.testing.assert_array_equal(out['rtgs'][:, 0], [4,3,2,1,4,3,2,1])
        np.testing.assert_array_equal(out['timesteps'], [0,1,2,3,0,1,2,3])

    def test_previous_action_preserved_at_mid_episode_window(self):
        out = ep.pack_episodes([trajectory(1)], 2)
        np.testing.assert_array_equal(out['previous_actions'][:, 0], [0,0,1,2])

    def test_split_groups_duplicate_episodes_and_is_reproducible(self):
        episodes = [trajectory(1), trajectory(2), trajectory(1), trajectory(3)]
        train, val = ep.split_episodes(episodes)
        hashes = lambda xs: [ep.episode_fingerprint(x) for x in xs]
        self.assertFalse(set(hashes(train)) & set(hashes(val)))
        self.assertEqual(hashes(val), hashes(ep.split_episodes(episodes)[1]))

    def test_validation_uses_supplied_train_normalization(self):
        out = ep.pack_episodes([trajectory(100)], 2, state_mean=np.array([1,1]), state_std=np.array([2,2]))
        np.testing.assert_allclose(out['states'], 49.5)

    def test_flat_timeouts_split_episodes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'data.npz'
            np.savez(p, observations=np.zeros((6,2)), actions=np.zeros((6,1)),
                     rewards=np.ones(6), terminals=np.zeros(6), timeouts=[0,0,1,0,0,1])
            episodes = ep.load_episodes(p)
            self.assertEqual([len(x['rewards']) for x in episodes], [3,3])

    def test_explicit_trajectory_ends_reset_without_terminal_flag(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'data.npz'
            ds = {}
            for i in range(2):
                for key, value in trajectory(i).items(): ds[f'traj_{i}_{key}'] = value
            np.savez(p, **ds)
            self.assertEqual(len(ep.load_episodes(p)), 2)

    def test_reject_single_episode_holdout(self):
        with self.assertRaises(ValueError): ep.split_episodes([trajectory(1)])

    def test_reject_short_context(self):
        with self.assertRaises(ValueError): ep.pack_episodes([trajectory(1)], 5)

    def test_reject_nonfinite_input(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'data.npz'
            x = trajectory(1); x['observations'][0,0] = np.nan
            np.savez(p, **x)
            with self.assertRaises(ValueError): ep.load_episodes(p)

if __name__ == '__main__': unittest.main()
