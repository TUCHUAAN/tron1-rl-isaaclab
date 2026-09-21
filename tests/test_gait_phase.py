"""Exercise the real gait clock against Isaac's reward/resample/reset ordering."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch


class CommandTermStub:
    def __init__(self, cfg, env):
        self.cfg, self._env = cfg, env
        self.num_envs, self.device = env.num_envs, env.device
        self.metrics = {}

    def reset(self, env_ids):
        self._resample_command(env_ids)
        return {}


_PATH = (Path(__file__).resolve().parents[1]
         / "exts/bipedal_locomotion/bipedal_locomotion/tasks/locomotion/mdp/commands/gait_command.py")
_CLASS = next(n for n in ast.parse(_PATH.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "GaitCommand")
_MODULE = ast.Module(body=[
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), _CLASS,
], type_ignores=[])
_NS = {"torch": torch, "CommandTerm": CommandTermStub}
exec(compile(ast.fix_missing_locations(_MODULE), str(_PATH), "exec"), _NS)
GaitCommand = _NS["GaitCommand"]


def make_gait(n=2, continuous=True, dt=.02):
    env = NS(num_envs=n, device="cpu", step_dt=dt, common_step_counter=0,
             episode_length_buf=torch.zeros(n, dtype=torch.long))
    cfg = NS(continuous_phase=continuous,
             ranges=NS(frequencies=(1.2, 1.2), offsets=(.5, .5), durations=(.5, .5), swing_height=(0., 0.)))
    term = GaitCommand(cfg, env)
    term.reset()
    return env, term


def advance(env, n=1):
    env.common_step_counter += n
    env.episode_length_buf += n


class GaitPhaseTest(unittest.TestCase):
    def test_constant_frequency_agrees_with_legacy_clock(self):
        env, term = make_gait()
        for _ in range(1000):
            advance(env)
            phase = term.phase.clone()
            legacy = (env.episode_length_buf * env.step_dt * term.command[:, 0]).remainder(1.)
            # Compare circular phase (0 and 1 are the same point).
            error = (phase - legacy + .5).remainder(1.) - .5
            self.assertLess(error.abs().max().item(), 3.e-5)

    def test_resampling_uses_old_frequency_for_elapsed_physics(self):
        for reward_read in (False, True):
            env, term = make_gait()
            advance(env, 19)
            before = term.phase.clone() if reward_read else torch.full((2,), 19 * .02 * 1.2)
            term.cfg.ranges.frequencies = (2.2, 2.2)
            term._resample_command(torch.arange(2))
            term._update_command()
            torch.testing.assert_close(term.phase, before)
            advance(env)
            torch.testing.assert_close(term.phase, (before + .02 * 2.2).remainder(1.))

    def test_repeated_reward_observation_and_command_reads_do_not_advance(self):
        env, term = make_gait()
        advance(env, 7)
        expected = term.phase.clone()
        for _ in range(5):
            term._update_command()
            torch.testing.assert_close(term.phase, expected)

    def test_partial_reset_before_episode_buffer_clears_does_not_rewind_other_envs(self):
        for ids in ([0], torch.tensor([0]), slice(0, 1)):
            env, term = make_gait()
            advance(env, 13)
            expected = term.phase.clone()
            term.cfg.ranges.frequencies = (2.2, 2.2)
            term.reset(ids)  # Isaac only zeros episode_length_buf after all managers reset.
            torch.testing.assert_close(term.phase, torch.tensor([0., expected[1]]))
            env.episode_length_buf[0] = 0
            term._update_command()
            advance(env)
            torch.testing.assert_close(term.phase, torch.tensor([.044, expected[1] + .024]))

    def test_legacy_remains_time_times_current_frequency(self):
        env, term = make_gait(continuous=False)
        advance(env, 13)
        term.cfg.ranges.frequencies = (2.2, 2.2)
        term._resample_command([0])
        torch.testing.assert_close(term.phase, torch.tensor([13 * .02 * 2.2, 13 * .02 * 1.2]))

    def test_empty_reset_and_full_reset(self):
        env, term = make_gait()
        advance(env, 8)
        before = term.phase.clone()
        self.assertEqual(term.reset([]), {})
        torch.testing.assert_close(term.phase, before)
        term.reset()
        env.episode_length_buf.zero_()
        torch.testing.assert_close(term.phase, torch.zeros(2))

    def test_observations_and_contact_reward_use_the_same_clock(self):
        env, term = make_gait(n=1)
        env.command_manager = NS(get_term=lambda name: term)
        advance(env, 30)  # .72 cycles, left swing / right stance.
        env.episode_length_buf.zero_()  # Deliberately contradict the shared clock.
        mdp_path = _PATH.parent.parent
        obs = next(n for n in ast.parse((mdp_path / "observations.py").read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == "get_gait_phase")
        reward = next(n for n in ast.parse((mdp_path / "rewards.py").read_text()).body
                      if isinstance(n, ast.ClassDef) and n.name == "GaitReward")
        contact_method = next(n for n in reward.body if isinstance(n, ast.FunctionDef) and n.name == "compute_contact_targets")
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  obs, contact_method], type_ignores=[])
        from test_reward_math import reward_math
        namespace = dict(torch=torch, distributions=torch.distributions, gait_contact_targets=reward_math.gait_contact_targets)
        exec(compile(ast.fix_missing_locations(module), "phase_consumers", "exec"), namespace)
        expected = torch.stack((torch.sin(2 * torch.pi * term.phase), torch.cos(2 * torch.pi * term.phase)), dim=1)
        torch.testing.assert_close(namespace["get_gait_phase"](env), expected)
        reward_stub = NS(_env=env, num_envs=1, command_name="gait_command", kappa_gait_probs=.05)
        contacts = namespace["compute_contact_targets"](reward_stub, term.command)
        self.assertLess(contacts[0, 0].item(), .001)
        self.assertGreater(contacts[0, 1].item(), .999)


if __name__ == "__main__":
    unittest.main()
