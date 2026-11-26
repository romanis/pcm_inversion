"""Run keys and spec hashes of representative command lines of both paper scripts are those of bd9942d (before the
spec classes were thinned), so every existing result directory still resolves to the same run directory and its
stored spec hash still matches. The expected values were computed on bd9942d by running this file as a script
(``python -m nfp_harness.tests.test_run_keys``). Stage files that ``--start-theta`` and ``--warm-delta`` read are
written into a fixed-name run directory under ``tmp_path``; the GMY spec hash holds their absolute path, so for those
command lines only the run key is compared."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
from typing import Any

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "nevo_replication" / ".vendor"
STAGE = "{stage}"   # replaced by the path of the fabricated stage file

PETRIN = {
    "petrin_all": ["run", "--stage", "all", "--nu", "16"],
    "petrin_gate_seed": ["run", "--stage", "gate", "--nu", "4", "--seed", "3"],
    "petrin_body_fixed_w": ["run", "--stage", "step1", "--nu", "16", "--robustness", "body_sigma", "--robustness",
                            "fixed_w"],
    "petrin_a_lo_margin": ["run", "--stage", "all", "--nu", "16", "--robustness", "a_lo_04", "--margin", "0.24"],
    "petrin_mpec_aliases": ["run", "--stage", "step1", "--nu", "8", "--outer", "mpec", "--robustness", "sargan",
                            "--robustness", "income_winsor"],
    "petrin_start_theta": ["run", "--stage", "a_lo_band", "--nu", "16", "--start-theta", STAGE, "--band-nash-all",
                           "--start-gate", "/nonexistent/gate"],
    "petrin_status": ["status", "--nu", "16"],
}
GMY = {
    "gmy_all_years": ["run", "--stage", "all", "--nu", "16"],
    "gmy_dev_years": ["run", "--stage", "prefit", "--nu", "4", "--years", "1980", "1991", "1999", "2005", "2015",
                      "2018", "--threads", "64"],
    "gmy_round2_fix": ["run", "--stage", "all", "--nu", "16", "--years", "1991", "1999", "--robustness", "no_age_taste",
                       "--robustness", "fixed_w", "--a-lo", "0.1", "--fix", "sigma_p=3.0", "--cold-draws", "64",
                       "--max-iterations", "5"],
    "gmy_options": ["run", "--stage", "step1", "--nu", "8", "--years", "1991", "1999", "--income-law", "petrin_bounded",
                    "--antithetic-taste", "--income-basis", "auto", "--prefit-rows", "age", "--min-share", "0",
                    "--households", "500", "--instruments", "gh", "--moments", "gmy_nolinear"],
    "gmy_alias": ["run", "--stage", "step2", "--nu", "16", "--years", "1991", "1999", "--moments", "demand"],
    "gmy_start_warm": ["run", "--stage", "all", "--nu", "16", "--years", "1991", "1999", "--warm-delta", STAGE,
                       "--start-theta", STAGE, "--start-theta-except", "sigma_z_log_height"],
    "gmy_in_cost_joint": ["run", "--stage", "all", "--nu", "64", "--years", "1991", "1999", "--households", "500",
                          "--antithetic-taste", "--moments", "demand_supply_msfd_own", "--weighting", "joint_cue",
                          "--drop-block", "own", "--cost-shifters", "in_cost", "--a-lo", "0.05"],
    "gmy_alo_moments": ["run", "--stage", "all", "--nu", "64", "--years", "1991", "1999", "--households", "500",
                        "--antithetic-taste", "--moments", "demand_supply_msfd_own", "--weighting", "joint_cue",
                        "--drop-block", "own", "--cost-shifters", "in_cost", "--a-lo", "0.05", "--alo-moments",
                        "c4c6c"],
    "gmy_horizontal_groups": ["run", "--stage", "all", "--nu", "64", "--years", "1991", "1999", "--households", "500",
                              "--antithetic-taste", "--moments", "demand_supply_msfd_own", "--weighting", "joint_cue",
                              "--drop-block", "own", "--cost-shifters", "in_cost", "--a-lo", "0.05", "--alo-moments",
                              "c4c6c", "--horizontal", "groups"],
    "gmy_horizontal_factor": ["run", "--stage", "all", "--nu", "64", "--years", "1991", "1999", "--households", "500",
                              "--antithetic-taste", "--moments", "demand_supply_msfd_own", "--weighting", "joint_cue",
                              "--drop-block", "own", "--cost-shifters", "in_cost", "--a-lo", "0.05", "--alo-moments",
                              "c4c6c", "--horizontal", "factor", "--factor-k", "2"],
}
#: command lines whose spec hash holds an absolute path of tmp_path: run key only
PATH_IN_HASH = {"gmy_start_warm"}

#: case -> (run directory name, spec hash) on bd9942d, unless noted
EXPECTED: dict[str, tuple[str, str]] = {
    "petrin_all": ("petrin_pcm_937151f021", "2ee5eb99cf49e53951701e940870eefad607083b533f19c5ed488cd9f4b79658"),
    "petrin_gate_seed": ("petrin_pcm_1390293108", "2ee5eb99cf49e53951701e940870eefad607083b533f19c5ed488cd9f4b79658"),
    "petrin_body_fixed_w": ("petrin_pcm_ad26918dff", "6c2f038b15f61a442b9fd0170d55394caa96a191548662309274eb8d820b206a"),
    "petrin_a_lo_margin": ("petrin_pcm_dcd667eea4", "ba8d6293328b1d17b097a4c99615ab264dfbbbacb2e653a72296dd149a9b9b14"),
    "petrin_mpec_aliases": ("petrin_pcm_97dae1da0b", "f523772ee04c267d6c54a9e4ecbac3370d2d70748da4e67f32c69fa5771f44de"),
    "petrin_start_theta": ("petrin_pcm_42d65e6c5d", "18d9acf61b5eb8c4c15f9074c94534e5512e99f4017d2513b5fd30b150a78fa5"),
    "petrin_status": ("petrin_pcm_937151f021", "2ee5eb99cf49e53951701e940870eefad607083b533f19c5ed488cd9f4b79658"),
    "gmy_all_years": ("gmy_pcm_83df55f00a", "a4f11b4e869459adef139e9eaa2f0a4508e733346303b3e78949c4c35286edb7"),
    "gmy_dev_years": ("gmy_pcm_2c4a7ee7fd", "3819ddd87d4c8a4a42642c053edf68925e9a2c4ac4a7a546663c564bce222252"),
    "gmy_round2_fix": ("gmy_pcm_c420a2f1b4", "aa9e1ed16e9fc2c26afe40dd7507377018ebc0bdea5df83e0b5a630bb68622c4"),
    "gmy_options": ("gmy_pcm_d41949d7be", "1e300a1f801d87c72f94493769719d475ab5a0a44b8e72b4076303786f0be3cb"),
    "gmy_alias": ("gmy_pcm_6a24c53b4d", "8e0fe673376949dd8013c333d2a4c29f08aa10a5e6071ce35630f2f6b1fc2bd4"),
    "gmy_start_warm": ("gmy_pcm_c012f0d0ac", "b87186d95e59478354ebde0ed00c17b8d22a530b5459a482ed444b7d74185380"),
    # computed on 7e84731 (before --alo-moments): the option left off keeps this directory and hash
    "gmy_in_cost_joint": ("gmy_pcm_11c22c8446", "b65b622f016c0d572b33a4139d5abb0faf6784f00e06cb84a8302b144633b6bf"),
    # computed when --alo-moments was added
    "gmy_alo_moments": ("gmy_pcm_0686bfbcc7", "90bb7e417b9dbe90262fecade816f385d8e3781c9fa3fbf1e87d6d92943ab39f"),
    # computed when --horizontal was added (off leaves every case above unchanged)
    "gmy_horizontal_groups": ("gmy_pcm_de05afeef7", "6d25777f02626263381578254d778f78621b32895ea38fdea501f0cc8e370e4f"),
    "gmy_horizontal_factor": ("gmy_pcm_0cce6d38e4", "266a2e3e8d3d50c28e03362339a51bdea3bf9e13ee29cb1b05769e440f7e20a9"),
}


def load_script(rel: str, name: str) -> Any:
    for p in (str(VENDOR), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_stage(base: pathlib.Path, run: str, names: list[str], theta: np.ndarray) -> pathlib.Path:
    """``<base>/<run>/stages/step1.npz`` holding ``theta`` and a model state, and the run's history.csv naming it."""
    from nfp_harness.checkpoint import RunDirectory

    spec_name, key = run.rsplit("_", 1)
    rd = RunDirectory(base, spec_name, "0" * 64, key + "0" * (64 - len(key)), options={})
    assert rd.path.name == run
    rd.save_stage("step1", {"theta": np.asarray(theta, dtype=float), "q": 1.0, "W": np.eye(2),
                            "problem_state": {"model": {"market_ids": ["1991"], "delta": [np.zeros(3)]}}})
    (rd.path / "history.csv").write_text("iteration," + ",".join(f"theta_{n}" for n in names) + "\n")
    return rd.path / "stages" / "step1.npz"


def compute(paper: str, case: str, base: pathlib.Path) -> tuple[str, str]:
    """(run directory name, spec hash) of the command line ``case`` of ``paper`` ("petrin" or "gmy")."""
    from nfp_harness import pipeline
    from nfp_harness.checkpoint import hash_options

    if paper == "petrin":
        mod = load_script("petrin_replication/estimate_pcm.py", "runkey_petrin_pcm")
        spec, argv, default_base = mod.PetrinSpec(), list(PETRIN[case]), mod.RESULTS
        run = "petrin_pcm_0stage0001"
    else:
        mod = load_script("gmy_replication/estimate_pcm.py", "runkey_gmy_pcm")
        spec, argv, default_base = mod.GMYSpec((1991, 1999)), list(GMY[case]), mod.HARNESS
        run = "gmy_pcm_0stage0002"
    if STAGE in argv:
        lay = spec.layout()
        theta = np.asarray(lay.starts(), dtype=float) + 0.01 * np.arange(1, lay.P + 1)
        path = write_stage(base, run, list(lay.names), theta)
        argv = [str(path) if a == STAGE else a for a in argv]
    args = pipeline._parser(16, default_base, spec).parse_args(argv)
    spec.configure(args)
    name = pipeline.run_dir_path(base, spec, args.nu, args.seed).name
    return name, hash_options(spec.hash_payload())


def _needs(paper: str) -> None:
    for p in (str(VENDOR), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if importlib.util.find_spec("pyblp") is None:
        pytest.skip("pyblp is not importable")
    if paper == "gmy" and not (ROOT / "gmy_replication" / "data" / "GMY_Replication" / "data" / "data_blp.csv").exists():
        pytest.skip("GMY replication package not unpacked")
    if paper == "gmy" and not (ROOT / "gmy_replication" / "results" / "blp_fit_h1000.json").exists():
        pytest.skip("GMY BLP fit missing")
    pytest.importorskip("nfp_harness._native")


CASES = [("petrin", c) for c in PETRIN] + [("gmy", c) for c in GMY]


@pytest.mark.parametrize(("paper", "case"), CASES, ids=[c for _, c in CASES])
def test_run_key_and_spec_hash_unchanged(paper: str, case: str, tmp_path: pathlib.Path) -> None:
    _needs(paper)
    assert case in EXPECTED, f"no expected value for {case}"
    name, spec_hash = compute(paper, case, tmp_path)
    want_name, want_hash = EXPECTED[case]
    assert name == want_name, f"{case}: run directory {name}, bd9942d resolved {want_name}"
    if case not in PATH_IN_HASH:
        assert spec_hash == want_hash, f"{case}: spec hash {spec_hash}, bd9942d {want_hash}"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        for paper, case in CASES:
            print(f"    {case!r}: {compute(paper, case, pathlib.Path(d))!r},")
