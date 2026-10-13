"""Gate S10, the clean-room audit. The first test in tier two.

**The entire contribution of this project is that no human priors exist anywhere in the system.** Every other
claim in the design has a gate; this one had a sentence. A reviewer will ask how it is known, and a passing
gate is a better answer than a paragraph.

What it asserts:

    no pretrained weights are loaded from anywhere
    no external data file is read at initialisation or during a run, other than the config
    initialisation draws only from an explicitly seeded generator, so nothing enters except the seed
    the declared dependency set ships no model weights or corpora

### Behavioural rather than declarative, where that was achievable

A deny-list of packages known to bundle weights is the easy version and it is only as good as the list, which
goes stale. The behavioural version tests the thing itself: run with outbound network disabled and observe
every file the process opens. It stays true as dependencies change.

Two instruments, and **each was verified to respond before being trusted**, because a sandbox that passes
because it was not restricting anything is a construction-class failure and would be invisible (D-022):

    network      unshare --user --map-root-user --net. Measured: a control connection succeeds outside the
                 namespace and raises OSError inside it, so the block is real rather than assumed
    filesystem   sys.addaudithook on the `open` and `os.open` events, recording every path opened during the
                 run. Measured: it catches a deliberate read of a file outside the repository

**The audit hook is the right instrument with a known blind spot, not a fallback for a sandbox.** That
distinction matters and an earlier version of this docstring got it wrong.

The claim being tested is that no external data file is read. An audit hook observes every open and asserts
none is outside the repository, which tests that claim directly. A filesystem sandbox tests something
adjacent and weaker: that the code still works when it cannot read outside, which code that never tries would
pass either way.

Where a sandbox genuinely helps is coverage, not strength. The hook sees opens routed through CPython, so a
native extension issuing a raw syscall would bypass it. That is the real limit, it is the reason to want
`bubblewrap`, and it is recorded in Q-015 as an untaken improvement rather than as a compromise already made.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# Packages that ship model weights, pretrained checkpoints, or text corpora. The weak half of this gate: it is
# only as good as this list and the list will go stale, which is stated rather than implied. The behavioural
# tests below are what actually carry the claim.
WEIGHT_SHIPPING = {
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tensorflow-hub",
    "keras",
    "transformers",
    "diffusers",
    "timm",
    "sentence-transformers",
    "spacy",
    "nltk",
    "gensim",
    "fasttext",
    "flair",
    "allennlp",
    "openai",
    "anthropic",
    "huggingface-hub",
    "datasets",
    "tokenizers",
    "scikit-learn",
}

NET_SANDBOX = ["unshare", "--user", "--map-root-user", "--net"]


def _installed():
    import importlib.metadata as md

    return {d.metadata["Name"].lower() for d in md.distributions() if d.metadata["Name"]}


def _offenders(denylist=None):
    """The one place the deny-list intersection happens, used by the gate and by its own power check.

    Factored deliberately. The first version computed the intersection inline in the gate and again in
    the power check, so replacing `& WEIGHT_SHIPPING` with `& set()` in the gate left the power check
    validating a constant the gate no longer consulted. Falsification caught it: the break passed.

    One implementation means a break to the mechanism breaks both, which is the structural fix rather
    than an assertion that the two agree.
    """
    return _installed() & (WEIGHT_SHIPPING if denylist is None else denylist)


# --------------------------------------------------------------------------- declarative


def test_no_dependency_ships_weights_or_corpora():
    """The deny-list half. Weak by construction, and the weakness is the docstring rather than a footnote."""
    offenders = _offenders()
    assert not offenders, (
        f"the environment contains {sorted(offenders)}, which ship model weights, checkpoints or corpora. "
        "The clean-room claim is that no human-derived prior exists anywhere in the system."
    )


def test_the_deny_list_would_catch_something():
    """Power check on the list itself. A deny-list that matches nothing installable asserts nothing.

    Confirms the comparison is against real distribution names rather than, say, import names that never
    appear in the metadata, which would make the test vacuous while looking thorough.
    """
    installed = _installed()
    assert installed, "no distributions found, so the deny-list comparison is against an empty set"
    assert "jax" in installed, (
        "distribution names are not in the expected form, so the match would never fire"
    )

    # The list itself must be non-empty and must contain names that are real distributions rather than import
    # names. An emptied or misspelled list passes the check above while matching nothing, which is the
    # polarity failure in another guise: "no offender found" and "no offender findable" look identical.
    assert len(WEIGHT_SHIPPING) >= 20, f"the deny-list has shrunk to {len(WEIGHT_SHIPPING)} entries"
    for canary in ("torch", "transformers", "tensorflow"):
        assert canary in WEIGHT_SHIPPING, f"{canary} is missing from the deny-list"

    # The mechanism itself, exercised through the same function the gate calls. Passing a list holding a
    # package that IS installed must produce a detection. This is what makes the gate's negative result
    # mean "nothing matched" rather than "nothing could match": the first version checked the constant
    # and the gate separately, and a break bypassing the constant passed both.
    assert _offenders({"jax"}) == {"jax"}, (
        "the deny-list mechanism did not detect an installed package that is on the list it was given, "
        "so a clean result from it means nothing"
    )


def test_no_weight_file_extensions_anywhere_in_the_repository():
    """No checkpoint has been committed, deliberately or by accident."""
    suspicious = {".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".pb", ".onnx", ".bin", ".npz", ".pkl"}
    skip = {".git", ".venv", "__pycache__", ".pytest_cache", "runs"}

    # Prune rather than filter. The first version used `REPO.rglob("*")` with a `parts` check, which walks
    # every file inside `.venv` before discarding it: that is the whole CUDA stack, tens of thousands of
    # files and several gigabytes, and it took longer than the rest of the suite combined. Pruning at the
    # directory level never descends into them.
    found = []
    stack = [REPO]
    while stack:
        d = stack.pop()
        for entry in d.iterdir():
            if entry.is_dir():
                if entry.name not in skip:
                    stack.append(entry)
            elif entry.suffix in suspicious:
                found.append(str(entry.relative_to(REPO)))

    assert not found, f"weight-shaped files in the repository: {found}"


# --------------------------------------------------------------------------- behavioural


def _run_isolated(script: str, sandbox: bool):
    """Run a script in a subprocess, optionally inside a network namespace."""
    cmd = (NET_SANDBOX if sandbox else []) + [sys.executable, "-c", script]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=REPO)


def test_the_network_sandbox_actually_blocks():
    """D-022 applied to the sandbox. Verify the instrument responds before trusting a run inside it.

    Without this, a passing isolation test would be indistinguishable from a sandbox that silently did
    nothing, which is the construction-class failure: the check and the broken check look identical.
    """
    probe = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=4).close()\n"
        "    print('REACHABLE')\n"
        "except Exception as e:\n"
        "    print('BLOCKED')\n"
    )
    outside = _run_isolated(probe, sandbox=False)
    if "REACHABLE" not in outside.stdout:
        pytest.skip("no network available outside the sandbox, so a block cannot be distinguished from it")

    inside = _run_isolated(probe, sandbox=True)
    assert "BLOCKED" in inside.stdout, (
        "the network namespace did not block an outbound connection, so any isolation result from it is "
        f"meaningless. stdout={inside.stdout!r} stderr={inside.stderr[-400:]!r}"
    )


def test_a_run_completes_with_outbound_network_disabled():
    """The behavioural claim: nothing fetches weights or data at runtime, because there is nothing to fetch with.

    A short episode over everything that exists at build step 4: obstacle layout, source placement, seeding,
    regrowth, pick resolution, the three pools, and death recording.
    """
    script = (
        "import jax, jax.numpy as jnp, numpy as np\n"
        "from configs.default import DEFAULT as P\n"
        "from src.env.grid import make_obstacle_grid\n"
        "from src.env.resources import place_sources, seed_grid, regrow, resolve_pick\n"
        "from src.env.entropy import PRNGSource\n"
        "from src.env.dynamics import update_energy, update_health, resolve_deaths\n"
        "obs = make_obstacle_grid(jax.random.key(0), 20, 20, P.obstacle_density)\n"
        "fs, ts, _ = place_sources(jax.random.key(1), obs, P)\n"
        "food = seed_grid(jax.random.key(2), fs, obs, P.cap_food, P.r_src)\n"
        "src_ = PRNGSource(key=jax.random.key(3))\n"
        "energy = jnp.full((4,), P.e_max); health = jnp.full((4,), P.h_max)\n"
        "alive = jnp.ones((4,), bool); cause = jnp.zeros((4,), jnp.int32)\n"
        "dstep = jnp.full((4,), -1, jnp.int32)\n"
        "starved = jnp.zeros((4,), bool)\n"
        "predated = jnp.zeros((4,), bool)\n"
        "for t in range(15):\n"
        "    food, _ = regrow(src_, food, fs, obs, P.cap_food, P.r_src, P.p_food, t,\n"
        "                     'regrow.food.trigger', 'regrow.food.cell')\n"
        "    energy = update_energy(energy, jnp.zeros((4,), bool), jnp.zeros((4,), bool), P)\n"
        "    health = update_health(health, jnp.zeros((4,), jnp.float32), energy, P)\n"
        "    alive, cause, dstep, starved, predated = resolve_deaths(\n"
        "        alive, health, energy, jnp.zeros((4,), bool), cause, dstep, starved, predated, t)\n"
        "print('EPISODE_OK', int(jnp.sum(food)), int(jnp.sum(alive)))\n"
    )
    # **This call site lives inside a STRING, so a grep for `resolve_deaths(` across the tests did not see
    # it when that signature changed.** It surfaced only when the subprocess raised a TypeError, which S10
    # reports as "a short episode did not complete" rather than as a signature error, two steps from the
    # cause. A source-level search cannot find code held as data, and this file holds a whole script that way.
    result = _run_isolated(script, sandbox=True)
    assert "EPISODE_OK" in result.stdout, (
        "a short episode did not complete with outbound network disabled. Something in the pipeline reaches "
        f"the network at import or at run time. stderr={result.stderr[-800:]!r}"
    )


def test_the_audit_hook_responds():
    """D-022 again, for the filesystem instrument. Confirm it catches a read it is meant to catch.

    Its limitation, stated rather than implied: the hook observes rather than prevents, and it sees opens
    routed through CPython. A C extension calling libc `open` directly is invisible to it. `bwrap` and
    `firejail` are absent on this machine, so a hard jail was not available. See Q-015.
    """
    script = (
        "import sys\n"
        "seen = []\n"
        "def hook(event, args):\n"
        "    if event in ('open', 'os.open') and args:\n"
        "        seen.append(str(args[0]))\n"
        "sys.addaudithook(hook)\n"
        "open('/etc/hostname').read()\n"
        "print('CAUGHT' if any('/etc/hostname' in p for p in seen) else 'MISSED')\n"
    )
    result = _run_isolated(script, sandbox=False)
    assert "CAUGHT" in result.stdout, (
        f"the audit hook did not observe a deliberate read outside the repository, so it cannot support any "
        f"claim about what a run opens. stdout={result.stdout!r}"
    )


def test_a_run_opens_no_data_file_outside_the_repository_or_the_environment():
    """The observational half: record every path opened during a run and classify it.

    Legitimate opens are the interpreter, the installed environment, and the repository's own source and
    config. What must not appear is a data file from anywhere else, which is what loading a prior would be.
    """
    script = (
        "import sys\n"
        "seen = []\n"
        "def hook(event, args):\n"
        "    if event in ('open', 'os.open') and args:\n"
        "        seen.append(str(args[0]))\n"
        "sys.addaudithook(hook)\n"
        "import jax, jax.numpy as jnp\n"
        "from configs.default import DEFAULT as P\n"
        "from src.env.grid import make_obstacle_grid\n"
        "obs = make_obstacle_grid(jax.random.key(0), 16, 16, P.obstacle_density)\n"
        "obs.block_until_ready()\n"
        "import json; print('PATHS', json.dumps(seen))\n"
    )
    result = _run_isolated(script, sandbox=False)
    assert "PATHS" in result.stdout, f"probe failed: {result.stderr[-600:]!r}"

    import json

    paths = json.loads(result.stdout.split("PATHS", 1)[1].strip())
    repo, venv = str(REPO), str(REPO / ".venv")
    # `/tmp` is deliberately NOT allowed. It was, and falsification caught it: a run that downloaded a
    # prior would land it in `/tmp`, so allowing that prefix made the assertion unable to see the exact
    # failure it exists for. The break, opening a `.csv` in `/tmp` during the run, passed. Scratch files
    # JAX writes there are not data-shaped and are excluded by the suffix filter instead.
    allowed_prefixes = (
        repo,
        venv,
        sys.prefix,
        sys.base_prefix,
        "/usr/lib",
        "/usr/share",
        "/proc",
        "/sys",
        "/dev",
        "/etc/ld.so",
    )
    DATA_SUFFIXES = {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
        ".h5",
        ".pb",
        ".onnx",
        ".csv",
        ".tsv",
        ".parquet",
        ".txt",
        ".vocab",
    }

    foreign_data = [
        p for p in paths if not p.startswith(allowed_prefixes) and os.path.splitext(p)[1] in DATA_SUFFIXES
    ]
    assert not foreign_data, (
        f"the run opened data files outside the repository and environment: {foreign_data[:8]}"
    )

    assert len(paths) > 0, "the audit hook recorded nothing, so this assertion is vacuous"


def test_construction_is_reproducible_from_the_seed_alone():
    """The operational form of "nothing enters except the seed".

    If any human-derived prior leaked into initialisation, two constructions from the same seed in separate
    processes could differ. Run in subprocesses rather than in-process so a shared module-level cache cannot
    manufacture the agreement.
    """
    script = (
        "import jax\n"
        "from configs.default import DEFAULT as P\n"
        "from src.env.grid import make_obstacle_grid\n"
        "import numpy as np, hashlib\n"
        "g = np.array(make_obstacle_grid(jax.random.key(12345), 24, 24, P.obstacle_density))\n"
        "print('HASH', hashlib.sha256(g.tobytes()).hexdigest()[:32])\n"
    )
    a = _run_isolated(script, sandbox=False)
    b = _run_isolated(script, sandbox=False)
    ha = a.stdout.split("HASH", 1)[1].strip()
    hb = b.stdout.split("HASH", 1)[1].strip()
    assert ha == hb, f"two constructions from one seed differ: {ha} against {hb}"

    different = script.replace("12345", "54321")
    c = _run_isolated(different, sandbox=False)
    hc = c.stdout.split("HASH", 1)[1].strip()
    assert hc != ha, (
        "a different seed produced an identical layout, so the hash is not sensitive to the seed and the "
        "reproducibility assertion above is vacuous"
    )
