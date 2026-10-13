# Emergent self-preservation: the sweep on the RTX A6000

A clean-room test of whether self-preservation emerges from selection alone: populations of small recurrent
networks evolve in a grid world, rewarded only for collecting forage targets and never for staying alive.
This repository holds exactly the code needed to check a machine, run the two baseline conditions, and run
the characterisation sweep, plus the commands to do it.

**Nobody needs to change any file to run it, and nobody should.** Every command is below. The run scripts
refuse to start if a shipped file has been edited. If something goes wrong, the run either recovers by
itself or stops with a message saying what to do, and the table at the end lists every message.

    source commit     88581992dc77f39a05770ef37c3a04f8962b267a
    verified commit   e6b93b782bcb9fe8df6c920842ccb00d1a665f19   the run code is byte-identical to this commit, which passed
                                           the full differential test tier

## At a glance

| Step | Command | Time on an RTX A6000 |
|---|---|---|
| 1. Set up | install uv, clone, open in VS Code, start tmux | 10 minutes |
| 2. Check the machine | `bash scripts/a6000/check.sh` | 45 to 75 minutes |
| 3. Baselines | `bash scripts/a6000/run.sh baselines` | 1 to 2 hours |
| 4. The sweep | `bash scripts/a6000/run.sh sweep` | about 2 to 3 days, estimated |
| 5. Send back | the folders `runs/sweep` and `results` | about 80 GB |

The sweep's time is an estimate until step 2 measures this machine. The run prints its own estimate from
the speed it actually achieves.

## 1. Set up

**What the machine needs.**

- Linux with an NVIDIA driver installed. `nvidia-smi` must print the RTX A6000. On Windows, use WSL2 and keep
  this folder inside the Linux home folder, never under `/mnt/c`.
- At least 100 GB free disk where the repository lives.
- tmux. Check with `tmux -V`. If it is missing, `sudo apt install tmux`, or ask the machine's administrator.
- Internet access the first time, to download the Python environment, about 3 GB.
- Nothing else using the GPU while the run is going, and automatic sleep or suspend switched off.

**Install uv**, the tool that builds the Python environment. Once per machine:

    curl -LsSf https://astral.sh/uv/install.sh | sh

Close and reopen the terminal afterwards so `uv` is found.

**Get the code.**

    cd ~
    git clone <the repository URL you were given> g-one-a6000
    cd g-one-a6000

**Open it in VS Code.** File, Open Folder, choose `g-one-a6000`. If the machine is remote, connect with the
Remote-SSH extension first, then open the folder on the remote machine. Open a terminal with Terminal, New
Terminal.

**Start tmux in that terminal. This matters more than anything else on this page.** A program started
directly in a VS Code terminal dies when the window closes or the remote connection drops. A program started
inside tmux keeps running.

    tmux new -s gone

Everything below is typed inside that tmux session. To leave it running and step away, press `Ctrl+b`, let
go, then press `d`. To come back later, from any terminal on that machine:

    tmux attach -t gone

To scroll back inside tmux, press `Ctrl+b` then `[`, use the arrow keys, and press `q` to stop scrolling.

If tmux cannot be installed, the fallback for steps 3 and 4 is `nohup`, shown in section 4.

## 2. Check the machine

    bash scripts/a6000/check.sh

In order, it:

- confirms the GPU, the disk space and that every file arrived intact;
- builds the environment from the lock file, so every library version matches the one the code was
  verified with;
- proves JAX is using the GPU and not the CPU;
- runs the engineering, differential and scientific tests;
- runs the whole sweep path at toy size twice, to prove that stopping and resuming works;
- measures this machine's speed for about 20 minutes.

It ends with one of two lines:

    MACHINE CHECK PASSED. results/MACHINE_VERIFIED written.
    MACHINE CHECK FAILED at: <the step>

If it fails, stop and send the folder `results/check-<time>` to the project team. Do not try to fix it.

**What is and is not tested here.** Every test file of the research repository is included and run.
A small number of individual tests, named at the top of `scripts/a6000/check.sh`, are deselected,
because they check the full research repository's own record files (its run registry, its list of
unswept parameters, its map of modules, its git history), which are not shipped with this code. They pass in
the full repository and concern no part of the code or the hardware. Gates S5 and S6 need the null baseline,
which does not exist yet, so the check skips them and step 3 runs them.

To run the tests by hand, with the same deselections:

    .venv/bin/python -m pytest tests/engineering/test_e0_setup.py -q
    .venv/bin/python -m pytest tests/differential -q

and the full deselect lists are at the top of `scripts/a6000/check.sh`.

## 3. Baselines

    bash scripts/a6000/run.sh baselines

This evolves the two baseline conditions, ten seeds each: one with the predator and one without. Then it
runs gate S5, which must pass, and gate S6, which is recorded. **S6 is expected to report FAILED at ten
seeds.** That is a known, deliberate result under the gate's conservative assumption, and the seed count is
the project owner's decision, so the run does not stop for it. The paired power analysis runs last.

When it ends with `baselines done`, send these three files to the project team and wait for the go-ahead
before step 4, unless you were told to run everything in one go:

    results/s5.log
    results/s6.log
    results/pairing_power.log

## 4. The sweep

    bash scripts/a6000/run.sh sweep

109 conditions, 10 seeds each, 300 generations each. To run steps 3 and 4 back to back without stopping:

    bash scripts/a6000/run.sh all

Without tmux, start it with `nohup` instead and follow the console log:

    ALLOW_NO_TMUX=1 nohup bash scripts/a6000/run.sh sweep > results/console.log 2>&1 &
    tail -f results/console.log

**After a crash, a reboot or a power cut, run the same command again.** Finished seeds and conditions are
never repeated, and at most the seed that was in progress is redone. If the machine rebooted, start tmux
again first.

## Watching it

The terminal shows four kinds of line. The numbers here are only examples.

    [  14/109] treatment__p_wander__0.5 starting
        gen  26/300   0.412 s  fit 0.0481 best 0.2120 zero 31.2%  predated 0.004 starved 0.957 survived 0.039
      seed 3/10 done in 2:08  final 0.0612 best 0.2441  0.411 s/gen  ram 2.10 GB  gpu mem 310 MB  gpu 71C 1755MHz 280W util 97%
    [  14/109] treatment__p_wander__0.5   20.6 min  fit 0.058 +/- 0.004  ram 2104MB  done  12.8%  elapsed 4:51:10  remaining ~1d 09:12

- **Condition lines**, starting `[  14/109]`, appear when a condition starts and when it finishes. The finishing
  line has its mean fitness, how much of the whole sweep is done, and the estimated time remaining.
- **Generation lines**, starting `gen`, appear every 25 generations of a seed. They show the time per
  generation, the mean and best fitness, the share of networks scoring zero, and the share of prey that were
  caught, starved or survived.
- **Seed lines**, starting `seed`, appear when each of the ten seeds of a condition finishes. They show its
  time, memory, and the GPU's temperature, clock, power and load.
- **The live line**, starting `[cond 14/109 | seed 3/10 | gen 187/300]`, updates in place every half second with
  the time left for the seed and the sweep.

Lines starting `WARNING` name anything out of place. They are explained in the table below.

From a second terminal, or after detaching from tmux:

    tail -f runs/sweep/sweep.log          everything the run prints, with times
    cat runs/sweep/status.json            where it is, how much is done, time left, recent warnings
    cat runs/sweep/warnings.log           only the warnings
    cat runs/sweep/run.log                restarts and stops by the run script
    watch -n 5 nvidia-smi                 the GPU, live

`status.json` is rewritten at least once a minute. If its `updated` time is more than an hour old, the run
is not progressing; the run script notices that by itself after 45 minutes and restarts it.

To make the output quieter or louder, the sweep takes `--progress quiet`, `normal` (the default) or
`verbose`. The run script uses the default.

## Warnings

A warning never stops the run by itself unless the table says so.

| Warning | What it means | What happens | What a person should do |
|---|---|---|---|
| `RECOMPILE` | JAX rebuilt a program in the middle of a seed. In an earlier version, the same fault leaked about 31 MB per generation | Continues | Watch for MEMORY warnings after it, and send the log |
| `MEMORY-SEED` | Host memory grew more than 400 MB during one seed | Continues | If it repeats, send the log |
| `MEMORY-RUN` | Host memory is more than 2 GB above its level after the first condition | Continues; restarts itself if memory runs out | Send the log |
| `GPU-MEMORY` | GPU memory in use grew more than 1 GB | Continues | If it keeps growing, send the log |
| `SLOW-GENERATION` | One generation took over five times its seed's usual time | Continues | Nothing, unless it repeats |
| `SLOWDOWN` | A whole seed ran 40 percent slower than the first seed of its condition | Continues | Check the GPU temperature and that nothing else is using the GPU |
| `GPU-HOT` | GPU at 87 C or more | Continues, more slowly | Check fans and airflow around the machine |
| `GPU-THROTTLE` | The GPU reports it is slowing itself for heat or power | Continues, more slowly | As above |
| `GPU-SHARED` | Another program is using the GPU | Continues, more slowly | Stop the other program |
| `GPU-UNREADABLE` | nvidia-smi gave no reading | Continues | Nothing; heat cannot be watched |
| `ZERO-POPULATION` | No network scored anything for 100 generations | Continues; a harsh condition can do this | Nothing |
| `DISK-LOW` | Under 25 GB free | Continues; stops cleanly at 5 GB | Free disk space now |
| `HOST-MEMORY` | Under 6 GB of memory available | Continues; restarts itself under 2 GB | Close other programs |
| `STALL` | No generation has finished for 10 minutes | Restarts itself at 30 minutes | Nothing, unless it happens repeatedly |

## If it stops

The run script restarts the sweep by itself after anything a restart can fix, up to eight times in a row.
It stops for good only for the reasons below. The last lines of `runs/sweep/sweep.log` and
`runs/sweep/run.log` say which.

| Exit | Meaning | What to do |
|---|---|---|
| 0 | Finished | Send the results, below |
| 1, 4, 5 | A crash, a hang, or memory exhausted. Restarted by itself | Nothing. After eight in a row it stops: send the logs |
| 2 | A check failed that a restart cannot fix: a changed file, a failed gate, a bad checkpoint | Do not retry. Send `runs/sweep/sweep.log`, `runs/sweep/run.log` and `results/` |
| 3 | The disk is nearly full. The run stopped before anything could be damaged | Free space, then run the same command again |
| 130 | Stopped with Ctrl-C | Run the same command again to continue |

Things not to do while it runs:

- Do not edit, move or delete any file in the repository or in `runs/sweep`.
- Do not start a second copy of the run. It refuses, and the first copy carries on.
- Do not run anything else on the GPU.

## When it finishes

The run ends with `SWEEP COMPLETE` and writes checksums of every result file. Send back:

    runs/sweep/        109 checkpoints, manifest.jsonl, sweep.log, warnings.log, run.log, status.json
    results/           the machine check, the baseline gates, the power analysis, sweep-checksums.sha256

At about 80 GB this goes on an external drive, not by email. Before deleting anything on this machine,
confirm the copy arrived intact by running, inside the copied `sweep` folder:

    sha256sum -c ../results/sweep-checksums.sha256
