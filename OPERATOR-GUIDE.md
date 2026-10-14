# Operator guide: running the sweep on the RTX A6000

Every command you will type, in order. Under each one: **why** it is there, **what you should see**, and
**whether you may stop it** part way. You never need to edit a file, and you should not: the run refuses to
start if a shipped file has changed. `README.md` holds the full tables of warnings and exit codes; this guide
is the walkthrough.

```
1. Before you start     what the machine needs                         10 minutes
2. Set up               install uv, clone, VS Code, tmux               10 minutes
3. Check the machine    bash scripts/a6000/check.sh                    45 to 75 minutes
4. Baselines            bash scripts/a6000/run.sh baselines            1 to 2 hours
5. The sweep            bash scripts/a6000/run.sh sweep                about 2 to 3 days, estimated
6. Send back            runs/sweep and results                         about 80 GB
```

The three run steps are chained by marker files in `results/`. The check writes `MACHINE_VERIFIED`, the
baselines need it and write `BASELINES_DONE`, and the sweep needs that one. A step can't be skipped by
mistake.

---

## 1. Before you start

### What the machine needs

| Need | How to check | Why |
|---|---|---|
| Linux, or Windows with WSL2 | `uname -a` | The code and scripts are Linux only |
| The NVIDIA driver, showing the A6000 | `nvidia-smi` | JAX talks to the GPU only through the driver |
| Driver supporting CUDA 12.9 | `nvidia-smi`, top right, `CUDA Version: 12.9` or higher | The environment ships CUDA 12.9 libraries. An older driver may still work; the check decides, at gate E0 |
| 100 GB free where you clone | `df -h ~` | About 80 GB of checkpoints, plus margin. The check refuses below 100 GB |
| About 5 GB free in your home folder | `df -h ~` | The download cache, about 3 GB, and JAX's compile cache, capped at 2 GB, live in `~/.cache` |
| No disk quota below 100 GB | `quota -s`, if the command exists | `df` cannot see a per-user quota. A quota hit mid-run crashes every restart. The check prints your quota |
| At least 8 GB of free RAM | `free -g`, the `available` column | The sweep refuses to start under 4 GB available and warns under 6 |
| tmux | `tmux -V` | Keeps the run alive when the terminal or SSH connection closes |
| Internet, at setup only | | The clone and the environment download. The run itself needs no network |
| An ordinary user account, not root | `whoami` | Run as root, every file ends up owned by root. Never use `sudo` for any command in this guide |

**On Windows with WSL2, also:** keep the clone in the Linux home folder (`~`), never under `/mnt/c`. The
check refuses `/mnt/c`, because small-file writes there are hundreds of times slower. Set Windows to never
sleep while plugged in, and pause Windows Update for the week, because an automatic restart stops the run.
After such a restart, see "If something happens" below.

### Packages, `requirements.txt`, and the virtual environment

**There is no `requirements.txt`, deliberately. `pyproject.toml` and `uv.lock` do that job, more strictly.**

- `pyproject.toml` lists the direct dependencies at exact versions.
- `uv.lock` pins every package, about 120 including all indirect ones and the CUDA libraries, to an exact
  version with a SHA-256 hash.
- The check installs it with `uv sync --frozen`:
  - "frozen" means uv may not change a single version;
  - a file whose hash does not match is refused;
  - uv downloads Python 3.12 itself if the machine lacks it.

A `requirements.txt` with `pip` would pin neither the Python version nor the hashes unless written very
carefully, and the run scripts would not use it anyway.

The key versions, for reference only. Nothing here needs installing by hand:

```
python 3.12          jax, jaxlib, jax-cuda12-plugin, jax-cuda12-pjrt 0.11.0
numpy 2.5.1          CUDA runtime 12.9.79, cuDNN 9.24.0.43, NCCL 2.30.7 (pip packages)
scipy 1.18.0         pytest 9.1.1
```

**You do not install CUDA.** The CUDA libraries come as Python packages inside the environment; only the
driver comes from the system.

**The virtual environment is `.venv/` inside the clone**, created by the check. The scripts call
`.venv/bin/python` by its path, so:

- you never need to activate it;
- an active conda or other environment in your terminal does no harm;
- **do not `pip install` anything into `.venv`.** If something was installed by mistake, run the check
  again: `uv sync` removes anything not in the lock file.

### Environment variables

| Variable | What happens to it | Why |
|---|---|---|
| Any `JAX_...` or `XLA_...` | **Removed** at the start of the check and of every run, and each removal is printed | These can silently move the run to the CPU, switch the arithmetic to 64-bit, change matrix precision or change the compiler. Profiles on shared machines sometimes set them |
| `PYTHONPATH`, `PYTHONHOME` | **Removed**, printed | Either could make Python load other code than the shipped code |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | **Set to `false`** by the code itself | JAX would otherwise grab 75 percent of GPU memory up front, and the memory readings would mean nothing |
| `JAX_COMPILATION_CACHE_DIR` | **Set** by the code to `~/.cache/jax-emergent-self-preservation` | Compiled programs are reused across restarts, so a restart costs seconds rather than minutes |
| `CUDA_VISIBLE_DEVICES`, `LD_LIBRARY_PATH` | **Left alone**, printed | On some machines the driver is only found through these. Whether JAX really has the GPU is tested directly before every run |
| `HTTPS_PROXY` | Set it yourself **only** if your network needs a proxy for downloads | `git clone` and `uv` read it |
| `ALLOW_NO_TMUX=1` | Set it yourself **only** for the `nohup` fallback in step 5 | Skips the 20-second "not inside tmux" warning |

You do not need to set anything else, and nothing needs exporting in `~/.bashrc`.

---

## 2. Set up

### 2.1 Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Why.** uv builds the Python environment from the lock file, exactly. Once per machine, no `sudo`.

**Expected.** A few lines ending with a message that uv is installed into `~/.local/bin`. Then **close the
terminal and open a new one**, so the new folder is on your PATH, and confirm:

```bash
uv --version
```

which prints one line, `uv` followed by a version number. `uv: command not found` means the terminal was not
reopened.

**May I stop it?** Yes, at any point. Run it again.

### 2.2 Get the code

```bash
cd ~
git clone https://github.com/Dxv-404/g-one-a6000.git g-one-a6000
cd g-one-a6000
```

**Why.** The repository is the code exactly as verified. `~` keeps it on the Linux filesystem.

**Expected.** The repository is private, so git asks for a username and a password.
- Give your GitHub username.
- For the password, paste a **personal access token**, not your GitHub password, which GitHub no longer
  accepts here. Make one under GitHub, Settings, Developer settings, Personal access tokens. If `gh` is
  installed, `gh auth login` once does the same job.
- Then `Cloning into 'g-one-a6000'...`, a few `Receiving objects` lines, and the prompt returns.
- `Repository not found` means your account has not been given access yet.

**May I stop it?** Yes. Delete the half-made folder with `rm -rf ~/g-one-a6000` before trying again, since
git refuses to clone into a folder that is not empty.

### 2.3 Open it in VS Code

File, Open Folder, choose `g-one-a6000`. If the machine is remote, first connect with the **Remote-SSH**
extension, then open the folder on the remote machine. Open a terminal with Terminal, New Terminal.

**Why.** Only for convenience. Everything below works the same in any terminal.

### 2.4 Start tmux

```bash
tmux new -s gone
```

**Why. This matters more than any other line in this guide.** A program started in a plain VS Code terminal
dies when the window closes, the laptop you connect from sleeps, or the SSH connection drops. A program
inside tmux keeps running on the machine.

**Expected.** The screen clears and a coloured bar appears along the bottom showing `[gone]`. Type every
remaining command inside this session.

| To do this | Press or type |
|---|---|
| Leave it running and step away | `Ctrl+b`, let go, then `d`. You see `[detached (from session gone)]` |
| Come back, from any terminal on the machine | `tmux attach -t gone` |
| Scroll back | `Ctrl+b`, then `[`, then the arrow keys or Page Up; `q` to stop |
| You see `duplicate session: gone` | The session already exists: `tmux attach -t gone` |

**May I stop it?** Never type `exit` or close tmux while a run is going: that ends the run. Detaching with
`Ctrl+b d` is always safe.

---

## 3. Check the machine

```bash
bash scripts/a6000/check.sh
```

**Why.** Before three days of GPU time, this proves the machine produces correct results with this code.

**What it runs, in order:**
1. Records the machine and the environment, and confirms the folder is not under `/mnt/c`.
2. Checks every shipped file against its checksum.
3. Checks there is at least 100 GB of disk.
4. Builds `.venv` from the lock file. The first time this downloads about 3 GB.
5. Gate E0: JAX really uses the GPU.
6. The engineering, differential and scientific test tiers.
7. A toy-size sweep, run twice, which proves that stopping and resuming works.
8. About 20 minutes measuring this machine's speed.

**Expected.** First the environment lines, then one block per step:

```
environment: no JAX, XLA or Python path variable was inherited
environment: CUDA_VISIBLE_DEVICES=<unset>  LD_LIBRARY_PATH=<unset>
23:02:28  machine check for the A6000 run, commit e6b93b7.... Results in results/check-20261013-230228

23:02:28  === environment from the lock file (first time downloads about 3 GB)
23:02:28      full output: results/check-20261013-230228/04-uv-sync.log
       + scipy==1.18.0
```

Each step can sit silently for many minutes, because its output goes to the log file named in its block.
To watch one live, detach and run `tail -f` on that file. On the reference laptop the long steps took:

```
engineering tests     345 passed, 13 deselected      about 12 minutes
differential tests    96 passed                      about 11 minutes
scientific tests      43 passed, 18 deselected, 1 xpassed     about 1 minute
```

**On the A6000 these should be faster, and the counts must be the same.** "Deselected" is expected: those
tests check record files that live only in the research repository, and they are listed at the top of
`check.sh`. "xpassed" is also expected; it is a recorded test of a proposed threshold. The check ends
with a summary and one of two lines:

```
  PASS    698s  engineering tests
  ...
MACHINE CHECK PASSED. results/MACHINE_VERIFIED written.
Next:  bash scripts/a6000/run.sh baselines
```

```
MACHINE CHECK FAILED at: <the step>
Do not start the run. Send the folder results/check-<time> to the project team.
```

A line marked `NOTE` is normal; it belongs to the speed measurements, which never fail the check.

**If it fails:** do not try to fix anything and do not run it again hoping for a pass. Send the
`results/check-<time>` folder. Sending a whole failed check costs nothing; a guessed fix could cost the run.

**May I stop it?** Yes, with `Ctrl+C`, at any point, and nothing is damaged. It prints `CHECK STOPPED by
Ctrl-C. Nothing was certified.` and the next attempt starts from the beginning, so you lose the time already
spent. Avoid stopping it during the test steps unless you must.

---

## 4. Baselines

```bash
bash scripts/a6000/run.sh baselines
```

**Why.** Every condition of the sweep is compared against two reference conditions, with the predator and
without it, ten seeds each. They have to come from **this machine**, because results across different GPUs
are not bitwise identical. After them, three analyses run on their output:
- gate S5, "is the comparison condition well behaved", which must pass;
- gate S6, statistical power;
- the paired power analysis.

The project team chooses the sweep's seed count from S6 and the power analysis.

**Expected.** First the run script's own checks, with the date on each line:

```
2026-10-14 10:00:00  run.sh  environment: no JAX, XLA or Python path variable was inherited
2026-10-14 10:00:00  run.sh  environment: CUDA_VISIBLE_DEVICES=<unset>  LD_LIBRARY_PATH=<unset>
2026-10-14 10:00:09  run.sh  gpu probe: JAX sees NVIDIA RTX A6000
2026-10-14 10:00:09  run.sh  starting the treatment baseline (attempt 1 of 8)
```

Then the sweep's header (GPU, driver, commit, where the logs are), and its progress lines:

```
[  1/1] treatment__baseline__None starting
    gen   1/300  22.688 s  fit 0.0037 best 0.0625 zero 90.6%  predated 0.001 starved 0.999 survived 0.000
    gen  26/300   0.412 s  fit 0.0481 best 0.2120 zero 31.2%  predated 0.004 starved 0.957 survived 0.039
  seed 1/10 done in 2:08  final 0.0612 best 0.2441  0.411 s/gen  ram 2.10 GB  gpu mem 310 MB  gpu 71C ...
```

- **The first generation of each seed is slow**, often 10 to 25 seconds, because JAX compiles the program
  then. Later generations are fast.
- **`fit`** is the mean forage score. It should drift upward over the 300 generations.
- **`zero`** is the share of networks scoring nothing. It starts high and should fall.
- Then the null baseline, the same way, and then:

```
2026-10-14 11:20:00  run.sh  gate S5: is the null well behaved? Required.
<a pytest summary with "passed">
2026-10-14 11:21:00  run.sh  gate S6: statistical power at ten seeds. Recorded, not required: ...
<a pytest summary with "failed">
2026-10-14 11:22:00  run.sh  baselines done. Send results/s5.log, results/s6.log and results/pairing_power.log to the project team.
```

**S6 reporting `failed` is expected and is not a fault.** It reports that ten seeds fall short under its
deliberately conservative assumption. That is the question the team decides from these files, so the run
does not stop for it.

**Then: send `results/s5.log`, `results/s6.log` and `results/pairing_power.log` to the project team, and
wait for their go-ahead before step 5.**

**May I stop it?** Yes, with `Ctrl+C`, at any point.
- **During the evolution:** it prints `stopped by Ctrl-C (exit 130). Finished seeds are kept; the seed that
  was running is redone.` Run the same command again and it continues from there.
- **During S5, S6 or the power analysis:** running the command again skips the finished conditions and
  re-runs only those few minutes of analysis.
- **Wait for the `stopped` line before closing anything.** It can take up to a minute.

---

## 5. The sweep

```bash
bash scripts/a6000/run.sh sweep
```

**Why.** This is the experiment: 109 conditions, each one parameter moved away from its default, ten seeds
of 300 generations each. The two baseline conditions from step 4 are reused, so it starts with 2 of 109
done.

**Expected.**
- The same opening lines as step 4.
- Then `plan: 109 conditions x 10 seeds x 300 generations; 2 conditions already complete`.
- Then the progress lines, now with a time remaining on each finished condition:

```
[  14/109] treatment__p_wander__0.5   20.6 min  fit 0.058 +/- 0.004  ram 2104MB  done  12.8%  elapsed 4:51:10  remaining ~1d 09:12
```

The estimate settles after the first few conditions. It ends, days later, with:

```
SWEEP COMPLETE. Send the folder runs/sweep and the folder results. Checksums: results/sweep-checksums.sha256
```

**Watching it without disturbing it.** Detach with `Ctrl+b d`. Then, from any terminal:

| Command | Shows |
|---|---|
| `cat runs/sweep/status.json` | Where it is, the share done, time remaining, recent warnings. Rewritten at least once a minute |
| `tail -f runs/sweep/sweep.log` | Everything the run prints, live. `Ctrl+C` stops only the `tail`, never the run |
| `cat runs/sweep/warnings.log` | Only the warnings |
| `cat runs/sweep/run.log` | Starts, restarts and stops by the run script |
| `watch -n 5 nvidia-smi` | The GPU, live. `Ctrl+C` stops only the `watch` |

**What healthy looks like:**
- the `ram` and `gpu mem` figures on seed lines stay roughly flat from one seed to the next;
- the GPU temperature stays below 87 C;
- `status.json`'s `updated` time is never more than a few minutes old.

A line starting `WARNING` names anything unusual; the table in `README.md` says what each one means. Most
need nothing from you.

**No tmux available?** Start it like this instead, and follow the console log:

```bash
ALLOW_NO_TMUX=1 nohup bash scripts/a6000/run.sh sweep > results/console.log 2>&1 &
```
```bash
tail -f results/console.log
```

**May I stop it?** You may, but **prefer not to.** Every stop throws away the seed that was running, a few
minutes of work.
- **If you must, for a planned reboot or maintenance:** press `Ctrl+C` once and **wait for `stopped by
  Ctrl-C (exit 130)`**, up to a minute. To continue, run the same command again. Nothing finished is ever
  repeated.
- **Do not use `kill -9`.** Nothing would be corrupted, since every file is written to a temporary name and
  renamed, but it skips the clean stop.
- **If you press `Ctrl+Z` by accident**, the run is paused, not stopped: type `fg` and press Enter.

---

## 6. If something happens

| What happened | What to do |
|---|---|
| A `WARNING` line | Usually nothing. Look it up in `README.md` |
| `ended with exit 1` (or 4, or 5) `... Restarting in 60 seconds` | Nothing. It recovers by itself, up to 8 times in a row |
| The machine rebooted, lost power, or Windows restarted | Open a terminal, `cd ~/g-one-a6000`, `tmux new -s gone`, and run the **same command** as before. At most one seed is redone |
| `STOPPED (exit 3): the disk is nearly full` | Free disk space, then run the same command. Nothing is lost |
| `STOPPED: JAX cannot use the GPU from this terminal` | Check `nvidia-smi` works. If it does not, the driver needs attention from an administrator. Then run again |
| `STOPPED: a shipped file has been changed` | Undo it with `git checkout -- .` and run again |
| `STOPPED (exit 2)` for any other reason, or `8 attempts in a row` | **Do not retry.** Send `runs/sweep/sweep.log`, `runs/sweep/run.log` and the `results` folder |
| `another sweep is already running` | A copy is still going. `tmux attach -t gone` to find it. Do not start a second one |

**Never, while the run is going:**
- edit, move or delete any file in the repository or in `runs/sweep`;
- run anything else on the GPU;
- start a second copy.

---

## 7. Send back

When `SWEEP COMPLETE` appears, copy both folders, about 80 GB, to an external drive:

```
runs/sweep/      109 checkpoints, manifest.jsonl, sweep.log, warnings.log, run.log, status.json
results/         the machine check, the baseline gates, the power analysis, sweep-checksums.sha256
```

Before deleting anything on the machine, prove the copy is intact. On the drive, inside the copied `sweep`
folder:

```bash
sha256sum -c ../results/sweep-checksums.sha256
```

**Expected.** One line per file ending in `OK`, and nothing else. A line ending in `FAILED` means that file
was damaged in copying: copy it again from the machine.

**Summary.** Clone, start tmux, and run three commands, each of which prints what to do next. Every stop is
recoverable by running the same command again, except an exit 2, which means send the logs and wait.
