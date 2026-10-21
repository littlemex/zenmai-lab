# [Bug Report] isaaclab.sh fails under non-tty TERM (cloud-init/SSM) due to 'tabs 4' + set -e

## Describe the bug

On line 16 of `isaaclab.sh`, the script unconditionally calls `tabs 4` to set tab spacing. The script is also run with `set -e` (exit on error). When the `TERM` environment variable is empty or set to `unknown` — as is the case in cloud-init, AWS SSM Run Command, Docker containers without a TTY, and other non-interactive execution environments — the `tabs` command exits with a non-zero status and prints an error such as `'unknown': I need something more specific.` or `'': unknown terminal type.`

Because `tabs 4` is the **first executable command** in `isaaclab.sh` and `set -e` is active, every single invocation of `isaaclab.sh` — regardless of the subcommand (`--install`, `--conda`, etc.) — fails immediately in these environments. Tab formatting is cosmetic and entirely non-essential to the script's function, yet it makes automated cloud deployments and CI pipelines impossible without workarounds.

**Root cause:** `tabs 4` requires a recognized, interactive terminal type in `$TERM`. Under non-tty execution environments, `TERM` is typically unset or `unknown`, causing `tabs` to fail. Combined with `set -e`, this fatal exit occurs before any useful work is done.

## Steps to Reproduce

The failure can be reproduced on any system with `bash` and the `tabs` utility. No Isaac Sim installation is required to observe the crash.

```bash
# Fails: TERM=unknown (common default in SSM / cloud-init)
$ TERM=unknown bash -c "set -e; tabs 4; echo OK"
'unknown': I need something more specific.
# exit code: 1

# Fails: TERM unset / empty (common in Docker without TTY)
$ TERM= bash -c "set -e; tabs 4; echo OK"
'': unknown terminal type.
# exit code: 1

# Succeeds: TERM set to a known interactive terminal
$ TERM=xterm bash -c "set -e; tabs 4; echo OK"
# (escape sequences printed)
OK
# exit code: 0
```

Mapping to `isaaclab.sh`:

```bash
# isaaclab.sh lines 14-16 (release/2.2.0)
# Set tab-spaces
tabs 4   # <-- fatal under non-tty TERM with set -e active
```

To reproduce the full failure:

```bash
# On a fresh Ubuntu 24.04 instance launched via cloud-init or SSM:
env -i HOME=$HOME PATH=$PATH bash /path/to/IsaacLab/isaaclab.sh --install
# Immediately exits with tabs error before any install logic runs.
```

## System Info

| Field | Value |
|---|---|
| IsaacLab Commit / Branch | `release/2.2.0` |
| Isaac Sim Version | `5.1.0-rc.19` |
| OS | Ubuntu 24.04 LTS |
| Instance Type | AWS `g6e.2xlarge` |
| GPU | NVIDIA L40S |
| GPU Driver | 580.126.09 |
| CUDA | 13.0 |

## Additional Context

This bug was discovered while deploying Isaac Lab via the [aws-samples/sample-physical-ai-scaffolding-kit](https://github.com/aws-samples/sample-physical-ai-scaffolding-kit), which provisions GPU instances and bootstraps Isaac Lab through cloud-init user-data scripts. Because cloud-init runs shell commands without a TTY and typically sets `TERM=unknown` or leaves it unset, every `isaaclab.sh` invocation failed at startup before performing any meaningful work.

**TERM values that trigger the failure:**

- `TERM=` (empty / unset) — Docker without `--tty`, Lambda, some CI runners
- `TERM=unknown` — AWS SSM Run Command, some cloud-init environments
- `TERM=dumb` — certain CI systems (e.g., Jenkins non-interactive agents)

**TERM values that work correctly:**

- `TERM=xterm`, `TERM=xterm-256color`, `TERM=linux`, `TERM=vt100`

This affects **any cloud bootstrap workflow**: EC2 user-data, SSM Run Command, Docker build steps, GitHub Actions without TTY allocation, and any CI system that does not allocate a pseudo-terminal. The impact is a complete blocker for automated deployment — the script cannot reach `--install`, `--conda`, or any other subcommand.

Note that `tabs 4` is purely cosmetic (it adjusts terminal tab-stop display width). The failure is only fatal because `set -e` is active in the script; without `set -e` it would merely print a warning. However, removing `set -e` is not the right fix.

**Suggested fix:**

Two safe options:

**Option (a)** — make `tabs` non-fatal (minimal change):

```bash
# Set tab-spaces
tabs 4 2>/dev/null || true
```

**Option (b)** — skip `tabs` when there is no interactive terminal (recommended):

```bash
# Set tab-spaces only when running in an interactive terminal
[ -t 1 ] && tabs 4
```

Alternatively, guard on `TERM`:

```bash
[ -n "${TERM:-}" ] && [ "$TERM" != "dumb" ] && [ "$TERM" != "unknown" ] && tabs 4
```

**Option (b) is recommended** because it preserves the intended tab-formatting behavior in real interactive terminals while silently skipping the non-essential call in all non-tty environments. It requires no suppression of error output and correctly handles the full class of TTY-less execution contexts via the POSIX `[ -t 1 ]` file-descriptor test.

## Checklist

- [x] I have checked that there is no similar bug report in the existing issues.
- [x] I have verified that this bug exists on the latest release branch (`release/2.2.0`).
- [x] I have provided a minimal reproduction case that does not require a full Isaac Sim installation.

## Acceptance Criteria

- Running `env -i HOME=$HOME PATH=$PATH bash isaaclab.sh --help` (or any subcommand) in a shell where `TERM` is unset or set to `unknown` completes without error and does not exit prematurely due to the `tabs` call.
- Running `isaaclab.sh --help` in a normal interactive terminal still applies tab formatting as before (i.e., the fix does not regress interactive usage).