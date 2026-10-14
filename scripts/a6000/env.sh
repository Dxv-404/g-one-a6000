# Sourced by check.sh and run.sh, never run on its own. D-063.
#
# What check.sh certifies holds only for the environment it ran in, and a login profile on a shared machine
# can carry variables that change the result without any error: which device JAX uses (JAX_PLATFORMS), the
# arithmetic (JAX_ENABLE_X64, JAX_DEFAULT_MATMUL_PRECISION), the compiler (XLA_FLAGS), or which code Python
# imports (PYTHONPATH, PYTHONHOME). So EVERY inherited JAX_ and XLA_ variable is removed, not a list of known
# bad ones, and both Python path variables, in the check and the run alike, and each removal is printed.
# src/__init__.py then sets the ones this project needs.
#
# CUDA_VISIBLE_DEVICES and LD_LIBRARY_PATH are reported and left alone: on some machines the driver is only
# found through them. Whether JAX really has the GPU is proved by gate E0 in the check and by gpu_probe
# before every run, which test the outcome rather than the variables.

scrub_environment() {
  local v removed=0
  for v in $(compgen -e | grep -E '^(JAX_|XLA_)') PYTHONPATH PYTHONHOME; do
    if [ -n "${!v+x}" ]; then
      echo "environment: removed $v=${!v}"
      unset "$v"
      removed=1
    fi
  done
  [ "$removed" = 0 ] && echo "environment: no JAX, XLA or Python path variable was inherited"
  echo "environment: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}  LD_LIBRARY_PATH=${LD_LIBRARY_PATH-<unset>}"
}

# JAX must see the GPU from this shell, now. A sweep that fell back to the CPU would run for weeks and its
# numbers would not be comparable with the baselines, since E1 rules out agreement across devices.
gpu_probe() {
  "$1" - <<'EOF'
import src  # noqa: F401  the runtime settings, before any JAX operation
import jax

d = jax.devices()[0]
assert d.platform == "gpu", f"JAX is using the {d.platform}, not the GPU"
print(f"gpu probe: JAX sees {d.device_kind}")
EOF
}
