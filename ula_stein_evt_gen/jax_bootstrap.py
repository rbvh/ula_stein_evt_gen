# This must be imported before any JAX imports
# to ensure the JAX compilation cache is set up correctly.
# It sets up a persistent cache for JAX compilations.
# The cache directory is created if it does not exist.
import os, pathlib

CACHE_DIR = pathlib.Path(__file__).with_name("jax_cache")
CACHE_DIR.mkdir(exist_ok=True)

os.environ["JAX_COMPILATION_CACHE_DIR"] = str(CACHE_DIR)
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "-1"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"

# Helpful logs
# os.environ["JAX_LOG_COMPILES"] = "1"
# os.environ["JAX_LOGGING_LEVEL"] = "DEBUG"
# os.environ["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.compilation_cache"
