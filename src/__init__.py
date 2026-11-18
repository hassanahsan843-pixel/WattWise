"""WattWise AI source package."""
import pandas as pd
import warnings

# Use fastparquet on Windows environments where pyarrow C++ binaries might be restricted
try:
    pd.set_option("io.parquet.engine", "fastparquet")
except Exception:
    pass

# Suppress pandas deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)

