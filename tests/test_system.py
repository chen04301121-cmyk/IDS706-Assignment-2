import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import matplotlib.image as mpimg
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_complete_analysis_pipeline(tmp_path):
    """Run the real analysis and verify its results and charts."""

    # Copy the real dataset into a temporary working directory.
    shutil.copy2(
        PROJECT_ROOT / "gold_data_2015_25.csv",
        tmp_path / "gold_data_2015_25.csv",
    )

    # Run plotting without opening windows.
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib_config")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "overview.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    # The entire script must finish successfully.
    assert result.returncode == 0, (
        f"Analysis failed.\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )

    # Check the filtering result recorded for this dataset.
    count_match = re.search(
        r"Number of rows where GLD is above average:\s*(\d+)",
        result.stdout,
    )
    assert count_match is not None, "Filtering result was not printed."
    assert int(count_match.group(1)) == 1299

    # Check the model evaluation output.
    mae_match = re.search(
        r"Mean Absolute Error:\s*(\S+)",
        result.stdout,
    )
    r2_match = re.search(
        r"R-squared:\s*(\S+)",
        result.stdout,
    )

    assert mae_match is not None, "MAE was not printed."
    assert r2_match is not None, "R-squared was not printed."

    mae = float(mae_match.group(1))
    r_squared = float(r2_match.group(1))

    assert math.isfinite(mae)
    assert math.isfinite(r_squared)

    # Compare with the existing project's documented results.
    assert mae == pytest.approx(31.27, abs=0.1)
    assert r_squared == pytest.approx(0.10, abs=0.02)

    # Both plots must exist and contain readable, non-uniform images.
    for filename in [
        "gld_price_trend.png",
        "actual_vs_estimated_gld.png",
    ]:
        image_path = tmp_path / filename

        assert image_path.is_file(), f"Missing chart: {filename}"

        image = mpimg.imread(image_path)

        assert image.shape[0] > 0
        assert image.shape[1] > 0
        assert image[:, :, :3].std() > 0, f"Blank chart: {filename}"