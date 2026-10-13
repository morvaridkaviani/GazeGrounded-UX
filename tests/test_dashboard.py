"""Dashboard pages execute without exceptions (with or without pipeline outputs present)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DASH = Path(__file__).resolve().parents[1] / "dashboard"
PAGES = [
    str(DASH / "app.py"),
    str(DASH / "explorer.py"),
    *sorted(str(p) for p in (DASH / "pages").glob("*.py")),
]


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=lambda p: Path(p).name)
def test_page_runs_without_exceptions(page):
    at = AppTest.from_file(page, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]


def test_plotly_builders(synthetic_ds):
    import numpy as np
    from PIL import Image

    from gazegrounded.visualization import dashboard_utils as du

    image = Image.open(synthetic_ds.image_path("shop_checkout_clean"))
    fig = du.overlay_figure(image, np.random.default_rng(0).random((112, 180)), "vlm")
    assert len(fig.data) == 1 and fig.layout.images
    import pandas as pd

    fix = pd.DataFrame(
        {
            "start_ms": [0, 300],
            "duration_ms": [200, 250],
            "centroid_x": [100.0, 600.0],
            "centroid_y": [100.0, 400.0],
            "aoi_id": ["header", None],
        }
    )
    fig = du.fixation_figure(
        image, fix, synthetic_ds.aois_for("shop_checkout_clean"), ["edit_address"]
    )
    assert len(fig.data) == 2  # scanpath line + markers
