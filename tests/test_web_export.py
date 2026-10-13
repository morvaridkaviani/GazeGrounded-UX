import json

from PIL import Image

from gazegrounded.web_export import export_site


def test_export_site_writes_consistent_json_and_maps(tmp_path, pipeline_cfg, pipeline_features):
    path = export_site(pipeline_cfg, out_dir=tmp_path)
    site = json.loads(path.read_text())
    assert site["meta"]["dataKind"] == "synthetic_demo" and site["meta"]["notice"]
    assert len(site["trials"]) == len(pipeline_features)
    ui_ids = {u["id"] for u in site["uis"]}
    for t in site["tasks"]:
        assert t["ui"] in ui_ids
        assert (tmp_path / t["humanMap"]).exists() and (tmp_path / t["modelMap"]).exists()
        assert all(0 <= v <= 1 for e in t["model"]["elements"] for v in e["box"])
    trial = site["trials"][0]
    heat = Image.open(tmp_path / trial["heatmap"])
    assert heat.mode == "L" and heat.size == (180, 112)
    for u in site["uis"]:
        assert (tmp_path / u["image"]).exists()
