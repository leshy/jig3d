"""End-to-end tests for the `jig` CLI, all examples, and f3d rendering."""

from pathlib import Path
import shutil

from PIL import Image
import pytest

from jig.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("script", ["wall_intercom.py", "inline_valve.py", "eva_helmet.py"])
def test_cli_build_describe_cost_metrics(script, tmp_path):
    src = EXAMPLES / script
    out_glb = tmp_path / (src.stem + ".glb")
    assert main(["build", str(src), "-o", str(out_glb)]) == 0
    assert out_glb.is_file() and out_glb.stat().st_size > 1000
    assert main(["describe", str(src)]) == 0
    assert main(["cost", str(src)]) == 0
    assert main(["metrics", str(out_glb)]) == 0


def test_cli_multi_variant_and_new_and_words(tmp_path):
    assert main(["build", str(EXAMPLES / "inline_valve.py"), "--all", "-o", str(tmp_path)]) == 0
    for name in ("inline_valve.glb", "inline_valve_r05.glb", "inline_valve_r08.glb"):
        assert (tmp_path / name).is_file()

    new_py = tmp_path / "starter_widget.py"
    assert main(["new", str(new_py)]) == 0
    assert main(["build", str(new_py), "-o", str(tmp_path / "starter_widget.glb")]) == 0
    assert main(["words", "chamfer"]) == 0
    assert main(["materials"]) == 0


def test_cli_sheet(tmp_path):
    pics = []
    for i in range(3):
        p = tmp_path / f"img_{i}.png"
        Image.new("RGB", (120, 90), (40 * i, 100, 180)).save(p)
        pics.append(str(p))
    out = tmp_path / "sheet.png"
    assert main(["sheet", *pics, "--labels", "one,-,three", "-o", str(out)]) == 0
    assert out.is_file()


@pytest.mark.skipif(not shutil.which("f3d"), reason="f3d not installed")
def test_cli_view_and_look_with_f3d(tmp_path):
    view_png = tmp_path / "helmet_view.png"
    assert main(["view", str(EXAMPLES / "eva_helmet.py"), "-o", str(view_png)]) == 0
    im = Image.open(view_png)
    assert im.width == 1800 and im.height > 1000

    scaled_png = tmp_path / "intercom_scale.png"
    assert main(["view", str(EXAMPLES / "wall_intercom.py"), "--scale", "--themes", "-o", str(scaled_png)]) == 0
    im_s = Image.open(scaled_png)
    assert im_s.width == 1800 and im_s.height == im.height + 380

    intercom_glb = tmp_path / "wall_intercom.glb"
    valve_glb = tmp_path / "inline_valve.glb"
    assert main(["build", str(EXAMPLES / "wall_intercom.py"), "-o", str(intercom_glb)]) == 0
    assert main(["build", str(EXAMPLES / "inline_valve.py"), "-o", str(valve_glb)]) == 0

    look_png = tmp_path / "compare.png"
    assert main(["look", str(intercom_glb), str(valve_glb), "--views", "3q,front", "--size", "400", "--slots", "--scale", "-o", str(look_png)]) == 0
    im2 = Image.open(look_png)
    assert im2.width >= 800 and im2.height >= 400
