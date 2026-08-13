"""
Tests for Multi-Load Cowboy: geometry maths plus a CPU run of the node.

The node is loaded inside the real comfy environment (no server) through
a synthetic package, so TrentNodes/__init__ discovery does not run. The
image files are written to a temp folder and folder_paths is pointed at
it, so the test never touches the real ComfyUI input directory.

Run from the ComfyUI root:

    venv/bin/python custom_nodes/TrentNodes/tests/test_multi_load_cowboy.py
"""

import importlib
import os
import sys
import tempfile
import types

ROOT = "/home/trent/ComfyUI"
PKG = os.path.join(ROOT, "custom_nodes", "TrentNodes")

sys.path.insert(0, ROOT)

pkg = types.ModuleType("TrentNodes")
pkg.__path__ = [PKG]
sys.modules["TrentNodes"] = pkg
for sub in ("nodes", "utils"):
    module = types.ModuleType(f"TrentNodes.{sub}")
    module.__path__ = [os.path.join(PKG, sub)]
    sys.modules[f"TrentNodes.{sub}"] = module

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import folder_paths  # noqa: E402

mlc = importlib.import_module("TrentNodes.nodes.multi_load_cowboy")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name} {detail}")


# --- fixtures --------------------------------------------------------

TMP = tempfile.mkdtemp(prefix="mlc_test_")


def write_image(name, width, height, alpha=False):
    """Write a test image and return its file name."""
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 31))
    data = (rng.random((height, width, 4 if alpha else 3)) * 255)
    data = data.astype(np.uint8)
    if alpha:
        # Left half opaque, right half transparent.
        data[:, :width // 2, 3] = 255
        data[:, width // 2:, 3] = 0
    img = Image.fromarray(data, mode="RGBA" if alpha else "RGB")
    img.save(os.path.join(TMP, name))
    return name


def patch_folder_paths():
    """Resolve slot values against the temp folder instead of input/."""
    folder_paths.get_annotated_filepath = lambda value, *a, **k: (
        os.path.join(TMP, value.replace(" [input]", ""))
    )
    folder_paths.exists_annotated_filepath = lambda value: os.path.isfile(
        os.path.join(TMP, value.replace(" [input]", ""))
    )


# --- geometry --------------------------------------------------------

def test_plan_size():
    print("\n[geom-1] plan_size across every mode")

    # stretch ignores the source aspect
    check(
        "stretch-exact",
        mlc.plan_size("stretch", 800, 400, 1024, 1024, 1) ==
        (1024, 1024, 1024, 1024),
    )

    # resize fits inside the box and keeps the aspect
    scaled_w, scaled_h, canvas_w, canvas_h = mlc.plan_size(
        "resize", 800, 400, 1024, 1024, 1
    )
    check(
        "resize-fits-inside",
        (scaled_w, scaled_h) == (1024, 512) and
        (canvas_w, canvas_h) == (1024, 512),
        f"got {scaled_w}x{scaled_h}",
    )

    # pad fits inside then keeps the full box as the canvas
    scaled_w, scaled_h, canvas_w, canvas_h = mlc.plan_size(
        "pad", 800, 400, 1024, 1024, 1
    )
    check(
        "pad-canvas-is-the-box",
        (scaled_w, scaled_h) == (1024, 512) and
        (canvas_w, canvas_h) == (1024, 1024),
        f"got {scaled_w}x{scaled_h} on {canvas_w}x{canvas_h}",
    )

    # crop fills the box exactly
    check(
        "crop-exact",
        mlc.plan_size("crop", 800, 400, 1024, 1024, 1) ==
        (1024, 1024, 1024, 1024),
    )

    # total_pixels keeps the aspect and matches the pixel budget
    scaled_w, scaled_h, _, _ = mlc.plan_size(
        "total_pixels", 800, 400, 1024, 1024, 1
    )
    area = scaled_w * scaled_h
    check(
        "total-pixels-area",
        abs(area - 1024 * 1024) / (1024 * 1024) < 0.01 and
        abs((scaled_w / scaled_h) - 2.0) < 0.02,
        f"got {scaled_w}x{scaled_h}, area {area}",
    )

    # a zero side is taken from the other one
    check(
        "zero-width-derives",
        mlc.plan_size("resize", 800, 400, 0, 200, 1) == (400, 200, 400, 200),
    )
    check(
        "zero-height-derives",
        mlc.plan_size("resize", 800, 400, 400, 0, 1) == (400, 200, 400, 200),
    )
    check(
        "zero-both-keeps-source",
        mlc.plan_size("resize", 800, 400, 0, 0, 1) == (800, 400, 800, 400),
    )

    # crop and total_pixels have no box to work from, so they fall back
    check(
        "crop-without-box-falls-back",
        mlc.plan_size("crop", 800, 400, 0, 512, 1) == (1024, 512, 1024, 512),
    )

    # divisible_by rounds down and never overhangs the canvas
    scaled_w, scaled_h, canvas_w, canvas_h = mlc.plan_size(
        "pad", 999, 501, 1000, 1000, 16
    )
    check(
        "divisible-by-16",
        canvas_w % 16 == 0 and canvas_h % 16 == 0 and
        scaled_w <= canvas_w and scaled_h <= canvas_h,
        f"{scaled_w}x{scaled_h} in {canvas_w}x{canvas_h}",
    )
    scaled_w, scaled_h, _, _ = mlc.plan_size(
        "resize", 999, 501, 1000, 1000, 8
    )
    check(
        "divisible-by-8-resize",
        scaled_w % 8 == 0 and scaled_h % 8 == 0,
        f"{scaled_w}x{scaled_h}",
    )


def test_color_and_boxes():
    print("\n[geom-2] colours, crop boxes and pad offsets")

    check("color-triple-bytes", mlc.parse_color("255, 128, 0") ==
          (1.0, 128 / 255.0, 0.0))
    check("color-hex", mlc.parse_color("#ff8000") ==
          (1.0, 128 / 255.0, 0.0))
    check("color-short-hex", mlc.parse_color("#fff") == (1.0, 1.0, 1.0))
    check("color-normalised", mlc.parse_color("1, 0.5, 0")[1] == 0.5)
    check("color-garbage-is-black", mlc.parse_color("nope") ==
          (0.0, 0.0, 0.0))

    # A wide source cropped to a square keeps the full height
    x, y, box_w, box_h = mlc.crop_box(800, 400, 512, 512, "center")
    check(
        "crop-box-center",
        (box_w, box_h) == (400, 400) and (x, y) == (200, 0),
        f"got {box_w}x{box_h} at {x},{y}",
    )
    x, _, _, _ = mlc.crop_box(800, 400, 512, 512, "left")
    check("crop-box-left", x == 0)
    x, _, box_w, _ = mlc.crop_box(800, 400, 512, 512, "right")
    check("crop-box-right", x == 800 - box_w)

    check("pad-offset-center", mlc.pad_offsets(1000, 1000, 800, 400,
                                               "center") == (100, 300))
    check("pad-offset-top", mlc.pad_offsets(1000, 1000, 800, 400,
                                            "top") == (100, 0))
    check("pad-offset-bottom", mlc.pad_offsets(1000, 1000, 800, 400,
                                               "bottom") == (100, 600))


# --- node ------------------------------------------------------------

def run_node(slots, **kwargs):
    node = mlc.MultiLoadCowboy()
    params = {
        "width": 512,
        "height": 512,
        "resize_mode": "pad",
        "upscale_method": "bilinear",
        "crop_position": "center",
        "pad_color": "0, 0, 0",
        "divisible_by": 8,
        "device": "cpu",
    }
    params.update(kwargs)
    for index, name in enumerate(mlc.slot_names()):
        params[name] = slots[index] if index < len(slots) else mlc.EMPTY
    with torch.inference_mode():
        return node.load(**params)


def test_partial_grid():
    print("\n[node-1] half-filled grid loads without an error")
    wide = write_image("wide.png", 640, 320)
    tall = write_image("tall.png", 300, 600)
    square = write_image("square.png", 400, 400)
    patch_folder_paths()

    out = run_node([wide, mlc.EMPTY, tall, mlc.EMPTY, square, mlc.EMPTY])
    images, masks, count, width, height = out[:5]
    singles = out[5:]

    check("count-is-three", count == 3, f"got {count}")
    check(
        "batch-shape",
        tuple(images.shape) == (3, 512, 512, 3),
        f"got {tuple(images.shape)}",
    )
    check("reported-size", (width, height) == (512, 512),
          f"got {width}x{height}")
    check(
        "mask-shape",
        tuple(masks.shape) == (3, 512, 512),
        f"got {tuple(masks.shape)}",
    )
    check(
        "empty-slots-output-nothing",
        singles[1] is None and singles[3] is None and singles[5] is None,
    )
    check(
        "filled-slots-are-tensors",
        all(torch.is_tensor(singles[i]) for i in (0, 2, 4)),
    )
    check(
        "values-stay-in-range",
        float(images.min()) >= 0.0 and float(images.max()) <= 1.0,
    )
    check("output-is-cpu-float32",
          images.device.type == "cpu" and images.dtype == torch.float32)

    # The wide image padded into a square: top and bottom rows are pad
    # colour, and the mask marks them.
    wide_out = singles[0]
    check(
        "pad-fills-with-color",
        float(wide_out[0, 0, :, :].abs().max()) < 1e-6,
        f"top row max {float(wide_out[0, 0, :, :].abs().max()):.4f}",
    )
    check(
        "pad-marks-the-mask",
        float(masks[0, 0, 0]) > 0.99 and float(masks[0, 256, 256]) < 0.01,
        f"corner {float(masks[0, 0, 0]):.2f}, "
        f"middle {float(masks[0, 256, 256]):.2f}",
    )


def test_alpha_and_modes():
    print("\n[node-2] alpha, crop and stretch")
    cutout = write_image("cutout.png", 400, 400, alpha=True)
    patch_folder_paths()

    out = run_node([cutout], resize_mode="stretch", width=256, height=128)
    images, masks, count = out[0], out[1], out[2]
    check("stretch-exact-size", tuple(images.shape) == (1, 128, 256, 3),
          f"got {tuple(images.shape)}")
    check(
        "alpha-becomes-mask",
        float(masks[0, 64, 20]) < 0.02 and float(masks[0, 64, 235]) > 0.98,
        f"left {float(masks[0, 64, 20]):.2f}, "
        f"right {float(masks[0, 64, 235]):.2f}",
    )

    out = run_node([cutout], resize_mode="crop", width=640, height=320)
    check("crop-exact-size", tuple(out[0].shape) == (1, 320, 640, 3),
          f"got {tuple(out[0].shape)}")

    out = run_node([cutout], resize_mode="pad_edge", width=512, height=256)
    edge = out[0]
    check(
        "pad-edge-is-not-black",
        float(edge[0, :, 0, :].abs().max()) > 0.01,
        f"left column max {float(edge[0, :, 0, :].abs().max()):.3f}",
    )

    out = run_node([cutout], resize_mode="pad", width=512, height=256,
                   pad_color="#ff0000")
    padded = out[0]
    check(
        "pad-color-honoured",
        float(padded[0, 0, 0, 0]) > 0.99 and float(padded[0, 0, 0, 1]) < 0.01,
        f"corner {[round(float(v), 2) for v in padded[0, 0, 0]]}",
    )


def test_mixed_sizes_batch():
    print("\n[node-3] aspect-keeping modes still make one batch")
    wide = write_image("wide2.png", 640, 320)
    tall = write_image("tall2.png", 320, 640)
    patch_folder_paths()

    out = run_node([wide, tall], resize_mode="resize", width=512,
                   height=512, divisible_by=8)
    images, masks, count = out[0], out[1], out[2]
    singles = out[5:]

    check("mixed-count", count == 2, f"got {count}")
    check(
        "mixed-batch-is-square-union",
        tuple(images.shape) == (2, 512, 512, 3),
        f"got {tuple(images.shape)}",
    )
    check(
        "singles-keep-their-own-size",
        tuple(singles[0].shape) == (1, 256, 512, 3) and
        tuple(singles[1].shape) == (1, 512, 256, 3),
        f"got {tuple(singles[0].shape)} and {tuple(singles[1].shape)}",
    )
    check("mixed-mask-shape", tuple(masks.shape) == (2, 512, 512),
          f"got {tuple(masks.shape)}")


def test_empty_and_missing():
    print("\n[node-4] an empty grid and a stale file never raise")
    patch_folder_paths()

    out = run_node([], width=768, height=256, divisible_by=8)
    images, masks, count, width, height = out[:5]
    check("empty-count-zero", count == 0, f"got {count}")
    check(
        "empty-returns-one-black-frame",
        tuple(images.shape) == (1, 256, 768, 3) and
        float(images.abs().max()) == 0.0,
        f"got {tuple(images.shape)}",
    )
    check("empty-reports-size", (width, height) == (768, 256))
    check("empty-singles-are-none", all(s is None for s in out[5:]))

    out = run_node(["does_not_exist.png"])
    check("missing-file-is-skipped", out[2] == 0, f"count {out[2]}")

    valid = mlc.MultiLoadCowboy.VALIDATE_INPUTS(
        image_1=mlc.EMPTY, image_2="does_not_exist.png"
    )
    check(
        "validate-flags-the-missing-file",
        isinstance(valid, str) and "does_not_exist.png" in valid,
        f"got {valid!r}",
    )
    check(
        "validate-accepts-an-empty-grid",
        mlc.MultiLoadCowboy.VALIDATE_INPUTS(image_1=mlc.EMPTY) is True,
    )


def test_is_changed():
    print("\n[node-5] IS_CHANGED tracks the files")
    name = write_image("changing.png", 64, 64)
    patch_folder_paths()

    first = mlc.MultiLoadCowboy.IS_CHANGED(image_1=name)
    same = mlc.MultiLoadCowboy.IS_CHANGED(image_1=name)
    check("stable-while-the-file-is", first == same)

    write_image("changing.png", 96, 96)
    after = mlc.MultiLoadCowboy.IS_CHANGED(image_1=name)
    check("changes-when-the-file-does", first != after)

    other = mlc.MultiLoadCowboy.IS_CHANGED(image_2=name)
    check("changes-with-the-slot", first != other)


def test_poisoned_widget_values():
    print("\n[node-7] a workflow with the wrong kind of value still runs")
    name = write_image("poison.png", 320, 200)
    patch_folder_paths()

    check("as_int reads numbers", mlc.as_int(512, 8) == 512)
    check("as_int reads numeric text", mlc.as_int("512", 8) == 512)
    check("as_int falls back on a combo string",
          mlc.as_int("(empty)", 1024) == 1024)
    check("as_int falls back on None", mlc.as_int(None, 64) == 64)

    # Exactly what an older shifted save left behind.
    out = run_node([name], width="(empty)", height="pad",
                   resize_mode="lanczos", upscale_method="center",
                   crop_position="0, 0, 0", divisible_by="cpu")
    images, count = out[0], out[2]
    check("poisoned run still loads the image", count == 1, f"count {count}")
    check(
        "poisoned run falls back to the defaults",
        tuple(images.shape) == (1, 1024, 1024, 3),
        f"got {tuple(images.shape)}",
    )


def test_node_contract():
    print("\n[node-6] the node's declared shape")
    types_in = mlc.MultiLoadCowboy.INPUT_TYPES()
    required = types_in["required"]
    check(
        "six-slots-declared",
        all(name in required for name in mlc.slot_names()),
    )
    check(
        "empty-is-the-default",
        all(required[name][1]["default"] == mlc.EMPTY
            for name in mlc.slot_names()),
    )
    check(
        "empty-is-an-option",
        all(mlc.EMPTY in required[name][0] for name in mlc.slot_names()),
    )
    check(
        "outputs-match-names",
        len(mlc.MultiLoadCowboy.RETURN_TYPES) ==
        len(mlc.MultiLoadCowboy.RETURN_NAMES) ==
        len(mlc.MultiLoadCowboy.OUTPUT_TOOLTIPS) == 11,
    )
    check(
        "category-is-a-trent-subfolder",
        mlc.MultiLoadCowboy.CATEGORY.startswith("Trent/"),
        mlc.MultiLoadCowboy.CATEGORY,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    test_plan_size()
    test_color_and_boxes()
    test_partial_grid()
    test_alpha_and_modes()
    test_mixed_sizes_batch()
    test_empty_and_missing()
    test_is_changed()
    test_poisoned_widget_values()
    test_node_contract()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)
