"""
Chuyen mo hinh YOLO da huan luyen sang Core ML de chay tren macOS/iOS.

Core ML la dinh dang native cua Apple, co the duoc nap truc tiep trong ung
dung iOS bang framework CoreML/Vision va co kha nang tang toc bang Apple
Neural Engine, GPU hoac CPU tuy tung thiet bi.

Vi du:
    python yolo/export_yolo_to_coreml.py

    python yolo/export_yolo_to_coreml.py \
        --model data/model_trained/vietnam_car_nano_model.pt \
        --output-dir data/model_trained/coreml \
        --imgsz 640 \
        --nms
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data/model_trained/vietnam_car_nano_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/model_trained/coreml"


def _parse_quantize(value: str | None) -> int | str | None:
    if value in {None, "", "none"}:
        return None
    if value in {"16", "fp16"}:
        return 16
    if value in {"8", "int8"}:
        return 8
    if value == "w8a16":
        return value
    raise argparse.ArgumentTypeError(
        "quantize chi nhan mot trong cac gia tri: none, 16, fp16, 8, int8, w8a16"
    )


def _ensure_coremltools_installed() -> None:
    if importlib.util.find_spec("coremltools") is not None:
        return

    raise RuntimeError(
        "Chua cai coremltools. Hay cai them dependency truoc khi export:\n"
        "    .venv/bin/python -m pip install 'coremltools>=9.0'\n"
        "Sau do chay lai script nay."
    )


def _move_exported_model(exported_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / exported_path.name

    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    shutil.move(str(exported_path), str(target_path))
    return target_path


def export_yolo_to_coreml(
    model_path: Path,
    output_dir: Path | None = DEFAULT_OUTPUT_DIR,
    imgsz: int = 640,
    batch: int = 1,
    dynamic: bool = False,
    nms: bool = True,
    conf: float = 0.25,
    iou: float = 0.45,
    quantize: int | str | None = 16,
    legacy_mlmodel: bool = False,
) -> Path:
    """
    Export YOLO PyTorch checkpoint sang Core ML.

    Mac/iOS hien dai nen uu tien `.mlpackage`. Chi dung `.mlmodel` khi ung
    dung iOS cu hoac pipeline hien tai bat buoc can file legacy.
    """
    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Khong tim thay model: {model_path}")
    if dynamic and nms:
        raise ValueError("Core ML khong ho tro dong thoi dynamic=True va nms=True.")

    _ensure_coremltools_installed()

    export_format = "mlmodel" if legacy_mlmodel else "coreml"
    model = YOLO(str(model_path))
    exported = model.export(
        format=export_format,
        imgsz=imgsz,
        batch=batch,
        dynamic=dynamic,
        nms=nms,
        conf=conf,
        iou=iou,
        quantize=quantize,
    )

    exported_path = Path(exported).resolve()
    if output_dir is None:
        return exported_path

    return _move_exported_model(exported_path, output_dir.resolve())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chuyen YOLO .pt sang Core ML .mlpackage/.mlmodel cho macOS va iOS."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Duong dan file YOLO .pt. Mac dinh: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thu muc chua file Core ML sau export. Mac dinh: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Kich thuoc input cua YOLO.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size khi export.")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Cho phep input size dong. Khong dung chung voi --nms.",
    )
    parser.add_argument(
        "--nms",
        dest="nms",
        action="store_true",
        default=True,
        help="Dong goi Non-Maximum Suppression vao Core ML model de app iOS de tich hop hon.",
    )
    parser.add_argument(
        "--no-nms",
        dest="nms",
        action="store_false",
        help="Khong dong goi NMS; app iOS se tu xu ly output raw cua YOLO.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Nguong confidence cho NMS.")
    parser.add_argument("--iou", type=float, default=0.45, help="Nguong IoU cho NMS.")
    parser.add_argument(
        "--quantize",
        type=_parse_quantize,
        default=16,
        help="Che do toi uu trong Core ML: none, 16/fp16, 8/int8, w8a16. Mac dinh: 16.",
    )
    parser.add_argument(
        "--legacy-mlmodel",
        action="store_true",
        help="Xuat file .mlmodel legacy thay vi .mlpackage.",
    )
    return parser


def main() -> None:
    if platform.system() not in {"Darwin", "Linux"}:
        raise RuntimeError("Ultralytics/Core ML export chi ho tro tren macOS hoac Linux.")

    args = build_arg_parser().parse_args()
    output_path = export_yolo_to_coreml(
        model_path=args.model,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=args.dynamic,
        nms=args.nms,
        conf=args.conf,
        iou=args.iou,
        quantize=args.quantize,
        legacy_mlmodel=args.legacy_mlmodel,
    )

    print("Export Core ML hoan tat.")
    print(f"File dau ra: {output_path}")
    print("Co the dua file nay vao Xcode va su dung bang CoreML/Vision trong ung dung iOS.")


if __name__ == "__main__":
    main()
