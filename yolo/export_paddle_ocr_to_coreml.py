"""
Chuyen mo hinh PaddleOCR recognition sang Core ML cho macOS/iOS.

Core ML Tools khong doc truc tiep file Paddle inference, nen flow an toan la:

    Paddle inference -> ONNX static -> PyTorch module -> Core ML .mlpackage

Trong project nay da co san ONNX static tai:

    data/model_trained/ocr_model_static.onnx

Vi vay cach chay thuong dung la:

    python yolo/export_paddle_ocr_to_coreml.py

Neu muon tao lai ONNX tu thu muc Paddle goc:

    python yolo/export_paddle_ocr_to_coreml.py --rebuild-onnx

Luu y: buoc --rebuild-onnx can co CLI paddle2onnx trong moi truong hien tai.
"""

from __future__ import annotations

import argparse
import importlib.util
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PADDLE_MODEL_DIR = PROJECT_ROOT / "data/latin_PP-OCRv5_mobile_rec"
DEFAULT_ONNX_MODEL_PATH = PROJECT_ROOT / "data/model_trained/ocr_model_static.onnx"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/model_trained/coreml"
DEFAULT_INPUT_SHAPE = (1, 3, 48, 320)
DEFAULT_INPUT_NAME = "x"
DEFAULT_MODEL_NAME = "latin_PP-OCRv5_mobile_rec"


def _ensure_module_installed(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    raise RuntimeError(
        f"Chua cai {module_name}. Hay cai them dependency truoc khi export:\n"
        f"    {install_hint}\n"
        "Sau do chay lai script nay."
    )


def _ensure_command_exists(command: str, install_hint: str) -> None:
    if shutil.which(command):
        return
    raise RuntimeError(
        f"Khong tim thay command `{command}`.\n"
        f"Hay cai them dependency truoc khi tao ONNX:\n"
        f"    {install_hint}\n"
        "Hoac bo --rebuild-onnx de dung file ONNX static da co san."
    )


def _target_from_name(coremltools, target_name: str):
    try:
        return getattr(coremltools.target, target_name)
    except AttributeError as exc:
        valid_targets = ", ".join(
            name for name in dir(coremltools.target) if name.startswith(("iOS", "macOS"))
        )
        raise ValueError(
            f"minimum_deployment_target khong hop le: {target_name}. "
            f"Cac gia tri co the dung: {valid_targets}"
        ) from exc


def _parse_input_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("input-shape phai co dang: 1,3,48,320") from exc

    if len(shape) != 4 or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("input-shape phai gom 4 so duong, vi du: 1,3,48,320")
    return shape


def export_paddle_ocr_to_onnx(
    paddle_model_dir: Path = DEFAULT_PADDLE_MODEL_DIR,
    output_onnx_path: Path = DEFAULT_ONNX_MODEL_PATH,
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    input_name: str = DEFAULT_INPUT_NAME,
    opset_version: int = 11,
    dynamic_onnx_path: Path | None = None,
) -> Path:
    """
    Tao ONNX static tu PaddleOCR inference model.

    Buoc nay phu thuoc CLI `paddle2onnx` va `onnxsim`. Tren macOS arm64,
    `paddle2onnx` co the keo version ONNX cu, nen neu gap loi dependency thi
    nen chay rieng buoc nay trong Colab/Linux, sau do copy file ONNX static ve
    `data/model_trained/ocr_model_static.onnx`.
    """
    paddle_model_dir = paddle_model_dir.resolve()
    output_onnx_path = output_onnx_path.resolve()
    dynamic_onnx_path = (
        dynamic_onnx_path.resolve()
        if dynamic_onnx_path is not None
        else output_onnx_path.with_name(f"{output_onnx_path.stem}_dynamic.onnx")
    )

    if not paddle_model_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc PaddleOCR: {paddle_model_dir}")

    model_file = paddle_model_dir / "inference.json"
    params_file = paddle_model_dir / "inference.pdiparams"
    if not model_file.exists() or not params_file.exists():
        raise FileNotFoundError(
            "Thu muc PaddleOCR phai co inference.json va inference.pdiparams."
        )

    _ensure_command_exists(
        "paddle2onnx",
        ".venv/bin/python -m pip install paddle2onnx",
    )
    _ensure_command_exists(
        "onnxsim",
        ".venv/bin/python -m pip install onnxsim",
    )

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "paddle2onnx",
            "--model_dir",
            str(paddle_model_dir),
            "--model_filename",
            model_file.name,
            "--params_filename",
            params_file.name,
            "--save_file",
            str(dynamic_onnx_path),
            "--opset_version",
            str(opset_version),
        ],
        check=True,
    )

    subprocess.run(
        [
            "onnxsim",
            str(dynamic_onnx_path),
            str(output_onnx_path),
            "--overwrite-input-shape",
            f"{input_name}:{','.join(str(dim) for dim in input_shape)}",
        ],
        check=True,
    )

    return output_onnx_path


def export_onnx_ocr_to_coreml(
    onnx_model_path: Path = DEFAULT_ONNX_MODEL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    input_name: str = DEFAULT_INPUT_NAME,
    minimum_deployment_target: str = "iOS15",
    compute_precision: str = "float16",
) -> Path:
    """
    Chuyen ONNX OCR static sang Core ML .mlpackage.

    Output Core ML van tra ve logits/sequence cua OCR. App macOS/iOS can dung
    CTC decoder voi dictionary tu `data/model_trained/orc_vino/ppocrv5_latin_dict.txt`
    de bien output thanh chuoi bien so.
    """
    onnx_model_path = onnx_model_path.resolve()
    output_dir = output_dir.resolve()
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"Khong tim thay ONNX model: {onnx_model_path}")

    _ensure_module_installed(
        "onnx",
        ".venv/bin/python -m pip install 'onnx>=1.16,<2'",
    )
    _ensure_module_installed(
        "onnx2torch",
        ".venv/bin/python -m pip install 'onnx2torch>=1.5,<2'",
    )
    _ensure_module_installed(
        "coremltools",
        ".venv/bin/python -m pip install 'coremltools>=9.0'",
    )

    import coremltools as ct
    import onnx
    import torch
    from onnx2torch import convert

    if compute_precision not in {"float16", "float32"}:
        raise ValueError("compute_precision chi nhan `float16` hoac `float32`.")

    onnx_model = onnx.load(str(onnx_model_path))
    torch_model = convert(onnx_model).eval()

    dummy_input = torch.randn(*input_shape)
    traced_model = torch.jit.trace(torch_model, dummy_input)

    target = _target_from_name(ct, minimum_deployment_target)
    precision = ct.precision.FLOAT16 if compute_precision == "float16" else ct.precision.FLOAT32
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / f"{model_name}.mlpackage"

    if package_path.exists():
        if package_path.is_dir():
            shutil.rmtree(package_path)
        else:
            package_path.unlink()

    mlmodel = ct.convert(
        traced_model,
        source="pytorch",
        inputs=[ct.TensorType(name=input_name, shape=tuple(input_shape))],
        minimum_deployment_target=target,
        convert_to="mlprogram",
        compute_precision=precision,
    )
    mlmodel.short_description = "PaddleOCR latin PP-OCRv5 recognition model exported to Core ML."
    mlmodel.input_description[input_name] = (
        "Anh bien so da resize/pad, tensor float32 dang NCHW, shape 1x3x48x320."
    )
    mlmodel.save(str(package_path))

    return package_path


def export_paddle_ocr_to_coreml(
    paddle_model_dir: Path = DEFAULT_PADDLE_MODEL_DIR,
    onnx_model_path: Path = DEFAULT_ONNX_MODEL_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rebuild_onnx: bool = False,
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    input_name: str = DEFAULT_INPUT_NAME,
    model_name: str = DEFAULT_MODEL_NAME,
    minimum_deployment_target: str = "iOS15",
    compute_precision: str = "float16",
) -> Path:
    """
    Ham tong hop de chuyen PaddleOCR sang Core ML.

    Mac dinh ham se dung ONNX static da co san trong `data/model_trained`. Dat
    rebuild_onnx=True neu can tao lai ONNX tu thu muc Paddle goc.
    """
    if rebuild_onnx:
        export_paddle_ocr_to_onnx(
            paddle_model_dir=paddle_model_dir,
            output_onnx_path=onnx_model_path,
            input_shape=input_shape,
            input_name=input_name,
        )

    return export_onnx_ocr_to_coreml(
        onnx_model_path=onnx_model_path,
        output_dir=output_dir,
        model_name=model_name,
        input_shape=input_shape,
        input_name=input_name,
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=compute_precision,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chuyen PaddleOCR recognition model sang Core ML .mlpackage cho macOS/iOS."
    )
    parser.add_argument(
        "--paddle-model-dir",
        type=Path,
        default=DEFAULT_PADDLE_MODEL_DIR,
        help=f"Thu muc PaddleOCR inference. Mac dinh: {DEFAULT_PADDLE_MODEL_DIR}",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=DEFAULT_ONNX_MODEL_PATH,
        help=f"File ONNX static. Mac dinh: {DEFAULT_ONNX_MODEL_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Thu muc chua .mlpackage dau ra. Mac dinh: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--rebuild-onnx",
        action="store_true",
        help="Tao lai ONNX static tu PaddleOCR truoc khi export Core ML.",
    )
    parser.add_argument(
        "--input-shape",
        type=_parse_input_shape,
        default=DEFAULT_INPUT_SHAPE,
        help="Shape input NCHW cho OCR, mac dinh: 1,3,48,320.",
    )
    parser.add_argument(
        "--input-name",
        default=DEFAULT_INPUT_NAME,
        help="Ten input tensor, mac dinh la `x` theo PaddleOCR/OpenVINO IR hien tai.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Ten package Core ML dau ra.",
    )
    parser.add_argument(
        "--minimum-deployment-target",
        default="iOS15",
        help="Deployment target cua Core ML, vi du: iOS15, iOS16, macOS12.",
    )
    parser.add_argument(
        "--compute-precision",
        choices=("float16", "float32"),
        default="float16",
        help="Do chinh xac khi convert sang ML Program. Mac dinh: float16.",
    )
    return parser


def main() -> None:
    if platform.system() != "Darwin":
        print("Canh bao: Core ML nen duoc export/kiem thu tren macOS de dam bao tuong thich iOS.")

    args = build_arg_parser().parse_args()
    output_path = export_paddle_ocr_to_coreml(
        paddle_model_dir=args.paddle_model_dir,
        onnx_model_path=args.onnx_model,
        output_dir=args.output_dir,
        rebuild_onnx=args.rebuild_onnx,
        input_shape=args.input_shape,
        input_name=args.input_name,
        model_name=args.model_name,
        minimum_deployment_target=args.minimum_deployment_target,
        compute_precision=args.compute_precision,
    )

    print("Export PaddleOCR sang Core ML hoan tat.")
    print(f"File dau ra: {output_path}")
    print("Luu y: app iOS can tien xu ly anh ve 1x3x48x320 va decode output bang CTC.")


if __name__ == "__main__":
    main()
