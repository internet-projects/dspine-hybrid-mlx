"""Pack an oMLX source tree whose native kernels are already built into one wheel.

The wheel (``dspine-omlx-kernels``) holds the unmodified ``omlx`` Python package of jundot/omlx
(Apache-2.0) at one commit, with the compiled kernel extensions and Metal libraries, and declares
only ``mlx`` as a dependency, so ``pip install`` needs no compiler, CMake or Metal toolchain and
pulls none of oMLX's server dependencies. It is what ``dspine-hybrid-mlx[fast]`` installs.
Do not install it next to a real oMLX: both provide the ``omlx`` package.

    cd omlx && OMLX_WITH_CUSTOM_KERNEL=1 python setup.py build_ext --inplace
    python packaging/build_kernel_wheel.py --omlx ./omlx-src --commit <sha> --out dist/

Run it with the same Python that built the extensions: the wheel is tagged for that version.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
import sysconfig
import zipfile
from pathlib import Path

NAME = "dspine_omlx_kernels"
KEEP = {".py", ".so", ".metallib"}  # no eval or calibration data: the kit never reads it


def _record_line(arc: str, data: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{arc},sha256={digest},{len(data)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--omlx", required=True, help="oMLX checkout with build_ext --inplace done")
    p.add_argument("--commit", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="0.1.0")
    p.add_argument("--mlx", default="0.32.4", help="the mlx version the extensions were built against")
    p.add_argument("--macos", default="15_0")
    a = p.parse_args()
    root = Path(a.omlx).resolve()
    pkg = root / "omlx"
    soabi = sysconfig.get_config_var("SOABI")  # e.g. cpython-312-darwin
    exts = sorted(pkg.glob("custom_kernels/*/_ext*.so"))
    libs = sorted(pkg.glob("custom_kernels/**/*.metallib"))
    if not exts or not libs:
        sys.exit("no built kernels under omlx/custom_kernels (run build_ext --inplace first)")
    if not any(soabi in e.name for e in exts if "quant_matmul" in str(e)):
        sys.exit(f"the quant_matmul extension was not built by this Python ({soabi}): {[e.name for e in exts]}")
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    tag = f"{py}-{py}-macosx_{a.macos}_arm64"
    info = f"{NAME}-{a.version}.dist-info"
    files: list[tuple[str, bytes]] = []
    for f in sorted(pkg.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or "csrc" in f.parts or f.suffix not in KEEP:
            continue
        if f.suffix == ".so" and soabi not in f.name:
            continue  # an extension left by another Python version
        files.append((f.relative_to(root).as_posix(), f.read_bytes()))
    for lic in ("LICENSE", "NOTICE"):
        if (root / lic).is_file():
            files.append((f"{info}/licenses/{lic}", (root / lic).read_bytes()))
    for f in sorted(pkg.rglob("LICENSE*")):
        files.append((f"{info}/licenses/{f.relative_to(root).as_posix()}", f.read_bytes()))
    meta = "\n".join([
        "Metadata-Version: 2.1",
        f"Name: {NAME.replace('_', '-')}",
        f"Version: {a.version}",
        "Summary: Prebuilt oMLX Metal kernels for dspine-hybrid-mlx's fast mode",
        "Home-page: https://github.com/internet-projects/dspine-hybrid-mlx",
        "License: Apache-2.0",
        f"Requires-Python: =={sys.version_info.major}.{sys.version_info.minor}.*",
        f"Requires-Dist: mlx=={a.mlx}",
        "Description-Content-Type: text/markdown",
        "",
        f"The unmodified `omlx` Python package of https://github.com/jundot/omlx (Apache-2.0) at commit `{a.commit}`,",
        f"with its custom Metal kernel extensions built against mlx {a.mlx}. Repackaged by NetProjects so that",
        "`pip install \"dspine-hybrid-mlx[fast]\"` needs no compiler or Metal toolchain. It declares none of oMLX's",
        "server dependencies and is not a way to run oMLX; for that, install oMLX itself. Do not install both.",
        "Licence and notices of oMLX are in this wheel's `licenses` folder. Not affiliated with the oMLX project.",
        "",
    ]).encode()
    wheel = f"Wheel-Version: 1.0\nGenerator: dspine-hybrid-mlx packaging/build_kernel_wheel.py\nRoot-Is-Purelib: false\nTag: {tag}\n".encode()
    files += [(f"{info}/METADATA", meta), (f"{info}/WHEEL", wheel), (f"{info}/top_level.txt", b"omlx\n")]
    record = "\n".join([_record_line(arc, data) for arc, data in files] + [f"{info}/RECORD,,", ""]).encode()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{NAME}-{a.version}-{tag}.whl"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, data in files:
            z.writestr(zipfile.ZipInfo(arc, date_time=(2026, 11, 21, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
        z.writestr(zipfile.ZipInfo(f"{info}/RECORD", date_time=(2026, 11, 21, 0, 0, 0)), record, zipfile.ZIP_DEFLATED)
    n_so = sum(1 for arc, _ in files if arc.endswith(".so"))
    n_lib = sum(1 for arc, _ in files if arc.endswith(".metallib"))
    print(f"{path.name}: {len(files) + 1} files, {n_so} extensions, {n_lib} metal libraries, "
          f"{path.stat().st_size / 1e6:.1f} MB, sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
