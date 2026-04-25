import subprocess

from pathlib import Path


def doc2docx(doc_path: str, outdir: str | None = None) -> str:
    file_path = Path(doc_path)
    _outdir = Path(outdir) if outdir is not None else file_path.parent

    subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "docx",
            doc_path,
            "--outdir",
            str(_outdir),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return str(_outdir / (file_path.stem + ".docx"))
