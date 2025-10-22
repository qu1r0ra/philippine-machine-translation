import tarfile
import zipfile
from pathlib import Path


def extract_archives(
    source_dir: Path,
    target_dir: Path | None = None,
    exts: tuple[str, ...] = (".zip", ".tar", ".tar.gz", ".tgz"),
    overwrite: bool = False,
) -> None:
    """
    Extract all archive files from a source directory into subfolders.

    Parameters
    ----------
    source_dir : Path
        Directory containing archive files to extract.
    target_dir : Path | None, optional
        Root directory where files will be extracted (defaults to `source_dir`).
    exts : tuple[str, ...], optional
        Supported archive extensions.
    overwrite : bool, optional
        Whether to overwrite existing extracted folders (default: False).

    Notes
    -----
    - Each archive is extracted into a subfolder named after its stem.
    - Skips already extracted folders unless `overwrite=True`.
    - Supports `.zip`, `.tar`, `.tar.gz`, `.tgz` by default.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir) if target_dir else source_dir

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    archives = [
        f
        for f in source_dir.iterdir()
        if f.suffix.lower() in exts or any(f.name.endswith(e) for e in exts)
    ]

    if not archives:
        print(f"No archive files found in {source_dir}.")
        return

    print(f"Found {len(archives)} archive(s) in '{source_dir}':")

    for archive in archives:
        extract_path = target_dir / archive.stem

        # Handle existing directories
        if extract_path.exists():
            if overwrite:
                print(f"Overwriting existing folder '{extract_path.name}' ...")
            else:
                print(f"'{extract_path.name}' already exists — skipping.")
                continue

        print(f"Extracting '{archive.name}' → '{extract_path}/' ...")

        # ZIP archives
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(extract_path)

        # TAR archives
        elif archive.suffix in (".tar", ".gz", ".tgz"):
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(extract_path)

        else:
            print(f"Skipping unsupported archive: {archive.name}")
            continue

        print(f"Done extracting '{archive.name}'")

    print("All archives processed.")
