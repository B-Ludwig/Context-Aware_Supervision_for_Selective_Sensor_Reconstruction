"""Acquire provider recordings without publishing or replacing local inputs.

FourTank is pinned to its provider commit and Git blob. HAI-CPPS recordings
come from IEEE DataPort, not the simulation-code repository. A supplied HAI
ZIP or direct HTTPS archive URL can be imported into the expected layout.
SWaT stays a user-supplied acquisition under the provider's access process.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen
import zipfile

FOURTANK_REPOSITORY = "https://github.com/thomasbierweiler/FaultsOf4-TankBatchProcess"
FOURTANK_COMMIT = "64f3904dc101713062b9f8d0ecdf787d17bed702"
FOURTANK_ARCHIVE = "SmA-Four-Tank-Batch-Process_V2.zip"
FOURTANK_CSV = "SmA-Four-Tank-Batch-Process_V2.csv"
FOURTANK_URL = f"https://raw.githubusercontent.com/thomasbierweiler/FaultsOf4-TankBatchProcess/{FOURTANK_COMMIT}/{FOURTANK_ARCHIVE}"
FOURTANK_BLOB_SHA1 = "a80b5c9e64e39fa5f9b412c4f36e68710b5dec16"
FOURTANK_ARCHIVE_BYTES = 29933112
HAI_REPOSITORY = "https://github.com/j-ehrhardt/hai-cpps-benchmark"
HAI_DATAPORT = "https://ieee-dataport.org/open-access/hai-cpps-hamburg-ai-benchmark-cyber-physical-production-systems-v2"
SWAT_PROVIDER = "https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/"
MAX_DOWNLOAD_BYTES = 4 * 1024**3
MAX_EXTRACT_BYTES = 8 * 1024**3
MAX_MEMBERS = 100000
CHUNK = 1024 * 1024


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _public_url(url):
    parts = urlsplit(str(url))
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        raise ValueError("Dataset archive URL must use HTTPS without embedded credentials")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _download(url, target, *, maximum=MAX_DOWNLOAD_BYTES):
    """Stream one archive to a temporary file, with size and HTTPS checks."""
    _public_url(url)
    request = Request(str(url), headers={"User-Agent": "safeprocess-reconstruction/0.1"})
    total = 0
    try:
        with urlopen(request, timeout=60) as response, Path(target).open("xb") as handle:
            if hasattr(response, "geturl"):
                _public_url(response.geturl())
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > maximum:
                raise ValueError("Dataset archive exceeds the download size limit")
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                total += len(block)
                if total > maximum:
                    raise ValueError("Dataset archive exceeds the download size limit")
                handle.write(block)
    except OSError as exc:
        code=getattr(exc,"code",None)
        suffix=f" (HTTP {code})" if isinstance(code,int) else ""
        raise OSError("Provider archive download failed"+suffix+". Retry or supply a local archive.") from None
    return total


def _git_blob(path):
    path = Path(path)
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _members(archive):
    members = archive.infolist()
    if len(members) > MAX_MEMBERS:
        raise ValueError("Dataset archive contains too many entries")
    size = 0
    names = set()
    for info in members:
        name = info.filename
        path = PurePosixPath(name)
        if ("\\" in name or path.is_absolute() or ".." in path.parts or
                any(":" in part for part in path.parts) or "\x00" in name):
            raise ValueError("Unsafe path in dataset archive")
        if info.flag_bits & 1:
            raise ValueError("Encrypted dataset archives are not supported")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError("Symlinks are not accepted in dataset archives")
        if name in names:
            raise ValueError("Duplicate path in dataset archive")
        names.add(name)
        size += info.file_size
        if size > MAX_EXTRACT_BYTES:
            raise ValueError("Dataset archive exceeds the extraction size limit")
    return members


def _extract(archive, info, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with archive.open(info) as source, target.open("xb") as destination:
        while True:
            block = source.read(CHUNK)
            if not block:
                break
            written += len(block)
            if written > info.file_size or written > MAX_EXTRACT_BYTES:
                raise ValueError("Dataset archive member exceeds its declared size")
            destination.write(block)
    if written != info.file_size:
        raise ValueError("Incomplete dataset archive member")


def _install(staged, destination):
    """Validate all collisions first, then atomically install each absent file."""
    pairs = [(p, destination / p.relative_to(staged))
             for p in sorted(staged.rglob("*")) if p.is_file()]
    for source, target in pairs:
        # Reject existing symbolic links anywhere below the supplied root.
        current = target
        while current != destination.parent:
            if current.is_symlink():
                raise ValueError("Refusing to write through a symbolic link")
            if current != target and current.exists() and not current.is_dir():
                raise ValueError("Existing path component is not a directory")
            if current == destination:
                break
            current = current.parent
        if target.exists() and (not target.is_file() or _sha256(source) != _sha256(target)):
            raise ValueError("Existing recording differs; acquisition will not overwrite it: " + target.name)
    installed = []
    for source, target in pairs:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            # Hard-link creation is atomic and cannot replace a concurrent file.
            try:
                os.link(source, target)
            except FileExistsError:
                if _sha256(source) != _sha256(target):
                    raise ValueError("Concurrent recording differs; refusing overwrite")
        installed.append(target)
    return installed


def _unit_numbers(datasets):
    numbers, four_tank, swat = set(), False, False
    for value in datasets:
        name = str(value).lower().replace("-", "_")
        if name in ("hai", "hai_cpps"):
            numbers.update(range(1, 11))
        elif re.fullmatch(r"hai_ds(?:10|[1-9])", name):
            numbers.add(int(name.removeprefix("hai_ds")))
        elif name in ("four_tank", "fourtank", "siemens"):
            four_tank = True
        elif name == "swat":
            swat = True
        elif name != "demo":
            raise ValueError("Unknown acquisition dataset: " + str(value))
    return numbers, four_tank, swat


def _child_or_root(root, names):
    if root.is_dir():
        children = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
        for name in names:
            if name.lower() in children:
                return children[name.lower()]
    return root


def _hai_base(root):
    candidate = _child_or_root(root, ("HAI", "HAI-CPPS", "hai_cpps"))
    if (candidate / "Datasets").is_dir():
        return candidate / "Datasets"
    if (candidate / "continuous-normal").is_dir():
        return candidate
    return root / "HAI" / "Datasets"


def _hai_existing(base, numbers):
    files = []
    for number in sorted(numbers):
        normal = [p for p in (base / "continuous-normal").glob("*/*.csv")
                  if re.match(rf"ds{number}(?:_|$)", p.parent.name)]
        if len(normal) != 1:
            return []
        files += normal
        files += [p for p in (base / "continuous-anomalies").glob("*/*.csv")
                  if re.match(rf"ds{number}(?:_|$)", p.parent.name)]
    for source in files:
        if not (source.parent / "sim_setup.json").is_file():
            return []
    return sorted(set(files + [p.parent / "sim_setup.json" for p in files]))


def _import_hai(archive_path, staging, numbers):
    with zipfile.ZipFile(archive_path) as archive:
        selected = {}
        for info in _members(archive):
            parts = PurePosixPath(info.filename).parts
            if info.is_dir():
                continue
            positions = [i for i, part in enumerate(parts)
                         if part in ("continuous-normal", "continuous-anomalies")]
            if not positions:
                continue
            tail = parts[positions[-1]:]
            if len(tail) != 3:
                continue
            category, scenario, filename = tail
            match = re.match(r"ds(10|[1-9])(?:_|$)", scenario)
            if not match or int(match.group(1)) not in numbers:
                continue
            if not (filename.endswith(".csv") or filename == "sim_setup.json"):
                continue
            relative = Path(category, scenario, filename)
            if relative in selected:
                raise ValueError("HAI archive contains duplicate recording destinations")
            selected[relative] = info
        if not selected:
            raise ValueError("HAI ZIP has no compatible continuous-normal/anomalies recordings for the selected configurations")
        for relative, info in selected.items():
            _extract(archive, info, staging / relative)
    if not _hai_existing(staging, numbers):
        raise ValueError("HAI ZIP must contain one normal CSV and neighboring sim_setup.json for each selected configuration")


def _file_rows(root, files):
    result = []
    for path in sorted(set(files)):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        result.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return result


def acquire_datasets(data_root, datasets, *, hai_archive=None, hai_url=None):
    """Download/import missing inputs and return a portable acquisition receipt.

    A manual_required provider does not invalidate other completed acquisitions.
    No external request is made for an already present dataset or for SWaT.
    HAI archive import is explicit; the benchmark is never regenerated or
    silently replaced by the simulation repository's newest release.
    """
    if hai_archive is not None and hai_url is not None:
        raise ValueError("Choose either --hai-archive or --hai-url")
    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    numbers, wants_four, wants_swat = _unit_numbers(datasets)
    providers, all_files = {}, []
    if wants_four:
        base = _child_or_root(root, ("FourTank", "four_tank", "four-tank"))
        existing = next((p for p in (base / FOURTANK_CSV, base / "raw" / FOURTANK_CSV)
                         if p.is_file()), None)
        if existing is not None:
            files = [existing]
            providers["four_tank"] = dict(status="available",acquisition="existing",repository=FOURTANK_REPOSITORY)
        else:
            destination = root / "FourTank" / "raw"
            with tempfile.TemporaryDirectory(prefix=".safeprocess-four-tank-", dir=root) as temp:
                temp = Path(temp)
                archive_path = temp / "provider.zip"
                size = _download(FOURTANK_URL, archive_path, maximum=FOURTANK_ARCHIVE_BYTES)
                if size != FOURTANK_ARCHIVE_BYTES or _git_blob(archive_path) != FOURTANK_BLOB_SHA1:
                    raise ValueError("FourTank provider archive does not match the pinned Git blob")
                with zipfile.ZipFile(archive_path) as archive:
                    matches = [i for i in _members(archive) if not i.is_dir() and PurePosixPath(i.filename).name == FOURTANK_CSV]
                    if len(matches) != 1:
                        raise ValueError("FourTank archive must contain exactly one expected measurement CSV")
                    _extract(archive,matches[0],temp/"extracted"/FOURTANK_CSV)
                files = _install(temp/"extracted",destination)
                providers["four_tank"] = dict(status="available",acquisition="downloaded",repository=FOURTANK_REPOSITORY,
                    commit=FOURTANK_COMMIT,url=FOURTANK_URL,archive_git_blob_sha1=FOURTANK_BLOB_SHA1,
                    archive_sha256=_sha256(archive_path),archive_bytes=size)
        providers["four_tank"]["files"] = _file_rows(root,files)
        all_files += files
    if numbers:
        destination = _hai_base(root)
        existing = _hai_existing(destination,numbers)
        if existing:
            files = existing
            providers["hai"] = dict(status="available",acquisition="existing",repository=HAI_REPOSITORY,recording_provider=HAI_DATAPORT)
        elif hai_archive is not None or hai_url is not None:
            with tempfile.TemporaryDirectory(prefix=".safeprocess-hai-",dir=root) as temp:
                temp=Path(temp)
                archive_path=Path(hai_archive).expanduser().resolve() if hai_archive is not None else temp/"provider.zip"
                if hai_url is not None:
                    _download(hai_url,archive_path)
                if not archive_path.is_file():
                    raise FileNotFoundError("HAI archive file does not exist")
                try:
                    _import_hai(archive_path,temp/"extracted",numbers)
                except zipfile.BadZipFile:
                    raise ValueError("HAI provider response is not a valid ZIP archive. Supply the recording ZIP rather than a login or download page.") from None
                files=_install(temp/"extracted",destination)
                providers["hai"]=dict(status="available",acquisition="archive_import",repository=HAI_REPOSITORY,
                    recording_provider=HAI_DATAPORT,archive_sha256=_sha256(archive_path),archive_bytes=archive_path.stat().st_size,
                    selected_configurations=[f"hai_ds{i}" for i in sorted(numbers)])
                if hai_url is not None:
                    providers["hai"]["archive_url"]=_public_url(hai_url)
        else:
            files=[]
            providers["hai"]=dict(status="manual_required",repository=HAI_REPOSITORY,recording_provider=HAI_DATAPORT,
                reason="The GitHub repository supplies simulation code. Obtain the compatible recording ZIP from IEEE DataPort, then use --hai-archive or a provider-issued direct --hai-url. The newest simulation release is not a substitute for the paper recordings.")
        providers["hai"]["files"]=_file_rows(root,files)
        all_files += files
    if wants_swat:
        base=_child_or_root(root,("SWaT","swat"))
        locations=[base,base/"Physical",base/"raw"/"Physical",base/"raw",base/"cache"/"physical"]
        stems=(("SWaT_Dataset_Normal_v1","SWaT_Dataset_Normal_v0"),("SWaT_Dataset_Attack_v0",))
        found=[]
        for aliases in stems:
            candidate=next((folder/(name+ext) for folder in locations for name in aliases
                            for ext in (".parquet",".xlsx",".xlsm") if (folder/(name+ext)).is_file()),None)
            if candidate is not None:
                found.append(candidate)
        available=len(found)==2
        providers["swat"]=dict(status="available" if available else "manual_required",acquisition="existing" if available else "user_supplied",
            provider=SWAT_PROVIDER,reason="Supply the physical normal and attack recordings obtained through SUTD iTrust. No SWaT download or credentials are requested.",
            files=_file_rows(root,found))
        all_files+=found
    for provider in providers.values():
        if provider["status"] == "manual_required":
            provider["message"] = provider["reason"]
            provider["provider_url"] = provider.get("recording_provider",provider.get("provider"))
    result=dict(schema="safeprocess-acquisition-v1",complete=all(p["status"]=="available" for p in providers.values()),
                providers=providers,files=_file_rows(root,all_files))
    receipt=root/".safeprocess-acquisition.json"
    temporary=receipt.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    temporary.replace(receipt)
    return result
