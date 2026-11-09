"""Provider acquisition contracts using tiny archives and mocked networking."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest
from safeprocess import acquisition as acq


def archive_bytes(files):
    memory=BytesIO()
    with zipfile.ZipFile(memory,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path,value in files.items():
            z.writestr(path,value)
    return memory.getvalue()


def hai_archive(tmp_path, *, normal=b"sensor\n1\n", extras=None):
    files={
        "release/Datasets/continuous-normal/ds1/normal.csv":normal,
        "release/Datasets/continuous-normal/ds1/sim_setup.json":b'{"model":{"modules":{}}}',
        "release/Datasets/continuous-anomalies/ds1_pump50/anomaly.csv":b"sensor\n2\n",
        "release/Datasets/continuous-anomalies/ds1_pump50/sim_setup.json":b'{"model":{"modules":{}}}',
        "release/Datasets/continuous-normal/ds2/normal.csv":b"other\n3\n",
        "release/Datasets/continuous-normal/ds2/sim_setup.json":b'{}',
        "release/code/network.pcap":b"not needed",
    }
    files.update(extras or {})
    path=tmp_path/"hai.zip";path.write_bytes(archive_bytes(files));return path


def no_network(*args,**kwargs):
    raise AssertionError("Unexpected external download")


def test_hai_local_import_selects_configurations_and_is_idempotent(tmp_path,monkeypatch):
    monkeypatch.setattr(acq,"_download",no_network)
    archive=hai_archive(tmp_path)
    root=tmp_path/"data"
    first=acq.acquire_datasets(root,["hai_ds1"],hai_archive=archive)
    assert first["complete"]
    assert first["providers"]["hai"]["acquisition"]=="archive_import"
    output=root/"HAI/Datasets"
    assert (output/"continuous-normal/ds1/normal.csv").read_bytes()==b"sensor\n1\n"
    assert not (output/"continuous-normal/ds2").exists()
    assert not any(root.rglob("*.pcap"))
    before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file() and p.suffix!=".json"}
    second=acq.acquire_datasets(root,["hai_ds1"])
    assert second["providers"]["hai"]["acquisition"]=="existing"
    assert before=={str(p.relative_to(root)):p.read_bytes() for p in root.rglob("*") if p.is_file() and p.suffix!=".json"}


def test_hai_collisions_are_checked_before_any_file_is_installed(tmp_path):
    archive=hai_archive(tmp_path)
    root=tmp_path/"data"
    normal=root/"HAI/Datasets/continuous-normal/ds1/normal.csv"
    normal.parent.mkdir(parents=True);normal.write_bytes(b"user original")
    with pytest.raises(ValueError,match="will not overwrite"):
        acq.acquire_datasets(root,["hai_ds1"],hai_archive=archive)
    assert normal.read_bytes()==b"user original"
    assert not (root/"HAI/Datasets/continuous-anomalies").exists()


@pytest.mark.parametrize("unsafe",["../escape.csv","/absolute.csv","C:/escape.csv","folder\\escape.csv"])
def test_hai_archive_rejects_unsafe_paths_even_when_unselected(tmp_path,unsafe):
    archive=hai_archive(tmp_path,extras={unsafe:b"bad"})
    root=tmp_path/"data"
    with pytest.raises(ValueError,match="Unsafe path"):
        acq.acquire_datasets(root,["hai_ds1"],hai_archive=archive)
    assert not (root/"HAI").exists()


def test_archive_symlinks_and_duplicate_destinations_are_rejected(tmp_path):
    archive=hai_archive(tmp_path)
    info=zipfile.ZipInfo("link");info.create_system=3
    info.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(archive,"a") as z:z.writestr(info,"../target")
    with pytest.raises(ValueError,match="Symlinks"):
        acq.acquire_datasets(tmp_path/"data",["hai_ds1"],hai_archive=archive)
    archive=hai_archive(tmp_path,extras={
      "another/continuous-normal/ds1/normal.csv":b"duplicate destination"})
    with pytest.raises(ValueError,match="duplicate recording destinations"):
        acq.acquire_datasets(tmp_path/"data",["hai_ds1"],hai_archive=archive)


def test_manual_hai_and_swat_do_not_make_network_requests(tmp_path,monkeypatch):
    monkeypatch.setattr(acq,"_download",no_network)
    result=acq.acquire_datasets(tmp_path/"data",["hai","swat"])
    assert not result["complete"]
    assert set(result["providers"])=={"hai","swat"}
    assert all(v["status"]=="manual_required" for v in result["providers"].values())
    assert "IEEE DataPort" in result["providers"]["hai"]["message"]
    assert "No SWaT download" in result["providers"]["swat"]["message"]


def test_swat_existing_files_are_reused_without_download(tmp_path,monkeypatch):
    monkeypatch.setattr(acq,"_download",no_network)
    root=tmp_path/"data";folder=root/"SWaT/raw/Physical";folder.mkdir(parents=True)
    for name in ("SWaT_Dataset_Normal_v1.xlsx","SWaT_Dataset_Attack_v0.xlsx"):
        (folder/name).write_bytes(b"user recording")
    result=acq.acquire_datasets(root,["swat"])
    assert result["complete"] and len(result["files"])==2


def fake_fourtank(monkeypatch,data):
    monkeypatch.setattr(acq,"FOURTANK_ARCHIVE_BYTES",len(data))
    monkeypatch.setattr(acq,"FOURTANK_BLOB_SHA1",hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest())
    calls=[]
    def fake(url,target,**kwargs):
        calls.append(url);Path(target).write_bytes(data);return len(data)
    monkeypatch.setattr(acq,"_download",fake)
    return calls


def test_fourtank_download_is_pinned_and_existing_csv_is_not_replaced(tmp_path,monkeypatch):
    csv=b"timestamp;DeviationID ValueY\n2021-01-01;1\n"
    data=archive_bytes({"folder/"+acq.FOURTANK_CSV:csv,"unneeded.pdf":b"not needed"})
    calls=fake_fourtank(monkeypatch,data)
    root=tmp_path/"data"
    first=acq.acquire_datasets(root,["four_tank"])
    assert first["complete"]
    assert acq.FOURTANK_COMMIT in calls[0]
    assert first["providers"]["four_tank"]["archive_sha256"]==hashlib.sha256(data).hexdigest()
    output=root/"FourTank/raw"/acq.FOURTANK_CSV
    assert output.read_bytes()==csv
    assert not any(root.rglob("*.pdf"))
    second=acq.acquire_datasets(root,["four_tank"])
    assert second["providers"]["four_tank"]["acquisition"]=="existing"
    assert len(calls)==1 and output.read_bytes()==csv


def test_fourtank_bad_checksum_installs_nothing(tmp_path,monkeypatch):
    data=archive_bytes({acq.FOURTANK_CSV:b"rows"})
    fake_fourtank(monkeypatch,data)
    monkeypatch.setattr(acq,"FOURTANK_BLOB_SHA1","0"*40)
    root=tmp_path/"data"
    with pytest.raises(ValueError,match="pinned Git blob"):
        acq.acquire_datasets(root,["four_tank"])
    assert not (root/"FourTank").exists()


def test_hai_url_receipt_drops_query_tokens(tmp_path,monkeypatch):
    archive=hai_archive(tmp_path);calls=[]
    def fake(url,target,**kwargs):
        calls.append(url);Path(target).write_bytes(archive.read_bytes());return archive.stat().st_size
    monkeypatch.setattr(acq,"_download",fake)
    url="https://provider.example/recordings.zip?temporary_token=secret#fragment"
    result=acq.acquire_datasets(tmp_path/"data",["hai_ds1"],hai_url=url)
    assert calls==[url]
    assert result["providers"]["hai"]["archive_url"]=="https://provider.example/recordings.zip"
    assert "secret" not in json.dumps(result)
    assert "secret" not in (tmp_path/"data/.safeprocess-acquisition.json").read_text()


def test_download_requires_https_and_enforces_stream_limit(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="HTTPS"):
        acq._download("http://example.test/archive.zip",tmp_path/"data")
    class Response(BytesIO):
        headers={}
        def geturl(self):return "https://example.test/archive.zip"
    monkeypatch.setattr(acq,"urlopen",lambda *args,**kwargs:Response(b"too many bytes"))
    with pytest.raises(ValueError,match="size limit"):
        acq._download("https://example.test/archive.zip",tmp_path/"data",maximum=2)


def test_hai_login_page_and_url_errors_are_actionable_without_token_leaks(tmp_path,monkeypatch):
    page=tmp_path/"login.zip";page.write_text("<html>Please sign in</html>")
    with pytest.raises(ValueError,match="not a valid ZIP"):
        acq.acquire_datasets(tmp_path/"data",["hai_ds1"],hai_archive=page)
    def fail(*args,**kwargs):
        raise OSError("failed https://provider.test/file?temporary_token=secret")
    monkeypatch.setattr(acq,"urlopen",fail)
    with pytest.raises(OSError,match="Provider archive download failed") as error:
        acq._download("https://provider.test/file?temporary_token=secret",tmp_path/"download.zip")
    assert "secret" not in str(error.value)
