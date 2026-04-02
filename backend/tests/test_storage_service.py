from services.storage_service import StorageService


def test_detect_google_drive_folder_link():
    service = StorageService()
    detected = service._detect_provider("https://drive.google.com/drive/folders/abc123?usp=drive_link")

    assert detected["provider"] == "google_drive"
    assert detected["cloud_root_id"] == "abc123"
    assert detected["cloud_link_kind"] == "folder_link"


def test_detect_dropbox_direct_folder_link():
    service = StorageService()
    detected = service._detect_provider("https://www.dropbox.com/home/Projects/AGROSIGNAL")

    assert detected["provider"] == "dropbox"
    assert detected["cloud_root_path"] == "Projects/AGROSIGNAL"
    assert detected["cloud_link_kind"] == "folder_link"


def test_detect_dropbox_shared_link_requires_direct_path():
    service = StorageService()
    detected = service._detect_provider("https://www.dropbox.com/scl/fo/abcxyz123/example?rlkey=demo&dl=0")

    assert detected["provider"] == "dropbox"
    assert detected["cloud_root_path"] is None
    assert detected["cloud_link_kind"] == "shared_link"


def test_detect_yandex_direct_folder_link():
    service = StorageService()
    detected = service._detect_provider("https://disk.yandex.ru/client/disk/AGROSIGNAL/cache")

    assert detected["provider"] == "yandex_disk"
    assert detected["cloud_root_path"] == "AGROSIGNAL/cache"
    assert detected["cloud_link_kind"] == "folder_link"


def test_detect_generic_remote_disk_as_webdav():
    service = StorageService()
    detected = service._detect_provider("https://cloud.example.com/remote.php/dav/files/user/Projects/AGROSIGNAL")

    assert detected["provider"] == "webdav"
    assert detected["cloud_base_url"] == "https://cloud.example.com"
    assert detected["cloud_root_path"] == "remote.php/dav/files/user/Projects/AGROSIGNAL"
    assert detected["cloud_link_kind"] == "webdav_endpoint"


def test_build_link_issue_runtime_for_yandex_public_link():
    service = StorageService()
    runtime = service._build_link_issue_runtime(
        {
            "provider": "yandex_disk",
            "cloud_link_kind": "shared_link",
            "workspace_root": "https://yadi.sk/d/example",
            "cloud_url": "https://yadi.sk/d/example",
        }
    )

    assert runtime["status"] == "link_format_error"
    assert "Яндекс" in runtime["message"]
    assert runtime["auth_prompt"]["title"]
