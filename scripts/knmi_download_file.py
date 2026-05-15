import os
from pathlib import Path

import requests


DATASET = "harmonie_arome_cy43_p1"
VERSION = "1.0"
FILENAME = "HARM43_V1_P1_2026051504.tar"

OUT_DIR = Path("data/raw/knmi/harmonie_arome_cy43_p1")
OUT_PATH = OUT_DIR / FILENAME


def get_download_url(dataset: str, version: str, filename: str) -> str:
    api_key = os.getenv("KNMI_API_KEY")
    if not api_key:
        raise RuntimeError("KNMI_API_KEY is not set")

    url = (
        "https://api.dataplatform.knmi.nl/open-data/v1/"
        f"datasets/{dataset}/versions/{version}/files/{filename}/url"
    )

    response = requests.get(
        url,
        headers={"Authorization": api_key},
        timeout=30,
    )
    print("URL status:", response.status_code)
    print(response.text[:500])
    response.raise_for_status()

    return response.json()["temporaryDownloadUrl"]


def download_file(download_url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(download_url, stream=True, timeout=300) as response:
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        downloaded = 0

        with out_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue

                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    pct = 100 * downloaded / total
                    print(
                        f"\rDownloaded {downloaded / 1e6:.1f} MB "
                        f"of {total / 1e6:.1f} MB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

    print(f"\nSaved to: {out_path}")


def main() -> None:
    if OUT_PATH.exists():
        print(f"File already exists: {OUT_PATH}")
        return

    download_url = get_download_url(DATASET, VERSION, FILENAME)
    download_file(download_url, OUT_PATH)


if __name__ == "__main__":
    main()
