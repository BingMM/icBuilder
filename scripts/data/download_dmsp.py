from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import deque
import re
import time

import requests
from bs4 import BeautifulSoup
from cdflib import CDF
from tqdm import tqdm


#BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/dmsp/"
BASE_URL = "https://cdaweb.gsfc.nasa.gov/pub/data/dmsp/"
OUT_DIR = Path("/media/bing/LaCie/dmsp_ssj")
SSJ_PRODUCT = "/ssj/precipitating-electrons-ions/"

YEARS = set(range(2000, 2004))
RETRY_DELAYS = (2, 5, 10, 20)

session = requests.Session()


def read_directory(url):
    """Read one archive directory, retrying temporary server failures."""

    n_attempts = len(RETRY_DELAYS) + 1

    for attempt in range(n_attempts):
        try:
            r = session.get(url, timeout=60)
            r.raise_for_status()
            return r.text
        except requests.RequestException as error:
            if attempt == n_attempts - 1:
                raise

            delay = RETRY_DELAYS[attempt]
            tqdm.write(f"Retrying directory in {delay} s: {error}")
            time.sleep(delay)


def find_files(base_url):
    """Find all files belonging to YEARS without directory loops."""

    files = []
    visited = set()

    # (directory URL, detected year)
    queue = deque([(base_url, None)])

    bar = tqdm(
        total=1,
        desc="Scanning directories",
        unit="dir",
    )

    while queue:
        url, year = queue.popleft()

        if url in visited:
            bar.update(1)
            continue

        visited.add(url)

        html = read_directory(url)
        soup = BeautifulSoup(html, "html.parser")

        for link in soup.find_all("a"):
            href = link.get("href")

            if not href:
                continue

            if href.startswith("?") or href.startswith("#"):
                continue

            full_url = urljoin(url, href)

            # Strip query strings and fragments
            parsed = urlparse(full_url)
            full_url = parsed._replace(
                query="",
                fragment="",
            ).geturl()

            # Stay inside the DMSP archive
            if not full_url.startswith(BASE_URL):
                continue

            # Do not follow links back up the directory tree
            if not full_url.startswith(url):
                continue

            # Do not revisit the current directory
            if full_url == url:
                continue

            if full_url.endswith("/"):
                relative = (
                    urlparse(full_url)
                    .path
                    .split("/pub/data/dmsp/", 1)[-1]
                    .strip("/")
                    .split("/")
                )

                # Below each satellite, follow only the SSJ particle product.
                if len(relative) >= 2 and relative[1].lower() != "ssj":
                    continue
                if (
                    len(relative) >= 3
                    and relative[2].lower() != "precipitating-electrons-ions"
                ):
                    continue

                dirname = (
                    urlparse(full_url)
                    .path
                    .rstrip("/")
                    .split("/")[-1]
                )

                new_year = year

                # Detect year directories
                if re.fullmatch(r"\d{4}", dirname):
                    candidate_year = int(dirname)

                    if candidate_year not in YEARS:
                        continue

                    new_year = candidate_year

                if full_url not in visited:
                    queue.append((full_url, new_year))

                    bar.total += 1
                    bar.refresh()

            elif year in YEARS and SSJ_PRODUCT in parsed.path.lower():
                files.append(full_url)

        bar.update(1)

    bar.close()

    # Just in case the archive exposes the same file more than once
    return sorted(set(files))


def valid_cdf(path):
    """Check that a downloaded CDF contains a complete day of SSJ data."""

    if not path.exists() or path.stat().st_size == 0:
        return False

    cdf = None

    try:
        cdf = CDF(path)
        n_time = len(cdf.varget("Epoch"))
        n_last_variable = len(cdf.varget("ION_AVG_ENERGY_STD"))
    except Exception:
        return False
    finally:
        if cdf is not None:
            cdf.close()

    return n_time == 86400 and n_last_variable == n_time


def download_file(url, path):
    """Download one file, retrying temporary connection failures."""

    if valid_cdf(path):
        return "skipped"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    partial_path = path.with_name(path.name + ".partial")
    n_attempts = len(RETRY_DELAYS) + 1

    for attempt in range(n_attempts):
        try:
            with session.get(
                url,
                stream=True,
                timeout=120,
            ) as r:
                r.raise_for_status()

                total = int(
                    r.headers.get("content-length", 0)
                )
                written = 0

                with open(partial_path, "wb") as f:
                    with tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=path.name,
                        leave=False,
                    ) as bar:

                        for chunk in r.iter_content(
                            chunk_size=1024 * 1024
                        ):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                bar.update(len(chunk))

            if total and written != total:
                raise OSError(
                    f"expected {total} bytes but received {written}"
                )

            if not valid_cdf(partial_path):
                raise OSError("downloaded file is not a complete SSJ CDF")

            partial_path.replace(path)
            return "downloaded"

        except (requests.RequestException, OSError) as error:
            if attempt == n_attempts - 1:
                tqdm.write(f"FAILED: {path.name}: {error}")
                return "failed"

            delay = RETRY_DELAYS[attempt]
            tqdm.write(
                f"Retrying {path.name} in {delay} s: {error}"
            )
            time.sleep(delay)


def main():

    print("Searching for SSJ precipitating-electron/ion files from 2000–2003...")

    files = find_files(BASE_URL)

    print(f"Found {len(files):,} files.")

    downloaded = 0
    skipped = 0
    failed = 0
    failed_paths = []

    for url in tqdm(
        files,
        desc="Overall",
        unit="file",
    ):
        relative = (
            urlparse(url)
            .path
            .split("/pub/data/dmsp/", 1)[1]
        )

        path = OUT_DIR / relative

        result = download_file(url, path)

        if result == "downloaded":
            downloaded += 1
        elif result == "skipped":
            skipped += 1
        else:
            failed += 1
            failed_paths.append(path)

    print()
    print("Done.")
    print(f"Downloaded: {downloaded:,}")
    print(f"Valid files skipped: {skipped:,}")
    print(f"Failed after retries: {failed:,}")

    if failed_paths:
        print("Run the script again to retry these files:")
        for path in failed_paths:
            print(path)


if __name__ == "__main__":
    main()
