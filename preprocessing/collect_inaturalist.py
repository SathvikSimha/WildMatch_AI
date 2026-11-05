"""
WildMatch AI - Stage 1: iNaturalist Dataset Collection (Pilot)

This module handles:
1. Querying observations from the official iNaturalist API v1.
2. Filtering for species-level research-grade observations with cc-by and cc-by-sa licenses.
3. Checking photo-level licensing and attribution across all observation photos.
4. Safe extraction and derivation of large image URLs.
5. Pillow-based image validation, decoding, and RGB JPEG conversion.
6. Deduplication and idempotent rerun safety with consistent base_dir pathing.
7. Metadata completeness guarantee (every counted image has verified file and metadata).
8. Generating comprehensive collection reports.
"""

import argparse
import io
import logging
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WildMatch_Collector")

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
INAT_API_BASE_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "WildMatchAI-Pilot/1.0 (Wildlife Research Project; contact: member1@wildmatch.local)"

# Target species with verified iNaturalist Taxon IDs
SPECIES_CONFIG = [
    {
        "common_name": "Tiger",
        "scientific_name": "Panthera tigris",
        "folder": "tiger",
        "taxon_id": 41967,
    },
    {
        "common_name": "Lion",
        "scientific_name": "Panthera leo",
        "folder": "lion",
        "taxon_id": 41964,  # Includes males, lionesses, and cubs
    },
    {
        "common_name": "Asian Elephant",
        "scientific_name": "Elephas maximus",
        "folder": "elephant",
        "taxon_id": 43697,
    },
    {
        "common_name": "Chital",
        "scientific_name": "Axis axis",
        "folder": "chital",
        "taxon_id": 42166,
    },
]

# Strict license whitelist for the pilot
ALLOWED_LICENSES = {"cc-by", "cc-by-sa"}

METADATA_COLUMNS = [
    "image_id",
    "observation_id",
    "photo_id",
    "species",
    "scientific_name",
    "image_path",
    "image_url",
    "photographer",
    "photo_license",
    "attribution",
    "observation_url",
    "observed_on",
    "original_width",
    "original_height",
]

REPORT_COLUMNS = [
    "species",
    "requested_count",
    "successfully_downloaded_count",
    "skipped_count",
    "failed_count",
    "reasons_for_skipping_or_failure",
    "target_achieved",
]


# ---------------------------------------------------------------------------
# Modular Functions
# ---------------------------------------------------------------------------

def create_http_session() -> requests.Session:
    """Create a configured requests session with appropriate headers."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    return session


def resolve_image_path(base_dir: str, rel_path: str) -> str:
    """
    Resolve a stored image path consistently against base_dir.
    Handles standard project-root-relative paths (dataset/raw/...)
    as well as fallback legacy relative paths.
    """
    if not rel_path or not isinstance(rel_path, str):
        return ""
    if os.path.isabs(rel_path):
        return rel_path

    # Standard resolution against base_dir
    direct_path = os.path.normpath(os.path.join(base_dir, rel_path))
    if os.path.isfile(direct_path):
        return direct_path

    # Fallback in case path was stored without 'dataset/' prefix
    legacy_path = os.path.normpath(os.path.join(base_dir, "dataset", rel_path))
    if os.path.isfile(legacy_path):
        return legacy_path

    return direct_path


def derive_large_image_url(photo: Dict[str, Any]) -> Optional[str]:
    """
    Prefer API-provided large_url when available; otherwise safely derive
    the large image URL from the returned photo URL.
    """
    # 1. Prefer API-provided large_url if present
    large_url = photo.get("large_url")
    if large_url and isinstance(large_url, str) and large_url.startswith("http"):
        return large_url

    # 2. Check original photo url
    raw_url = photo.get("url")
    if not raw_url or not isinstance(raw_url, str):
        return None

    # Replace size suffixes (square, medium, small, thumb) with large
    derived_url = re.sub(
        r"/(square|medium|small|thumb)\.([a-zA-Z0-9]+)(\?.*)?$",
        r"/large.\2\3",
        raw_url,
        flags=re.IGNORECASE,
    )
    return derived_url


def is_photo_license_valid(photo: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate the license on the specific photo object (not observation level).
    Accept only cc-by and cc-by-sa. Reject nc, nd, missing, or unknown.
    """
    license_code = photo.get("license_code")
    if not license_code:
        return False, "missing_or_unknown_license"

    normalized_code = str(license_code).strip().lower()
    if normalized_code in ALLOWED_LICENSES:
        return True, normalized_code
    return False, f"disallowed_license_{normalized_code}"


def extract_photographer_and_attribution(photo: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract photographer name and full attribution from the photo object.
    Do NOT assume the observation uploader is the photographer.
    If photographer cannot be reliably determined from the photo, return 'Unknown'.
    """
    attribution_text = str(photo.get("attribution") or "").strip()
    photographer = "Unknown"

    if attribution_text:
        match = re.search(r"\((?:c|C)\)\s*([^,(-]+)", attribution_text)
        if match:
            extracted = match.group(1).strip()
            if extracted and extracted.lower() not in {
                "some rights reserved",
                "all rights reserved",
                "no rights reserved",
            }:
                photographer = extracted

    # Check native photo fields if photographer could not be parsed from attribution text
    if photographer == "Unknown":
        native_realname = photo.get("native_realname")
        native_username = photo.get("native_username")
        if native_realname and isinstance(native_realname, str) and native_realname.strip():
            photographer = native_realname.strip()
        elif native_username and isinstance(native_username, str) and native_username.strip():
            photographer = native_username.strip()

    return photographer, attribution_text


def fetch_observations_page(
    taxon_id: int,
    page: int = 1,
    per_page: int = 200,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
    request_timeout: int = 15,
) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Fetch a single page of research-grade observations with licensed photos
    for a given taxon_id from the iNaturalist API.

    Returns:
        (success, results, error_message):
        - On success: (True, results_list, "")
        - On failure (exhausted retries or HTTP errors): (False, [], descriptive_error_message)
    """
    sess = session or create_http_session()
    params = {
        "taxon_id": taxon_id,
        "photos": "true",
        "photo_licensed": "true",
        "quality_grade": "research",
        "per_page": per_page,
        "page": page,
        "order": "desc",
        "order_by": "observed_on",
    }

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = sess.get(
                INAT_API_BASE_URL,
                params=params,
                timeout=request_timeout,
            )
            if response.status_code == 200:
                try:
                    payload = response.json()
                    results = payload.get("results", [])
                    return True, results, ""
                except Exception as json_err:
                    last_error = f"api_json_decode_error: {str(json_err)}"
                    logger.error(f"Failed to decode API JSON response: {json_err}")
                    return False, [], last_error
            elif response.status_code == 429:
                wait_time = 2 ** attempt * 2
                last_error = f"api_rate_limited_429 (attempt {attempt}/{max_retries})"
                logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            elif 500 <= response.status_code < 600:
                wait_time = 2 ** attempt
                last_error = f"api_server_error_{response.status_code} (attempt {attempt}/{max_retries})"
                logger.warning(f"Server error ({response.status_code}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                last_error = f"api_http_error_{response.status_code}"
                logger.error(f"API request failed with status {response.status_code}: {response.text[:200]}")
                return False, [], last_error
        except requests.exceptions.RequestException as e:
            last_error = f"api_network_error: {str(e)}"
            logger.warning(f"Request attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    return False, [], last_error


def validate_and_convert_image(
    image_bytes: bytes,
    save_path: str,
) -> Tuple[bool, int, int, str]:
    """
    Validate image bytes with Pillow:
    1. Verify file can be decoded.
    2. Convert to RGB JPEG format without resizing, cropping, or normalizing.
    3. Save to disk.
    Returns (success, width, height, error_message).
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            orig_width, orig_height = img.size

            if img.mode != "RGB":
                rgb_img = img.convert("RGB")
            else:
                rgb_img = img

            temp_path = f"{save_path}.tmp"
            rgb_img.save(temp_path, format="JPEG", quality=95)

            if os.path.exists(save_path):
                os.remove(save_path)
            os.replace(temp_path, save_path)

            return True, orig_width, orig_height, ""
    except Exception as e:
        return False, 0, 0, f"pillow_decode_error: {str(e)}"


def download_image_with_validation(
    image_url: str,
    destination_path: str,
    session: requests.Session,
    timeout: int = 15,
) -> Tuple[bool, int, int, str]:
    """
    Download image from URL with timeout and validate via Pillow.
    Returns (success, width, height, error_reason).
    """
    try:
        resp = session.get(image_url, timeout=timeout)
        if resp.status_code != 200:
            return False, 0, 0, f"http_status_{resp.status_code}"

        return validate_and_convert_image(resp.content, destination_path)
    except requests.exceptions.Timeout:
        return False, 0, 0, "download_timeout"
    except requests.exceptions.RequestException as e:
        return False, 0, 0, f"download_network_error: {str(e)}"


def load_existing_metadata(metadata_file: str) -> pd.DataFrame:
    """
    Load existing metadata CSV if present.
    If the file exists but cannot be read or parsed, halt execution with a clear
    error and preserve a backup to prevent silent data loss.
    """
    if not os.path.exists(metadata_file):
        return pd.DataFrame(columns=METADATA_COLUMNS)

    try:
        df = pd.read_csv(metadata_file)
        missing_cols = [c for c in METADATA_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Existing metadata is missing required columns: {missing_cols}")
        return df[METADATA_COLUMNS]
    except Exception as e:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{metadata_file}.corrupt_backup_{timestamp}"
        try:
            shutil.copy2(metadata_file, backup_path)
            logger.critical(f"Preserved safety backup of unreadable metadata file at: {backup_path}")
        except Exception as backup_err:
            logger.critical(f"Failed to copy safety backup: {backup_err}")

        error_msg = (
            f"CRITICAL: Failed to read existing metadata file '{metadata_file}': {e}. "
            f"A backup copy has been created at '{backup_path}'. "
            f"Execution halted to avoid overwriting or losing existing metadata records. "
            f"Please verify or repair the file before rerunning."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def verify_existing_file(file_path: str) -> bool:
    """
    Check whether a file exists and satisfies all three conditions:
    1. The image can be opened and fully decoded by Pillow.
    2. The image format is JPEG.
    3. The image colour mode is RGB.

    If any condition fails, return False.
    """
    if not file_path or not os.path.isfile(file_path):
        return False
    try:
        with Image.open(file_path) as img:
            # Condition 2: Format must be JPEG
            if img.format != "JPEG":
                return False
            # Condition 3: Colour mode must be RGB
            if img.mode != "RGB":
                return False
            # Condition 1: Fully decode raster data to verify integrity
            img.load()
        return True
    except Exception:
        return False


def get_image_dimensions(file_path: str) -> Tuple[int, int]:
    """Safely obtain width and height of an existing image file using Pillow."""
    try:
        with Image.open(file_path) as img:
            return img.size
    except Exception:
        return 0, 0


def collect_species(
    species_info: Dict[str, Any],
    target_count: int,
    base_dir: str,
    raw_dir: str,
    existing_records: List[Dict[str, Any]],
    processed_photo_ids: Set[int],
    session: requests.Session,
    request_delay: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Collect up to target_count validated images for a single species.
    Ensures that every counted image has a verified file on disk and a complete metadata record.
    """
    species_name = species_info["common_name"]
    scientific_name = species_info["scientific_name"]
    folder_name = species_info["folder"]
    taxon_id = species_info["taxon_id"]

    target_dir = os.path.join(raw_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    # 1. Identify existing verified records for this species (deduplicated by photo_id)
    verified_existing_for_species = []
    species_existing_photo_ids: Set[int] = set()

    for r in existing_records:
        if r.get("species") == species_name:
            rel_p = str(r.get("image_path", ""))
            abs_p = resolve_image_path(base_dir, rel_p)
            p_id = r.get("photo_id")
            if pd.notna(p_id) and str(p_id).isdigit():
                int_p_id = int(p_id)
                # Count only if photo_id is verified (in processed_photo_ids) and file decodes properly
                if int_p_id in processed_photo_ids and verify_existing_file(abs_p):
                    if int_p_id not in species_existing_photo_ids:
                        species_existing_photo_ids.add(int_p_id)
                        verified_existing_for_species.append(r)

    downloaded_count = len(verified_existing_for_species)
    new_records: List[Dict[str, Any]] = []

    skip_reasons: Dict[str, int] = {}
    failure_reasons: Dict[str, int] = {}
    api_failed = False
    observations_exhausted = False

    logger.info(
        f"--- Starting collection for {species_name} ({scientific_name}, Taxon ID: {taxon_id}) ---"
    )
    logger.info(
        f"Target: {target_count} images. Verified existing on disk with metadata: {downloaded_count}."
    )

    if downloaded_count >= target_count:
        logger.info(f"Target already satisfied for {species_name} ({downloaded_count}/{target_count}).")
        report = {
            "species": species_name,
            "requested_count": target_count,
            "successfully_downloaded_count": downloaded_count,
            "skipped_count": 0,
            "failed_count": 0,
            "reasons_for_skipping_or_failure": "None (Target satisfied by existing verified images)",
            "target_achieved": True,
        }
        return new_records, report

    page = 1
    per_page = 200

    while downloaded_count < target_count:
        logger.info(f"Fetching page {page} for {species_name} (Current: {downloaded_count}/{target_count})...")
        success, results, err_msg = fetch_observations_page(
            taxon_id=taxon_id,
            page=page,
            per_page=per_page,
            session=session,
        )
        time.sleep(request_delay)

        if not success:
            logger.error(f"API query failed for {species_name} on page {page}: {err_msg}")
            failure_reasons[err_msg] = failure_reasons.get(err_msg, 0) + 1
            api_failed = True
            break

        if len(results) == 0:
            logger.info(f"API returned 0 observations for {species_name} at page {page}. Observations exhausted.")
            observations_exhausted = True
            break

        for obs in results:
            if downloaded_count >= target_count:
                break

            obs_id = obs.get("id")
            photos = obs.get("photos", [])

            if not photos:
                skip_reasons["no_photos_in_observation"] = skip_reasons.get("no_photos_in_observation", 0) + 1
                continue

            # Iterate through available photos to find the first qualifying photo
            qualifying_photo: Optional[Dict[str, Any]] = None
            qualifying_url: Optional[str] = None
            qualifying_license: Optional[str] = None
            obs_skip_reasons: List[str] = []

            for photo in photos:
                p_id = photo.get("id")
                if not p_id:
                    obs_skip_reasons.append("missing_photo_id")
                    continue

                int_p_id = int(p_id) if str(p_id).isdigit() else None
                if int_p_id is None:
                    obs_skip_reasons.append("invalid_photo_id")
                    continue

                if int_p_id in processed_photo_ids:
                    obs_skip_reasons.append("duplicate_photo_id")
                    continue

                is_valid_lic, lic_res = is_photo_license_valid(photo)
                if not is_valid_lic:
                    obs_skip_reasons.append(lic_res)
                    continue

                img_url = derive_large_image_url(photo)
                if not img_url:
                    obs_skip_reasons.append("unresolvable_large_url")
                    continue

                # Found the first qualifying photo for this observation
                qualifying_photo = photo
                qualifying_url = img_url
                qualifying_license = lic_res
                break

            # If no photo in this observation qualified, skip observation and record tallies
            if not qualifying_photo or not qualifying_url or not qualifying_license:
                for reason in obs_skip_reasons:
                    skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue

            photo_id = int(qualifying_photo["id"])
            image_filename = f"{folder_name}_{photo_id}.jpg"
            dest_path = os.path.join(target_dir, image_filename)

            # Standardized project-root-relative path (e.g. dataset/raw/tiger/tiger_123.jpg)
            rel_image_path = os.path.relpath(dest_path, start=base_dir).replace("\\", "/")

            # Attribution and photographer extraction from photo object
            photographer, attribution_text = extract_photographer_and_attribution(qualifying_photo)
            obs_url = obs.get("uri") or f"https://www.inaturalist.org/observations/{obs_id}"
            observed_on = obs.get("observed_on") or obs.get("time_observed_at") or ""

            # Check if file already exists on disk
            if os.path.exists(dest_path):
                if verify_existing_file(dest_path):
                    # Check if already present in metadata
                    if photo_id in species_existing_photo_ids:
                        processed_photo_ids.add(photo_id)
                        continue

                    # Valid on disk but missing from metadata: reconstruct metadata record
                    orig_w, orig_h = get_image_dimensions(dest_path)
                    metadata_entry = {
                        "image_id": f"{folder_name}_{photo_id}",
                        "observation_id": obs_id,
                        "photo_id": photo_id,
                        "species": species_name,
                        "scientific_name": scientific_name,
                        "image_path": rel_image_path,
                        "image_url": qualifying_url,
                        "photographer": photographer,
                        "photo_license": qualifying_license,
                        "attribution": attribution_text,
                        "observation_url": obs_url,
                        "observed_on": observed_on,
                        "original_width": orig_w,
                        "original_height": orig_h,
                    }
                    new_records.append(metadata_entry)
                    processed_photo_ids.add(photo_id)
                    species_existing_photo_ids.add(photo_id)
                    downloaded_count += 1
                    logger.info(
                        f"[{downloaded_count}/{target_count}] Reconstructed metadata for existing file '{image_filename}'"
                    )
                    continue
                else:
                    logger.warning(f"Existing file '{dest_path}' failed Pillow validation. Re-downloading...")

            # Download and Pillow validation
            success_dl, orig_w, orig_h, err_msg = download_image_with_validation(
                image_url=qualifying_url,
                destination_path=dest_path,
                session=session,
            )

            if not success_dl:
                failure_reasons[err_msg] = failure_reasons.get(err_msg, 0) + 1
                logger.warning(f"Failed to download/validate photo {photo_id}: {err_msg}")
                continue

            metadata_entry = {
                "image_id": f"{folder_name}_{photo_id}",
                "observation_id": obs_id,
                "photo_id": photo_id,
                "species": species_name,
                "scientific_name": scientific_name,
                "image_path": rel_image_path,
                "image_url": qualifying_url,
                "photographer": photographer,
                "photo_license": qualifying_license,
                "attribution": attribution_text,
                "observation_url": obs_url,
                "observed_on": observed_on,
                "original_width": orig_w,
                "original_height": orig_h,
            }

            new_records.append(metadata_entry)
            processed_photo_ids.add(photo_id)
            species_existing_photo_ids.add(photo_id)
            downloaded_count += 1

            logger.info(
                f"[{downloaded_count}/{target_count}] Downloaded & validated {species_name} photo {photo_id} "
                f"({orig_w}x{orig_h}, {qualifying_license})"
            )

            time.sleep(0.3)

        page += 1

    total_skipped = sum(skip_reasons.values())
    total_failed = sum(failure_reasons.values())
    target_achieved = downloaded_count >= target_count

    reason_parts = []
    if skip_reasons:
        reason_parts.append("Skips: " + ", ".join(f"{k}={v}" for k, v in skip_reasons.items()))
    if failure_reasons:
        reason_parts.append("Failures: " + ", ".join(f"{k}={v}" for k, v in failure_reasons.items()))
    if not target_achieved:
        if api_failed:
            reason_parts.append(
                f"Shortfall: API query failure prevented reaching target ({downloaded_count}/{target_count})"
            )
        elif observations_exhausted:
            reason_parts.append(
                f"Shortfall: Exhausted API observations with valid license before reaching {target_count} ({downloaded_count}/{target_count})"
            )
        else:
            reason_parts.append(f"Shortfall: Target not reached ({downloaded_count}/{target_count})")

    report = {
        "species": species_name,
        "requested_count": target_count,
        "successfully_downloaded_count": downloaded_count,
        "skipped_count": total_skipped,
        "failed_count": total_failed,
        "reasons_for_skipping_or_failure": "; ".join(reason_parts) if reason_parts else "None",
        "target_achieved": target_achieved,
    }

    return new_records, report


def run_collection(
    target_per_species: int = 10,
    base_dir: str = ".",
    species_filter: Optional[List[str]] = None,
    delay: float = 1.0,
) -> None:
    """
    Main orchestration routine for iNaturalist pilot collection.
    """
    base_dir = os.path.abspath(base_dir)
    dataset_dir = os.path.join(base_dir, "dataset")
    raw_dir = os.path.join(dataset_dir, "raw")
    metadata_csv_path = os.path.join(dataset_dir, "metadata.csv")
    report_csv_path = os.path.join(dataset_dir, "collection_report.csv")

    os.makedirs(raw_dir, exist_ok=True)

    # 1. Load existing metadata for rerun safety (raises error and preserves backup if corrupt)
    existing_df = load_existing_metadata(metadata_csv_path)

    # Deduplicate existing metadata by photo_id upfront to avoid counting duplicates
    if not existing_df.empty:
        existing_df = existing_df.drop_duplicates(subset=["photo_id"], keep="first")
    existing_records = existing_df.to_dict("records")

    # Standardize image_path in any existing records in memory to project-root-relative format
    for rec in existing_records:
        raw_p = str(rec.get("image_path", ""))
        rec["image_path"] = os.path.relpath(
            resolve_image_path(base_dir, raw_p),
            start=base_dir,
        ).replace("\\", "/")

    # Only mark photo_id as processed if its corresponding metadata record points
    # to an existing image file that passes Pillow verification.
    # Missing or corrupt files are omitted from processed_photo_ids so they remain eligible for re-download.
    processed_photo_ids: Set[int] = set()
    for r in existing_records:
        p_id = r.get("photo_id")
        if pd.notna(p_id) and str(p_id).isdigit():
            int_p_id = int(p_id)
            rel_p = str(r.get("image_path", ""))
            abs_p = resolve_image_path(base_dir, rel_p)
            if verify_existing_file(abs_p):
                processed_photo_ids.add(int_p_id)
            else:
                logger.warning(
                    f"Existing metadata record for photo {int_p_id} ('{rel_p}') is missing on disk or corrupt. "
                    f"Marking as eligible for re-download."
                )

    session = create_http_session()

    # Filter species if requested
    selected_species = SPECIES_CONFIG
    if species_filter:
        lower_filter = [s.strip().lower() for s in species_filter]
        selected_species = [
            s for s in SPECIES_CONFIG
            if s["common_name"].lower() in lower_filter or s["folder"].lower() in lower_filter
        ]

    all_new_records: List[Dict[str, Any]] = []
    collection_reports: List[Dict[str, Any]] = []

    logger.info("============================================================")
    logger.info("   WildMatch AI - iNaturalist Pilot Dataset Collection      ")
    logger.info("============================================================")
    logger.info(f"Project base dir: {base_dir}")
    logger.info(f"Target count per species: {target_per_species}")
    logger.info(f"Selected species count: {len(selected_species)}")
    logger.info(f"Allowed licenses: {sorted(list(ALLOWED_LICENSES))}")
    logger.info("Quality grade: research only")
    logger.info("============================================================")

    for sp_info in selected_species:
        new_records, report = collect_species(
            species_info=sp_info,
            target_count=target_per_species,
            base_dir=base_dir,
            raw_dir=raw_dir,
            existing_records=existing_records,
            processed_photo_ids=processed_photo_ids,
            session=session,
            request_delay=delay,
        )
        all_new_records.extend(new_records)
        collection_reports.append(report)

    # 2. Update and write dataset/metadata.csv
    if all_new_records:
        new_photo_ids = {r["photo_id"] for r in all_new_records}
        # Replace stale records for re-downloaded photos
        retained_existing = [r for r in existing_records if r.get("photo_id") not in new_photo_ids]
        updated_records = retained_existing + all_new_records
        final_meta_df = pd.DataFrame(updated_records)[METADATA_COLUMNS]
    else:
        final_meta_df = pd.DataFrame(existing_records)[METADATA_COLUMNS] if existing_records else existing_df[METADATA_COLUMNS]

    # Final deduplication by photo_id
    final_meta_df = final_meta_df.drop_duplicates(subset=["photo_id"], keep="last")

    # Safety backup before overwriting metadata
    if os.path.exists(metadata_csv_path):
        ts = time.strftime("%Y%m%d_%H%M%S")
        meta_backup = f"{metadata_csv_path}.bak_{ts}"
        shutil.copy2(metadata_csv_path, meta_backup)
        logger.info(f"Created safety backup of metadata at: {meta_backup}")

    final_meta_df.to_csv(metadata_csv_path, index=False)
    logger.info(f"Updated metadata saved to: {metadata_csv_path} (Total rows: {len(final_meta_df)})")

    # 3. Generate and write dataset/collection_report.csv
    report_df = pd.DataFrame(collection_reports)[REPORT_COLUMNS]
    report_df.to_csv(report_csv_path, index=False)
    logger.info(f"Collection report saved to: {report_csv_path}")

    # Summary table
    logger.info("\n--- FINAL COLLECTION REPORT ---")
    print(report_df.to_string(index=False))

    all_achieved = all(r["target_achieved"] for r in collection_reports)
    if not all_achieved:
        logger.warning("One or more species did not achieve the requested target without relaxing filters.")
    else:
        logger.info("All species successfully collected and verified!")


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="WildMatch AI - Stage 1: iNaturalist Image Collector (Pilot)"
    )
    parser.add_argument(
        "--target-per-species",
        type=int,
        default=10,
        help="Maximum target images to collect per species (default: 10)",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=None,
        help="Optional specific species to collect (tiger, lion, elephant, chital)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Politeness delay in seconds between API calls (default: 1.0)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Root directory of the project (default: current directory)",
    )

    args = parser.parse_args()

    run_collection(
        target_per_species=args.target_per_species,
        base_dir=args.base_dir,
        species_filter=args.species,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()
