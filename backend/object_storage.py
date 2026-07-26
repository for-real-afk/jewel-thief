"""
Cloudflare R2 (S3-compatible) storage for catalog item images.

Fixes a live data-loss bug, not a scaling nice-to-have: Render's disk is
ephemeral (DEPLOYMENT.md §1) -- any image added or replaced through the live
/catalog admin page was written only to backend/static/catalog/*.jpg on the
running container's disk, and vanished on the next restart/redeploy. This
module uploads to R2 instead and returns a public URL, so image_url survives
restarts the same way the rest of a catalog item's metadata (Supabase, §7)
already does.

R2 chosen over S3 for no egress fees; the code is nearly identical either
way since both speak the S3 API via boto3.
"""
import boto3
from botocore.config import Config

from config import get_settings
from utils import external_api_retry

settings = get_settings()

_client = boto3.client(
    "s3",
    endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
    aws_access_key_id=settings.r2_access_key_id,
    aws_secret_access_key=settings.r2_secret_access_key,
    config=Config(signature_version="s3v4"),
    region_name="auto",
)


def _key_for(item_id: str) -> str:
    return f"catalog/{item_id}.jpg"


@external_api_retry
def upload_catalog_image(item_id: str, image_bytes: bytes) -> str:
    """Uploads a catalog item's normalized JPEG to R2 (overwriting any
    existing object for this item_id) and returns its public URL."""
    key = _key_for(item_id)
    _client.put_object(Bucket=settings.r2_bucket_name, Key=key, Body=image_bytes, ContentType="image/jpeg")
    return f"{settings.r2_public_url_base}/{key}"


def delete_catalog_image(item_id: str) -> None:
    """delete_object is idempotent -- no error if the key doesn't exist, so
    callers don't need an existence check first (unlike the old local-disk
    Path.unlink(), which required Path.exists())."""
    _client.delete_object(Bucket=settings.r2_bucket_name, Key=_key_for(item_id))
