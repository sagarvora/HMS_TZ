"""Content-addressed storage for Signature fields.

Signature fields are `longtext` holding `data:image/png;base64,...` URIs. Denormalized
copies (a practitioner's signature stamped onto every encounter) duplicate ~14 KB per
row, which is where the bloat comes from.

Each distinct signature is written once under `private/files/signatures/` and the column
holds `sig:<sha256>.<ext>` instead. The store is keyed by the hash of the original value,
so it is append-only: nothing is mutated or garbage collected, a signature outlives the
master record it was copied from, and re-signing yields a new hash while old documents
keep pointing at what was actually signed.

The filename lives in the pointer, so resolving is one stat and one read — the extension
never has to be guessed, and the format is not recorded anywhere a second time. A value
carrying a recognised image type is decoded and stored as a real `.png`/`.jpg` (smaller
than base64 and openable outside Frappe); anything else is stored verbatim as `.txt` and
round-trips byte for byte. An unfamiliar format therefore degrades to exact preservation
rather than to a filename that lies about its contents.

Nothing renders a pointer on its own: every consumer resolves it first, via `hydrate()`
for a whole doc, `resolve()` for one value, or the `resolve` jinja helper in a print
format. An earlier design stored a fetchable URL so `<img src="...">` would work
untouched, but that only holds for interactive prints — `frappe.utils.pdf` inlines images
solely for URLs backed by a File record, and a queued PDF job carries no session cookie,
so the same markup silently 403s in background rendering.
"""

import os
import re
from base64 import b64decode, b64encode
from hashlib import sha256

import frappe
from frappe import scrub
from frappe.utils import get_files_path

SIGNATURE_DIR = "signatures"
POINTER_PREFIX = "sig:"
DATA_URI_PREFIX = "data:"
VERBATIM_EXT = "txt"
POINTER_PATTERN = re.compile(r"^sig:([0-9a-f]{64})\.([a-z0-9]+)$")
DATA_URI_PATTERN = re.compile(r"^data:\s*([\w.+/-]*)\s*;base64,(.*)$", re.IGNORECASE | re.DOTALL)
# Types worth storing as real image files. Anything else is kept verbatim, so this table
# only ever decides "smaller on disk", never whether a value can be stored.
IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg"}
MIME_FOR_EXT = {"png": "image/png", "jpg": "image/jpeg"}
# Old-format pointers, resolvable so a site migrated by an earlier version keeps working.
LEGACY_ENDPOINT = "/api/method/hms_tz.signature_registry.get_image"
LEGACY_PATTERN = re.compile(re.escape(LEGACY_ENDPOINT) + r"\?h=([0-9a-f]{64})$")
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def is_pointer(value: str | None) -> bool:
    return bool(value) and bool(POINTER_PATTERN.match(value.strip()))


def is_payload(value: str | None) -> bool:
    return bool(value) and value.strip().startswith(DATA_URI_PREFIX)


def pointer_for(digest: str, extension: str) -> str:
    return f"{POINTER_PREFIX}{digest}.{extension}"


def hash_of(payload: str) -> str:
    return sha256(payload.strip().encode()).hexdigest()


def path_for(digest: str, extension: str) -> str:
    return get_files_path(SIGNATURE_DIR, f"{digest}.{extension}", is_private=True)


def store(payload: str) -> str:
    """Write a signature to the registry and return its pointer. Idempotent.

    Writes before the transaction commits, and does not clean up on rollback. That ordering
    is deliberate: a crash then leaves an unreferenced file (reclaimed next time the same
    signature is stored) instead of a committed pointer to a file that never got written,
    which would lose a signature for good. Deleting on rollback would be unsafe anyway —
    documents share signatures, so the file may belong to one that committed.
    """
    digest = hash_of(payload)
    extension, content = _classify(payload)
    path = path_for(digest, extension)

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # write-then-rename so a concurrent reader never sees a partial file
        tmp_path = f"{path}.{frappe.generate_hash(length=8)}.tmp"
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return pointer_for(digest, extension)


def resolve(value: str | None) -> str:
    """Return the inline `data:` URI for a stored signature value.

    - falsy            -> `""`
    - `sig:<hash>.ext` -> the signature, or `frappe.throw` if the file is gone
    - `data:...`       -> unchanged, already inline
    - anything else    -> unchanged, not ours

    Throws rather than degrading on a missing file: a pointer with no file behind it means a
    signature has been lost, and on a medico-legal record that must stop the caller instead
    of quietly reaching an insurer as a malformed payload.
    """
    if not value:
        return ""

    located = _locate(value)
    if not located:
        return value

    path, extension = located
    if not os.path.exists(path):
        frappe.throw(
            frappe._("Signature file is missing from the registry: {0}").format(os.path.basename(path)),
            frappe.DoesNotExistError,
        )

    with open(path, "rb") as f:
        content = f.read()

    if extension == VERBATIM_EXT:
        return content.decode()
    return f"data:{MIME_FOR_EXT[extension]};base64," + b64encode(content).decode()


def _classify(payload: str) -> tuple[str, bytes]:
    """Pick the on-disk form: a real image file, or the value verbatim.

    Decoding is only ever an optimisation — a recognised image type is stored as raw bytes
    because that is smaller than base64 and openable outside Frappe. What is preserved is
    the image content, not the exact spelling of the data URI, so a value whose base64 is
    line-wrapped or whose mime is `image/PNG` still becomes a plain `.png`.

    Everything else — unknown or absent mime, base64 that will not decode — is written
    verbatim, so an unfamiliar format degrades to exact preservation instead of a filename
    that lies about its contents.
    """
    value = payload.strip()
    match = DATA_URI_PATTERN.match(value)
    if match:
        extension = IMAGE_EXTENSIONS.get(match.group(1).lower().strip())
        if extension:
            try:
                content = b64decode(match.group(2))
            except Exception:
                content = None
            if content:
                return extension, content

    return VERBATIM_EXT, value.encode()


def _locate(value: str) -> tuple[str, str] | None:
    """Map a stored value to (path, extension), or None if it is not a pointer."""
    value = value.strip()

    match = POINTER_PATTERN.match(value)
    if match:
        digest, extension = match.groups()
        return path_for(digest, extension), extension

    # A bare hash or an old URL pointer carries no extension, so probe the known ones. Only
    # reachable for data written by an earlier version, or a hand-written call.
    legacy = LEGACY_PATTERN.match(value)
    digest = legacy.group(1) if legacy else (value if HASH_PATTERN.match(value) else None)
    if not digest:
        return None

    for extension in (*MIME_FOR_EXT, VERBATIM_EXT):
        path = path_for(digest, extension)
        if os.path.exists(path):
            return path, extension

    # Nothing on disk: report the canonical name so the error names something real.
    return path_for(digest, VERBATIM_EXT), VERBATIM_EXT


# --- doctype events -------------------------------------------------------------------


def owned_by_hms_tz(module: str) -> bool:
    """Whether hms_tz owns a module, and so the tables of the doctypes in it.

    `.get()` rather than indexing: a module can be missing from `module_app` (a leftover
    custom Module Def, or one from an uninstalled app), and since these hooks run on every
    doctype on the site, raising here would make unrelated documents unsaveable.
    """
    return frappe.local.module_app.get(scrub(module)) == "hms_tz"


def signature_columns() -> list[tuple[str, str]]:
    """Every (doctype, fieldname) Signature column hms_tz owns, standard or custom.

    Read from the database, not the filesystem, so custom fields that exist only on this
    site are included. Singles are skipped — their values live in `tabSingles`.
    """
    modules = frappe.get_all("Module Def", filters={"app_name": "hms_tz"}, pluck="name")
    if not modules:
        return []

    doctypes = frappe.get_all(
        "DocType",
        filters={"module": ["in", modules], "issingle": 0, "is_virtual": 0},
        pluck="name",
    )
    if not doctypes:
        return []

    standard = frappe.get_all(
        "DocField",
        filters={"fieldtype": "Signature", "parent": ["in", doctypes]},
        fields=["parent", "fieldname"],
    )
    custom = frappe.get_all(
        "Custom Field",
        filters={"fieldtype": "Signature", "dt": ["in", doctypes]},
        fields=["dt as parent", "fieldname"],
    )

    columns = []
    for row in standard + custom:
        pair = (row.parent, row.fieldname)
        if pair not in columns and frappe.db.has_column(row.parent, row.fieldname):
            columns.append(pair)

    return columns


def signature_fields(meta) -> list[str]:
    """Signature fieldnames on an hms_tz doctype's meta, including custom fields.

    Takes a Meta so hooks can pass `doc.meta` and skip a lookup. App-scoped because the
    hooks are registered against `*`.

    Never cached here: Meta already is, and frappe invalidates it when a DocType or Custom
    Field changes. A stale empty list would silently send signatures back to inline base64.
    """
    if meta.issingle or not owned_by_hms_tz(meta.module):
        return []
    return [df.fieldname for df in meta.get("fields", {"fieldtype": "Signature"})]


def dehydrate(doc, method=None, *args, **kwargs):
    """Replace inline signature payloads with registry pointers.

    Hooked on `validate` because that is the last event before the row is written; by
    `on_update` the payload is already in the column and only a second write could undo
    it, leaving the bloat in both the table and `tabVersion`.
    """
    for fieldname in signature_fields(doc.meta):
        value = doc.get(fieldname)
        if is_payload(value):
            doc.set(fieldname, store(value))


def hydrate(doc, method=None, *args, **kwargs):
    """Replace every signature pointer on a doc with its `data:` URI, in place.

    Hooked on `onload` and `before_print`.

    Only safe on a doc that will not be written. It mutates in place, so calling it on a doc
    mid-save — anywhere from `validate` onwards, e.g. `before_submit` — persists the base64
    straight back into the column, since `dehydrate` has already run by then. Use `resolve()`
    per field in those paths.

    `*args` absorbs event-supplied arguments; `before_print` passes `print_settings`.
    """
    for fieldname in signature_fields(doc.meta):
        value = doc.get(fieldname)
        if is_pointer(value):
            doc.set(fieldname, resolve(value))


# --- disk accounting ------------------------------------------------------------------


def usage() -> dict:
    """Size of the registry on disk."""
    directory = get_files_path(SIGNATURE_DIR, is_private=True)
    if not os.path.isdir(directory):
        return {"files": 0, "bytes": 0}

    total = 0
    count = 0
    for entry in os.scandir(directory):
        if entry.is_file() and not entry.name.endswith(".tmp"):
            total += entry.stat().st_size
            count += 1
    return {"files": count, "bytes": total}


def audit(delete_orphans: bool = False) -> dict:
    """Report files on disk that no document points at, optionally deleting them.

    Orphans are harmless — a rolled-back save, or a deleted document — and get reused if the
    same signature is stored again. Deleting is only safe because every hms_tz Signature
    column is scanned, so a file shared by several documents is never called an orphan.
    """
    referenced = set()
    for doctype, fieldname in signature_columns():
        for (value,) in frappe.db.sql(
            f"""select distinct `{fieldname}` from `tab{doctype}`
            where `{fieldname}` like %s""",
            (f"{POINTER_PREFIX}%",),
        ):
            located = _locate(value or "")
            if located:
                referenced.add(os.path.basename(located[0]))

    directory = get_files_path(SIGNATURE_DIR, is_private=True)
    orphans = []
    reclaimable = 0
    if os.path.isdir(directory):
        for entry in os.scandir(directory):
            if not entry.is_file() or entry.name.endswith(".tmp"):
                continue
            if entry.name not in referenced:
                orphans.append(entry.path)
                reclaimable += entry.stat().st_size

    if delete_orphans:
        for path in orphans:
            os.unlink(path)

    return {
        "referenced": len(referenced),
        "orphans": len(orphans),
        "reclaimable_bytes": reclaimable,
        "deleted": delete_orphans,
        **usage(),
    }
