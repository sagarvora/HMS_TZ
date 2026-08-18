"""Move inline signature payloads into the content-addressed registry.

Designed to run against a live site: work happens in small autocommitted batches keyed
on the primary key, never in one long transaction, so no batch holds row locks or a
read view for more than a moment. It is interruptible and re-runnable — each pass only
looks at rows that still hold a `data:` payload, so a killed run simply resumes.

Writes go through raw SQL rather than the ORM on purpose. `doc.save()` would record a
`14 KB -> pointer` diff per row in `tabVersion`, which on ~1.6M encounters means writing
tens of GB into a table that is already the larger problem. Raw UPDATEs also skip
`modified`, which is correct here: exchanging a payload for a pointer to the same image
is not a change to the document's content.

Converting the values only marks the freed pages reusable inside the tablespace; the file
on disk does not shrink. So each converted table is then rebuilt with
`ALTER TABLE ... FORCE, ALGORITHM=INPLACE, LOCK=NONE`, which writes a fresh, densely
packed tablespace and unlinks the old one. `LOCK=NONE` is what makes this safe to run
here: MariaDB refuses the statement outright if it cannot rebuild without blocking
writes, so a refusal is logged and skipped rather than becoming an outage.
"""

import frappe

from hms_tz.signature_registry import is_payload, signature_columns, store

BATCH_SIZE = 200
LOG_EVERY = 20_000
# How long our ALTER waits for the table's metadata lock. This is `lock_wait_timeout`
# (MDL/DDL), not `innodb_lock_wait_timeout` (row locks) — different variable, default 86400.
#
# Acquisition time has nothing to do with row count: an MDL is an in-memory lock entry, so
# the wait is however long the transactions currently holding the table take to finish.
# Short saves are milliseconds; a long report or an idle-in-transaction connection is
# unbounded.
#
# One value has to cover two opposed needs, since MariaDB has no separate knob:
#   - at the START, waiting long is bad — a pending MDL request blocks every request queued
#     behind it, so our wait is exactly how long the convoy can grow.
#   - at the END, waiting short is worse — the ALTER upgrades to an exclusive lock to swap
#     the tablespace, and losing that race discards the entire rebuild after hours of work.
# The second failure costs far more, so this errs long. It bounds the day-long default
# without making a transient reader throw the rebuild away.
LOCK_WAIT_TIMEOUT = 300


def execute():
    converted = {}
    for doctype, fieldname in signature_columns():
        converted[doctype] = converted.get(doctype, 0) + migrate_column(doctype, fieldname)

    # Rebuild once per table, after every column on it has been converted. Only tables we
    # actually rewrote rows on are worth rebuilding.
    for doctype, rows in converted.items():
        if rows:
            reclaim_space(doctype)


def estimate():
    """Report what the migration would move, without writing anything.

    Run this first. The registry is populated while the old payloads are still in place,
    so both exist at once and the disk needs to hold `registry_mb` on top of the current
    tables. `column_mb` is what the tables stop carrying, though InnoDB only returns it to
    the filesystem after a rebuild.

    Counts distinct hashes, so `registry_mb` is the real (deduplicated) cost.
    """
    from hms_tz.signature_registry import hash_of

    report = []
    for doctype, fieldname in signature_columns():
        table = f"tab{doctype}"
        rows, column_bytes = frappe.db.sql(
            f"""select count(*), coalesce(sum(length(`{fieldname}`)), 0)
            from `tab{doctype}` where `{fieldname}` like 'data:%'"""
        )[0]
        if not rows:
            continue

        # Distinct payloads decide registry size. Hashing is done here rather than with
        # SQL SHA2() so it matches the hash the registry will actually write.
        seen = {}
        last_seen = ""
        while True:
            batch = frappe.db.sql(
                f"""select name, `{fieldname}` as value from `{table}`
                where name > %s and `{fieldname}` like 'data:%%'
                order by name limit {BATCH_SIZE}""",
                (last_seen,),
                as_dict=True,
            )
            if not batch:
                break
            last_seen = batch[-1].name
            for row in batch:
                # base64 is ~4/3 of the raw bytes it encodes.
                seen[hash_of(row.value)] = len(row.value) * 3 // 4

        report.append(
            {
                "doctype": doctype,
                "fieldname": fieldname,
                "rows": rows,
                "distinct": len(seen),
                "column_mb": round(column_bytes / 1024**2, 1),
                "registry_mb": round(sum(seen.values()) / 1024**2, 1),
            }
        )

    for line in report:
        print(
            f"{line['doctype']}.{line['fieldname']}: {line['rows']} rows "
            f"({line['distinct']} distinct) | in tables {line['column_mb']} MB "
            f"-> registry {line['registry_mb']} MB"
        )
    if report:
        print(
            f"\ntotal: tables lose {sum(r['column_mb'] for r in report):.1f} MB, "
            f"disk gains {sum(r['registry_mb'] for r in report):.1f} MB"
        )
    else:
        print("nothing to migrate")

    return report




def migrate_column(doctype: str, fieldname: str):
    """Walk one column in primary-key order, converting payloads to pointers.

    Pagination is by keyset (`name > last_seen`) rather than OFFSET or a bare
    `where <col> like 'data:%'` limit: each batch is then a short range scan on the
    clustered index instead of a fresh full scan, and progress is monotonic regardless of
    what concurrent traffic writes behind us.

    A streaming (unbuffered) cursor is deliberately not used. It would hold one result set
    open across the whole table, and `frappe.db.commit()` issues its `COMMIT` on that same
    cursor — committing mid-stream with rows still unread breaks the connection. Keyset
    batching gets the same O(1) memory profile while keeping transactions short.
    """
    table = f"tab{doctype}"
    estimate = frappe.db.estimate_count(doctype)
    print(f"{doctype}.{fieldname}: scanning ~{estimate} rows")

    last_seen = ""
    scanned = 0
    done = 0
    failed = 0

    while True:
        batch = frappe.db.sql(
            f"""select name, `{fieldname}` as value from `{table}`
            where name > %s
            order by name
            limit {BATCH_SIZE}""",
            (last_seen,),
            as_dict=True,
        )
        if not batch:
            break

        last_seen = batch[-1].name
        scanned += len(batch)

        for row in batch:
            if not is_payload(row.value):
                continue
            try:
                pointer = store(row.value)
            except Exception:
                # One malformed payload must not abort the run; log it and move on.
                frappe.log_error(
                    title="Signature migration failed",
                    message=f"{doctype} {row.name}.{fieldname}\n\n{frappe.get_traceback()}",
                )
                failed += 1
                continue

            # Guard on the old value so a concurrent save that already wrote a pointer
            # (or a newly drawn signature) is never clobbered.
            frappe.db.sql(
                f"""update `{table}` set `{fieldname}` = %s
                where name = %s and `{fieldname}` = %s""",
                (pointer, row.name, row.value),
            )
            done += 1

        # Commit per batch: keeps transactions short so live traffic is never waiting on
        # our locks, and makes progress durable as we go.
        frappe.db.commit()

        if scanned % LOG_EVERY < BATCH_SIZE:
            print(f"  scanned {scanned}, migrated {done}")

    print(f"{doctype}.{fieldname}: migrated {done}, failed {failed} (scanned {scanned})")
    return done


def table_space(doctype: str) -> dict:
    """On-disk size and reusable-but-not-returned space for a table, in MB."""
    row = frappe.db.sql(
        """select round(data_length/1024/1024, 1), round(data_free/1024/1024, 1), row_format
        from information_schema.tables
        where table_schema = database() and table_name = %s""",
        (f"tab{doctype}",),
        as_dict=False,
    )
    if not row:
        return {}
    data_mb, free_mb, row_format = row[0]
    return {"data_mb": data_mb, "free_mb": free_mb, "row_format": row_format}


def reclaim_space(doctype: str):
    """Rebuild a table so the space freed by the conversion returns to the filesystem.

    `ALTER TABLE ... FORCE` writes a new tablespace and unlinks the old one. `LOCK=NONE`
    is the safety mechanism rather than an optimisation: MariaDB rejects the statement if
    it cannot rebuild while writes continue, so the worst case is a skipped table, never a
    blocked one. `ALGORITHM=INPLACE` likewise refuses rather than silently falling back to
    a blocking copy.

    Needs free disk for a second copy of the table while it runs.
    """
    before = table_space(doctype)
    if not before:
        return

    # Deliberately not gated on `data_free`. It reflects stale table statistics and lags
    # behind InnoDB's purge, so a table that just had gigabytes of overflow pages released
    # can still report a few MB free. Measured: converting 1500 x 14 KB signatures showed
    # only 5 MB free while 67 MB was genuinely reclaimable. Callers gate on having actually
    # converted rows instead, which is the reliable signal.
    print(f"{doctype}: {before['data_mb']} MB on disk ({before['free_mb']} MB reported free) — rebuilding")

    # `SET SESSION` is scoped to the connection, not the transaction: it is not rolled back
    # and outlives each commit, so it stays in effect for the rest of this process. That is
    # fine for a patch run, which exits shortly after.
    #
    # `innodb_online_alter_log_max_size` is deliberately left alone. It is GLOBAL-only and
    # needs SUPER, which a site's DB user never has. Its 128 MB default sizes the buffer for
    # writes arriving *during* the rebuild, and once signatures are pointers a save writes
    # ~114 bytes where it used to write 14 KB, so the buffer fills far slower than it did
    # before this patch. If it ever does overflow, the ALTER rolls back with the table
    # untouched and can be retried in a quieter window.
    frappe.db.sql(f"set session lock_wait_timeout = {LOCK_WAIT_TIMEOUT}")

    try:
        frappe.db.sql_ddl(f"alter table `tab{doctype}` force, algorithm=inplace, lock=none")
    except Exception as e:
        # `LOCK=NONE` trades "might block writes" for "might not run", so a refusal here is
        # the mechanism working. The table is left byte-identical either way — both a
        # refusal and a rolled-back rebuild are no-ops — but say so loudly, because the
        # cost of missing it is a table that quietly stays large.
        frappe.log_error(
            title="Signature space reclaim skipped",
            message=(
                f"tab{doctype} was not rebuilt and still occupies {before['data_mb']} MB.\n"
                f"Signatures were converted correctly, so the space is reusable by new rows —\n"
                f"the table just will not shrink on disk. It is unchanged and safe.\n\n"
                f"Re-run `reclaim_space('{doctype}')` once the cause below is resolved, or\n"
                f"rebuild it in a maintenance window without LOCK=NONE.\n\n{frappe.get_traceback()}"
            ),
        )
        print(f"  !! SKIPPED {doctype}: {e}")
        print(f"  !! table unchanged at {before['data_mb']} MB, not shrunk. See Error Log.")
        return

    after = table_space(doctype)
    reclaimed = round(before["data_mb"] - after["data_mb"], 1)
    print(f"  {before['data_mb']} MB -> {after['data_mb']} MB on disk (reclaimed {reclaimed} MB)")
