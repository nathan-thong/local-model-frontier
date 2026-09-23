"""Secure, inventory-pinned extraction of the Q10-PREP-002 PMC prose scope."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ALI_NAMESPACE = "http://www.niso.org/schemas/ali/1.0/"
MATHML_NAMESPACE = "http://www.w3.org/1998/Math/MathML"
CC_BY_4_URI = "https://creativecommons.org/licenses/by/4.0/"
EXCLUDED_ELEMENTS = {
    "back",
    "ref-list",
    "ack",
    "permissions",
    "license",
    "license_ref",
    "fig",
    "fig-group",
    "table-wrap",
    "table-wrap-group",
    "table",
    "caption",
    "supplementary-material",
    "inline-supplementary-material",
    "supplement",
    "media",
    "inline-media",
    "graphic",
    "inline-graphic",
    "floats-group",
    "xref",
    "label",
    "fn",
    "fn-group",
    "disp-formula",
    "inline-formula",
    "chem-struct-wrap",
    "code",
    "preformat",
    "boxed-text",
    "disp-quote",
    "verse-group",
    "speech",
    "app-group",
    "app",
    "copyright-statement",
    "copyright-holder",
    "copyright-year",
}
BLOCK_ELEMENTS = {
    "abstract",
    "body",
    "list",
    "list-item",
    "p",
    "sec",
    "title",
}


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _namespace(name: str) -> str:
    return name[1:].split("}", 1)[0] if name.startswith("{") else ""


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity_sha256(text: str) -> str:
    return _sha256(_normalize(text).encode("utf-8"))


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"inventory must contain a JSON object: {path}")
    return value, raw


def _validate_inventories(preparation_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    preparation, _ = _read_json(preparation_path)
    if preparation.get("experiment_id") != "Q10-PREP-002":
        raise ValueError("preparation inventory is not Q10-PREP-002")

    repository_root = Path(__file__).resolve().parents[3]
    rights_path = repository_root / preparation.get(
        "input_inventory_path", "docs/q10g_pmc_ali_rights_audit.json"
    )
    source5_path = repository_root / preparation.get(
        "source5_inventory_path", "docs/q10f_pmc_article_rights_inventory.json"
    )
    rights, rights_raw = _read_json(rights_path)
    _, source5_raw = _read_json(source5_path)
    if _sha256(rights_raw) != preparation.get("input_inventory_sha256"):
        raise ValueError("Q10-SOURCE-007 inventory hash differs from the preregistration")
    if _sha256(source5_raw) != preparation.get("source5_inventory_sha256"):
        raise ValueError("Q10-SOURCE-005 inventory hash differs from the preregistration")
    if rights.get("experiment_id") != "Q10-SOURCE-007":
        raise ValueError("pinned rights inventory is not Q10-SOURCE-007")
    if rights.get("source5_inventory_sha256") != preparation.get("source5_inventory_sha256"):
        raise ValueError("Q10-SOURCE-007 does not pin the Q10-SOURCE-005 inventory")

    frozen = preparation.get("frozen_records")
    cleared = rights.get("preregistered_records")
    if not isinstance(frozen, list) or not isinstance(cleared, list) or len(frozen) != len(cleared):
        raise ValueError("frozen preparation and rights record lists do not agree")
    rights_by_id = {
        (record.get("pmcid"), record.get("cloud_version")): record
        for record in cleared
        if isinstance(record, dict)
    }
    if len(rights_by_id) != len(cleared):
        raise ValueError("rights inventory has missing or duplicate PMCID.version identities")
    for record in frozen:
        identity = (record.get("pmcid"), record.get("cloud_version"))
        rights_record = rights_by_id.get(identity)
        if rights_record is None:
            raise ValueError(f"preparation record is absent from rights audit: {identity}")
        if rights_record.get(
            "current_rights_review_status"
        ) != "passed_frozen_abstract_main_body_prose_scope" or not rights_record.get(
            "body_rights_review_status", ""
        ).startswith("reviewed_"):
            raise ValueError(f"record lacks the Q10-SOURCE-007 rights pass: {identity}")
        evidence = rights_record.get("rights_review_evidence", {})
        if not all(
            evidence.get(key) is True
            for key in (
                "xml_object_size_md5_sha256_verified",
                "pmcid_version_title_doi_exact",
                "no_unresolved_conflicting_terms_for_included_prose",
                "body_rights_review_completed",
            )
        ):
            raise ValueError(f"record has incomplete Q10-SOURCE-007 evidence: {identity}")
        ali_pointer = evidence.get("niso_ali_license_ref", {})
        if (
            ali_pointer.get("namespace_uri") != ALI_NAMESPACE
            or ali_pointer.get("uri") != CC_BY_4_URI
            or ali_pointer.get("exact") is not True
            or record.get("license_uri") != CC_BY_4_URI
            or rights_record.get("xml_object_key") != record.get("xml_object_key")
        ):
            raise ValueError(f"record does not match the frozen CC BY rights pointer: {identity}")
        expected = {
            "xml_size_bytes": rights_record.get("xml_size_bytes"),
            "xml_md5": rights_record.get("xml_md5"),
            "xml_sha256": rights_record.get("xml_sha256"),
            "title": rights_record.get("cloud_title"),
            "doi": rights_record.get("doi"),
        }
        for key, value in expected.items():
            if value is not None and record.get(key) != value:
                raise ValueError(f"Q10-PREP-002 and rights audit disagree on {key}: {identity}")
    minimum = preparation.get("minimum_complete_documents")
    if not isinstance(minimum, int) or len(frozen) < minimum:
        raise ValueError("frozen allowlist is below the preregistered complete-document minimum")
    return preparation, frozen


def _record_input_path(input_dir: Path, record: dict[str, Any]) -> Path:
    filename = f"{record['pmcid']}.{record['cloud_version']}.xml"
    flat_path = input_dir / filename
    object_path = input_dir / record["xml_object_key"]
    candidates = [path for path in (flat_path, object_path) if path.is_file()]
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one frozen XML object {filename} under {input_dir}; found {len(candidates)}"
        )
    return candidates[0]


def _direct_child(parent: ET.Element, local_name: str) -> ET.Element | None:
    return next((child for child in parent if _local_name(child.tag) == local_name), None)


def _article_metadata(root: ET.Element, record: dict[str, Any]) -> ET.Element:
    pmc_ids = {
        (element.text or "").strip()
        for element in root.iter()
        if _local_name(element.tag) == "article-id"
        and element.attrib.get("pub-id-type", "").casefold() in {"pmc", "pmcid"}
    }
    if record["pmcid"].casefold() not in {value.casefold() for value in pmc_ids}:
        raise ValueError(f"XML PMCID does not match frozen record {record['pmcid']}")
    versioned_ids = {
        (element.text or "").strip().casefold()
        for element in root.iter()
        if _local_name(element.tag) == "article-id"
        and element.attrib.get("pub-id-type", "").casefold() == "pmcid-ver"
    }
    expected_versioned_id = f"{record['pmcid']}.{record['cloud_version']}".casefold()
    if expected_versioned_id not in versioned_ids:
        raise ValueError(f"XML PMCID.version does not match frozen record {record['pmcid']}")
    front = _direct_child(root, "front")
    meta = _direct_child(front, "article-meta") if front is not None else None
    if meta is None:
        raise ValueError(f"XML lacks front/article-meta for {record['pmcid']}")
    title_group = _direct_child(meta, "title-group")
    title = _direct_child(title_group, "article-title") if title_group is not None else None
    actual_title = _normalize("".join(title.itertext())) if title is not None else ""
    if actual_title != _normalize(str(record["title"])):
        raise ValueError(f"XML title does not match frozen record {record['pmcid']}")
    dois = {
        _normalize(element.text or "")
        .casefold()
        .removeprefix("https://doi.org/")
        .removeprefix("doi:")
        for element in root.iter()
        if _local_name(element.tag) == "article-id"
        and element.attrib.get("pub-id-type", "").casefold() == "doi"
    }
    expected_doi = str(record["doi"]).strip().casefold()
    if expected_doi not in dois:
        raise ValueError(f"XML DOI does not match frozen record {record['pmcid']}")
    return meta


def _clean_prose(element: ET.Element) -> str:
    pieces: list[str] = []

    def visit(node: ET.Element) -> None:
        name = _local_name(node.tag)
        if name in EXCLUDED_ELEMENTS or _namespace(node.tag) == MATHML_NAMESPACE:
            return
        block = name in BLOCK_ELEMENTS
        if block:
            pieces.append(" ")
        if node.text:
            pieces.append(node.text)
        for child in node:
            visit(child)
            # A tail belongs to the parent and remains eligible even when the child subtree is not.
            if child.tail:
                pieces.append(child.tail)
        if block:
            pieces.append(" ")

    visit(element)
    return _normalize("".join(pieces))


def extract_pmc_document(
    xml_path: str | Path, record: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Verify one frozen source object, then extract only its abstract and article body prose."""
    path = Path(xml_path)
    raw = path.read_bytes()
    if len(raw) != record.get("xml_size_bytes"):
        raise ValueError(f"XML byte length differs from frozen record: {path.name}")
    if hashlib.md5(raw).hexdigest() != record.get("xml_md5"):
        raise ValueError(f"XML MD5 differs from frozen record: {path.name}")
    if _sha256(raw) != record.get("xml_sha256"):
        raise ValueError(f"XML SHA-256 differs from frozen record: {path.name}")
    if re.search(rb"<!\s*ENTITY\b", raw, re.IGNORECASE):
        raise ValueError(f"XML entity declarations are prohibited: {path.name}")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as error:
        raise ValueError(f"malformed XML source object {path.name}: {error}") from error
    if _local_name(root.tag) != "article":
        raise ValueError(f"XML root is not an article element: {path.name}")
    metadata = _article_metadata(root, record)
    abstracts = [
        child
        for child in metadata
        if _local_name(child.tag) == "abstract"
        and not child.attrib.get("abstract-type", "").strip()
    ]
    body = _direct_child(root, "body")
    if body is None:
        raise ValueError(f"XML lacks direct article/body for {record['pmcid']}")
    prose_parts = [_clean_prose(abstract) for abstract in abstracts]
    prose_parts.append(_clean_prose(body))
    document = _normalize(" ".join(part for part in prose_parts if part))
    if not document:
        raise ValueError(f"XML has no eligible abstract or article body prose: {record['pmcid']}")
    source_record = {
        "pmcid": record["pmcid"],
        "cloud_version": record["cloud_version"],
        "xml_size_bytes": len(raw),
        "xml_md5": hashlib.md5(raw).hexdigest(),
        "xml_sha256": _sha256(raw),
        "extracted_text_sha256": _sha256(document.encode("utf-8")),
        "normalized_identity_sha256": _identity_sha256(document),
        "extracted_characters": len(document),
        "extracted_utf8_bytes": len(document.encode("utf-8")),
    }
    return document, source_record


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def extract_preregistered_pmc_corpus(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    preparation_inventory: str | Path = Path("docs/q10h_pmc_corpus_preparation.json"),
) -> dict[str, Any]:
    """Extract a complete rights-cleared allowlist and write it atomically into ignored data/."""
    prep_path = Path(preparation_inventory).resolve()
    preparation, records = _validate_inventories(prep_path)
    source_root = Path(input_dir)
    target = Path(output_dir).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError(f"refusing to overwrite non-empty data output: {target}")

    documents: list[str] = []
    source_records: list[dict[str, Any]] = []
    attribution_records: list[dict[str, Any]] = []
    for record in records:
        xml_path = _record_input_path(source_root, record)
        document, result = extract_pmc_document(xml_path, record)
        documents.append(document)
        source_records.append(result)
        attribution_records.append(
            {
                "pmcid": record["pmcid"],
                "cloud_version": record["cloud_version"],
                "title": record["title"],
                "authors": record["authors"],
                "citation": record["citation"],
                "doi": record["doi"],
                "license_uri": record["license_uri"],
                "source_url": record["source_url"],
                "xml_size_bytes": record["xml_size_bytes"],
                "xml_md5": record["xml_md5"],
                "xml_sha256": record["xml_sha256"],
            }
        )
    if len(documents) < preparation["minimum_complete_documents"]:
        raise ValueError("complete extracted document count is below the preregistered minimum")

    corpus_bytes = ("\n".join(documents) + "\n").encode("utf-8")
    source5_sha256 = preparation["source5_inventory_sha256"]
    source_metadata = {
        "dataset": "PMC Open Access Subset, restricted abstract and main-body prose",
        "content_origin": "human",
        "license": "CC BY 4.0 for the restricted abstract/main-body prose scope cleared by Q10-SOURCE-007",
        "license_uri": CC_BY_4_URI,
        "rights_experiment_id": "Q10-SOURCE-007",
        "preparation_experiment_id": "Q10-PREP-002",
        "source_revision": "exact PMCID.version XML objects in the frozen Q10-PREP-002 allowlist",
        "source5_inventory_sha256": source5_sha256,
        "source7_inventory_sha256": preparation["input_inventory_sha256"],
        "attribution_required": preparation["source"]["required_attribution"],
        "nlm_acknowledgement": "This research uses PMC Open Access Subset content. NLM does not endorse this work.",
        "articles": attribution_records,
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "Q10-PREP-002",
        "preparation_inventory_sha256": _sha256(prep_path.read_bytes()),
        "rights_inventory_sha256": preparation["input_inventory_sha256"],
        "source5_inventory_sha256": source5_sha256,
        "extraction_contract": preparation["extraction_contract"],
        "document_count": len(documents),
        "corpus_sha256": _sha256(corpus_bytes),
        "corpus_utf8_bytes": len(corpus_bytes),
        "records": source_records,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        (staging / "corpus.txt").write_bytes(corpus_bytes)
        _write_json(staging / "extraction_manifest.json", manifest)
        _write_json(staging / "source_metadata.json", source_metadata)
        if target.exists():
            target.rmdir()
        os.replace(staging, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
