import hashlib
import json

import pytest

from frontier.cli import main
from frontier.data.pmc import extract_pmc_document
from frontier.tokenization import ByteBPETokenizer, load_tokenizer_artifact, tokenizer_artifact
from frontier.tokenization.bpe import fit_byte_bpe_from_file


def _source_record(raw: bytes, pmcid: str = "PMC123") -> dict:
    return {
        "pmcid": pmcid,
        "cloud_version": 1,
        "xml_size_bytes": len(raw),
        "xml_md5": hashlib.md5(raw).hexdigest(),
        "xml_sha256": hashlib.sha256(raw).hexdigest(),
        "title": "A title",
        "doi": "10.1000/example",
    }


def _xml(prefix: str = "") -> bytes:
    return (
        prefix
        + """<article>
<front><article-meta>
<article-id pub-id-type="pmc">PMC123</article-id>
<article-id pub-id-type="doi">10.1000/example</article-id>
<title-group><article-title>A <italic>title</italic></article-title></title-group>
<abstract><p>Primary <italic>inline</italic> prose <xref>citation</xref>tail.</p></abstract>
<abstract abstract-type="web-summary"><p>Web summary leak.</p></abstract>
</article-meta></front>
<body><sec><title>Heading</title><p>Body <xref>reference leak</xref> remains. <fig><caption>Figure leak.</caption></fig> after.</p>
<table-wrap><caption>Table leak.</caption></table-wrap><inline-formula>formula leak</inline-formula>
<p>Second paragraph.</p></sec><ack>Acknowledgement leak.</ack></body>
<back><ref-list><ref>Reference leak.</ref></ref-list></back>
</article>"""
    ).encode("utf-8")


def _reference_content_encode(text: str, merges: tuple[tuple[int, int], ...]) -> list[int]:
    tokens = list(text.encode("utf-8"))
    for pair in merges:
        output = []
        index = 0
        while index < len(tokens):
            if index + 1 < len(tokens) and (tokens[index], tokens[index + 1]) == pair:
                output.append(259 + merges.index(pair))
                index += 2
            else:
                output.append(tokens[index])
                index += 1
        tokens = output
    return tokens


def test_byte_bpe_fit_is_deterministic_and_round_trips_unicode(tmp_path):
    documents = ["abab café", "baba 🐇", "abab café"]
    first = ByteBPETokenizer.fit(
        documents,
        training_source_sha256="a" * 64,
        target_vocab_size=280,
        min_pair_frequency=2,
    )
    second = ByteBPETokenizer.fit(
        documents,
        training_source_sha256="a" * 64,
        target_vocab_size=280,
        min_pair_frequency=2,
    )

    assert first.to_dict() == second.to_dict()
    assert first.merges[0] == (ord("a"), ord("b"))
    assert first.vocab_size <= first.target_vocab_size
    for document in documents:
        ids = first.encode(document)
        assert ids[1:-1] == _reference_content_encode(document, first.merges)
        assert first.decode(ids) == document
        assert sum(first.token_byte_counts(document)) == len(document.encode("utf-8"))


def test_byte_bpe_never_merges_across_documents_and_artifact_round_trips(tmp_path):
    tokenizer = ByteBPETokenizer.fit(
        ["a", "b"],
        training_source_sha256="b" * 64,
        target_vocab_size=260,
        min_pair_frequency=1,
    )
    artifact_path = tmp_path / "tokenizer.json"
    artifact_path.write_text(json.dumps(tokenizer_artifact(tokenizer)), encoding="utf-8")

    loaded = load_tokenizer_artifact(artifact_path)

    assert tokenizer.vocab_size == 259
    assert loaded.to_dict() == tokenizer.to_dict()

    artifact = tokenizer.to_dict()
    artifact["min_pair_frequency"] = True
    with pytest.raises(TypeError, match="must be an integer"):
        ByteBPETokenizer.from_dict(artifact)


def test_fit_tokenizer_cli_accepts_only_a_train_split_file(tmp_path, capsys):
    train_path = tmp_path / "train.txt"
    raw = b"training words\ntraining words again\n"
    train_path.write_bytes(raw)
    output = tmp_path / "tokenizer.json"

    assert main(["fit-tokenizer", "--train-file", str(train_path), "--output", str(output)]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    loaded = load_tokenizer_artifact(output)
    assert artifact["training_source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert artifact["training_document_count"] == 2
    assert artifact["training_utf8_bytes"] == len(b"training words") + len(b"training words again")
    assert loaded.name == "byte-bpe-v1"
    assert json.loads(capsys.readouterr().out)["artifact_sha256"] == artifact["artifact_sha256"]

    validation_path = tmp_path / "validation.txt"
    validation_path.write_text("held out\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "fit-tokenizer",
                "--train-file",
                str(validation_path),
                "--output",
                str(tmp_path / "invalid.json"),
            ]
        )


def test_byte_bpe_file_fit_records_exact_training_input_hash(tmp_path):
    source = tmp_path / "train.txt"
    source.write_bytes(b"alpha alpha\nbeta beta\n")

    tokenizer = fit_byte_bpe_from_file(source)

    assert tokenizer.training_source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert tokenizer.training_document_count == 2
    assert len(tokenizer.training_document_identity_sha256) == 2


def test_pmc_extractor_includes_only_preregistered_abstract_and_body_prose(tmp_path):
    xml = _xml()
    source = tmp_path / "article.xml"
    source.write_bytes(xml)

    document, record = extract_pmc_document(source, _source_record(xml))

    assert document == "Primary inline prose tail. Heading Body remains. after. Second paragraph."
    for excluded in (
        "Web summary",
        "citation",
        "reference leak",
        "Figure leak",
        "Table leak",
        "formula leak",
        "Acknowledgement",
        "Reference leak",
        "A title",
    ):
        assert excluded not in document
    assert record["extracted_text_sha256"] == hashlib.sha256(document.encode("utf-8")).hexdigest()
    assert (
        record["normalized_identity_sha256"] == hashlib.sha256(document.encode("utf-8")).hexdigest()
    )


def test_pmc_extractor_does_not_resolve_external_dtd_and_rejects_entity_declarations(tmp_path):
    external_dtd = _xml(
        '<?xml version="1.0"?><!DOCTYPE article SYSTEM "http://127.0.0.1:9/missing.dtd">'
    )
    external_path = tmp_path / "external-dtd.xml"
    external_path.write_bytes(external_dtd)
    document, _ = extract_pmc_document(external_path, _source_record(external_dtd))
    assert document.startswith("Primary inline prose")

    declared_entity = _xml('<!DOCTYPE article [<!ENTITY leak SYSTEM "file:///forbidden">]>')
    entity_path = tmp_path / "entity.xml"
    entity_path.write_bytes(declared_entity)
    with pytest.raises(ValueError, match="entity declarations are prohibited"):
        extract_pmc_document(entity_path, _source_record(declared_entity))


def test_pmc_extractor_checks_frozen_hash_and_identity(tmp_path):
    xml = _xml()
    source = tmp_path / "article.xml"
    source.write_bytes(xml)
    record = _source_record(xml)

    altered = dict(record, xml_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 differs"):
        extract_pmc_document(source, altered)
    with pytest.raises(ValueError, match="PMCID does not match"):
        extract_pmc_document(source, _source_record(xml, pmcid="PMC999"))
