from pathlib import Path
import re


def test_compose_published_ports_are_loopback_only():
    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8")
    port_lines = [line.strip() for line in compose.splitlines() if re.match(r"\s*-\s*\"[^\"]+:\d+", line)]
    assert port_lines, "expected explicit published ports in docker-compose.yml"
    assert all('"127.0.0.1:' in line for line in port_lines), port_lines
    assert not any('"0.0.0.0:' in line or '"[::]:' in line for line in port_lines)


def test_compose_documents_internal_container_bind_override():
    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'ALLOW_NON_LOOPBACK: "1"' in compose
    assert "RESEARCH_API_HOST: 0.0.0.0" in compose
    assert '"127.0.0.1:8001:8001"' in compose


def test_compose_forwards_partial_ot_trust_configuration():
    compose = (Path(__file__).resolve().parents[2] / "docker-compose.yml").read_text(encoding="utf-8")
    assert "TRUST_MODE: ${TRUST_MODE:-heterogeneity}" in compose
    assert "OT_TRUST_TAU: ${OT_TRUST_TAU:-50.0}" in compose
    assert "OT_TRUST_K: ${OT_TRUST_K:-100.0}" in compose
