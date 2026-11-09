"""Small offline CPSWatch description builder for the paper's data contract.

The bundled schema is CPSWatch 1.0.1 (CC BY 4.0). The RDF extension records
column roles, order and acquisition identities. No reasoner or network request
is used. Process-flag connectivity is an induced, undirected view of explicitly
declared ``hasSystemConnection`` edges, never of component hierarchy.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import DCTERMS, OWL, XSD

from .datasets.common import digest, resource

CPSWATCH = Namespace("https://w3id.org/CPSWatch#")
CPSB = Namespace("https://w3id.org/CPSWatch/benchmark#")
SOSA = Namespace("http://www.w3.org/ns/sosa/")


def validate_manifest(manifest: dict) -> dict:
    features = list(manifest.get("features", []))
    context = list(manifest.get("context_columns", []))
    owners = list(manifest.get("owners", []))
    if not features or len(features) != len(set(features)):
        raise ValueError("An ordered, nonempty, unique sensor list is required")
    if len(features) != len(owners) or any(not x for x in owners):
        raise ValueError("Each model sensor must have exactly one component owner")
    if len(context) != len(set(context)) or set(features) & set(context):
        raise ValueError("Context channels must be unique and separate from model inputs")
    annotations = {"Label", "Normal/Attack", "DeviationID ValueY", "is_anomaly", "attack_number"}
    if annotations & (set(features) | set(context)):
        raise ValueError("Evaluator annotations cannot be sensor/context inputs")
    components = set(manifest.get("components", [])) | set(owners)
    for edge in manifest.get("connections", []):
        if len(edge) != 2 or not all(isinstance(x, str) and x for x in edge):
            raise ValueError("Each declared connection needs two named components")
        if any(x not in components for x in edge):
            raise ValueError(f"Connection has an undeclared endpoint: {edge}")
    identities = set()
    allowed = {"train", "validation", "calibration", "test_normal", "native"}
    for recording in manifest.get("recordings", []):
        if recording["id"] in identities:
            raise ValueError("Duplicate recording identity")
        identities.add(recording["id"])
        path = Path(recording["path"])
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Recording paths must be portable relative paths")
        previous = 0
        for segment in sorted(recording["segments"], key=lambda x: x["start"]):
            start, end = segment["start"], segment["end"]
            if not 0 <= start < end <= recording["rows"] or start < previous:
                raise ValueError("Recording segments overlap or exceed source bounds")
            if segment["split"] not in allowed:
                raise ValueError(f"Unknown split: {segment['split']}")
            if not recording["normal"] and segment["split"] != "native":
                raise ValueError("Fault recordings cannot enter normal partitions")
            previous = end
    return {
        "valid": True,
        "sensors": len(features),
        "context_channels": len(context),
        "components": len(components),
        "recordings": len(identities),
        "declared_connections": len(manifest.get("connections", [])),
        "graph_semantics": "Induced graph on peak owners using declared system connections only",
    }


def induced_connections(peak_components, connections) -> list[tuple[str, str]]:
    """Filter endpoints directly. Never infer paths through nonpeak components."""
    peaks = set(peak_components)
    return sorted({tuple(edge) for edge in connections if len(edge) == 2 and set(edge) <= peaks})


def connected_peak_components(peak_components, connections) -> bool:
    """Return true for at least two distinct owners connected in their induced graph."""
    peaks = set(peak_components)
    if len(peaks) < 2:
        return False
    adjacency = {owner: set() for owner in peaks}
    for left, right in induced_connections(peaks, connections):
        adjacency[left].add(right)
        adjacency[right].add(left)
    pending = [min(peaks)]
    seen = set()
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(adjacency[node] - seen)
    return seen == peaks


def build_description(manifest: dict) -> Graph:
    validate_manifest(manifest)
    graph = Graph()
    for prefix, namespace in (("cpswatch", CPSWATCH), ("cpsb", CPSB), ("sosa", SOSA)):
        graph.bind(prefix, namespace)
    graph.bind("dcterms", DCTERMS)
    graph.bind("owl", OWL)
    root = f"https://w3id.org/CPSWatch/safeprocess/instance/{quote(manifest['dataset'], safe='')}/"
    plant = URIRef(root + "plant")
    ontology = URIRef(root + "description")
    graph.add((ontology, RDF.type, OWL.Ontology))
    graph.add((ontology, OWL.imports, URIRef("https://w3id.org/CPSWatch/1.0.1")))
    graph.add((ontology, DCTERMS.title, Literal(f"SafeProcess {manifest['dataset']} acquisition description")))
    graph.add((plant, RDF.type, CPSWATCH.MonitoredSystem))
    graph.add((plant, CPSB.datasetIdentifier, Literal(manifest["dataset"])))
    graph.add((plant, CPSB.schemaIdentifier, Literal(manifest["schema_version"])))
    for name in ("channelRole", "featureIndex", "generatedFromColumn", "sourceFile", "sha256", "condition", "split", "datasetIdentifier", "schemaIdentifier"):
        graph.add((CPSB[name], RDF.type, OWL.DatatypeProperty))
    for name in ("hasComponent", "hasChannel", "hasRecording"):
        graph.add((CPSB[name], RDF.type, OWL.ObjectProperty))
    for name in ("ControlStateChannel", "DatasetRecording"):
        graph.add((CPSB[name], RDF.type, OWL.Class))
    owners = set(manifest.get("components", [])) | set(manifest["owners"])
    component_uri = {name: URIRef(root + "component/" + quote(name, safe="")) for name in owners}
    for name, node in sorted(component_uri.items()):
        graph.add((node, RDF.type, CPSWATCH.MonitoredSystem))
        graph.add((node, RDFS.label, Literal(name)))
        graph.add((plant, CPSB.hasComponent, node))
    for left, right in manifest.get("connections", []):
        graph.add((component_uri[left], CPSWATCH.hasSystemConnection, component_uri[right]))
    descriptions = {row["name"]: row for row in manifest.get("channel_descriptions", [])}
    for index, (name, owner) in enumerate(zip(manifest["features"], manifest["owners"])):
        sensor = URIRef(root + "channel/" + quote(name, safe=""))
        graph.add((sensor, RDF.type, CPSWATCH.ContinuousMonitoringSensor))
        graph.add((sensor, RDF.type, CPSWATCH.MonitoringSensor))
        graph.add((sensor, RDFS.label, Literal(name)))
        graph.add((sensor, CPSB.generatedFromColumn, Literal(name)))
        graph.add((sensor, CPSB.channelRole, Literal("sensor")))
        graph.add((sensor, CPSB.featureIndex, Literal(index, datatype=XSD.nonNegativeInteger)))
        graph.add((component_uri[owner], CPSWATCH.hasMonitoringSensor, sensor))
        graph.add((plant, CPSB.hasChannel, sensor))
        description = descriptions.get(name, {})
        if description.get("observable_property"):
            observed_property = URIRef(root + "property/" + quote(description["observable_property"], safe=""))
            graph.add((observed_property, RDF.type, SOSA.ObservableProperty))
            graph.add((observed_property, RDFS.label, Literal(description["observable_property"])))
            graph.add((sensor, SOSA.observes, observed_property))
        if description.get("unit_uri"):
            graph.add((sensor, CPSWATCH.hasUnit, URIRef(description["unit_uri"])))
        if description.get("description"):
            graph.add((sensor, RDFS.comment, Literal(description["description"])))
        if description.get("evidence"):
            graph.add((sensor, DCTERMS.source, Literal(description["evidence"])))
    for name in manifest["context_columns"]:
        node = URIRef(root + "context/" + quote(name, safe=""))
        graph.add((node, RDF.type, CPSB.ControlStateChannel))
        graph.add((node, CPSB.generatedFromColumn, Literal(name)))
        graph.add((node, CPSB.channelRole, Literal("operating_context")))
        graph.add((plant, CPSB.hasChannel, node))
    for row in manifest["recordings"]:
        node = URIRef(root + "recording/" + quote(row["id"], safe=""))
        graph.add((node, RDF.type, CPSB.DatasetRecording))
        graph.add((node, CPSB.sourceFile, Literal(row["path"])))
        graph.add((node, CPSB.sha256, Literal(row["sha256"])))
        graph.add((node, CPSB.condition, Literal("normal" if row["normal"] else "native")))
        graph.add((plant, CPSB.hasRecording, node))
        for segment in row["segments"]:
            graph.add((node, CPSB["split"], Literal(json.dumps(segment, sort_keys=True))))
    return graph


def write_description(manifest: dict, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    validation = validate_manifest(manifest)
    target = out / "ontology.ttl"
    build_description(manifest).serialize(target, format="turtle")
    # Parsing the just-written description is an offline syntax check. OWL
    # imports are descriptive and never fetched.
    Graph().parse(target, format="turtle")
    validation["ontology_sha256"] = digest(target)
    validation["schema_sha256"] = digest(resource("CPSWatch-1.0.1.ttl"))
    (out / "ontology-validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation
