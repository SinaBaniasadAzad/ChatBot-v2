"""
Load the taxonomy from YAML into Python objects.

Every other part of the system (prompt, schema, evaluation) reads the
categories *only* from here, so adding a new label/layer in the YAML
automatically propagates everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from config.settings import settings


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    definition: str
    cues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Layer:
    id: str
    name: str
    description: str
    role: str
    labels: list[Label]

    @property
    def label_ids(self) -> list[str]:
        return [lbl.id for lbl in self.labels]

    def get_label(self, label_id: str) -> Label | None:
        for lbl in self.labels:
            if lbl.id == label_id:
                return lbl
        return None


@dataclass(frozen=True)
class Taxonomy:
    layers: list[Layer]
    signals: dict = field(default_factory=dict)

    @property
    def layer_ids(self) -> list[str]:
        return [layer.id for layer in self.layers]

    def get_layer(self, layer_id: str) -> Layer | None:
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        return None


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    path = path or settings.taxonomy_path
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    layers: list[Layer] = []
    for ldata in raw.get("layers", []):
        labels = [
            Label(
                id=str(lbl["id"]).strip(),
                name=lbl.get("name", lbl["id"]),
                definition=(lbl.get("definition") or "").strip(),
                cues=[str(c).strip() for c in lbl.get("cues", [])],
            )
            for lbl in ldata.get("labels", [])
        ]
        layers.append(
            Layer(
                id=str(ldata["id"]).strip(),
                name=ldata.get("name", ldata["id"]),
                description=(ldata.get("description") or "").strip(),
                role=ldata.get("role", ""),
                labels=labels,
            )
        )
    if not layers:
        raise ValueError("taxonomy.yaml defines no layers.")
    return Taxonomy(layers=layers, signals=raw.get("signals") or {})
