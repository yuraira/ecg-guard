"""Generate a CycloneDX SBOM from the installed ECG Guard environment."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


LICENSE_ALIASES = {
    "apache-2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "bsd": "BSD-3-Clause",
    "bsd 3-clause license": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "mit": "MIT",
    "mit license": "MIT",
}
CLASSIFIER_LICENSES = {
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
}


def distribution_name(distribution: metadata.Distribution) -> str:
    return canonicalize_name(distribution.metadata["Name"])


def active_requirements(
    distribution: metadata.Distribution,
) -> list[str]:
    """Return installed-environment requirements, excluding optional extras."""
    names: list[str] = []
    for raw_requirement in distribution.requires or ():
        requirement = Requirement(raw_requirement)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        names.append(canonicalize_name(requirement.name))
    return sorted(set(names))


def collect_distributions(
    root_name: str,
    *,
    direct_only: bool,
) -> tuple[metadata.Distribution, dict[str, metadata.Distribution]]:
    """Collect the root's direct or complete reachable dependency closure."""
    root = metadata.distribution(root_name)
    collected: dict[str, metadata.Distribution] = {}
    pending = active_requirements(root)
    while pending:
        name = pending.pop(0)
        if name in collected:
            continue
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"installed dependency is missing from SBOM environment: {name}"
            ) from error
        collected[name] = distribution
        if not direct_only:
            pending.extend(
                dependency
                for dependency in active_requirements(distribution)
                if dependency not in collected
            )
    return root, collected


def license_expression(distribution: metadata.Distribution) -> str | None:
    """Read a concise SPDX-like expression without copying long license text."""
    expression = distribution.metadata.get("License-Expression")
    if expression:
        return expression.strip()

    raw_license = distribution.metadata.get("License", "").strip()
    normalized = re.sub(r"\s+", " ", raw_license).lower()
    if normalized in LICENSE_ALIASES:
        return LICENSE_ALIASES[normalized]

    for classifier in distribution.metadata.get_all("Classifier", ()):
        if classifier in CLASSIFIER_LICENSES:
            return CLASSIFIER_LICENSES[classifier]
    return None


def project_references(
    distribution: metadata.Distribution,
) -> list[dict[str, str]]:
    """Convert Python project URLs into CycloneDX external references."""
    references: list[dict[str, str]] = []
    for entry in distribution.metadata.get_all("Project-URL", ()):
        label, separator, url = entry.partition(",")
        if not separator or not url.strip().startswith(("https://", "http://")):
            continue
        normalized_label = label.strip().lower()
        reference_type = (
            "vcs"
            if any(
                token in normalized_label
                for token in ("source", "repository", "code")
            )
            else "website"
        )
        references.append(
            {"type": reference_type, "url": url.strip()}
        )
    return references


def component_for(distribution: metadata.Distribution) -> dict[str, object]:
    """Build one Python package component."""
    name = distribution_name(distribution)
    version = distribution.version
    component: dict[str, object] = {
        "type": "library",
        "bom-ref": f"pkg:pypi/{name}@{quote(version, safe='.')}",
        "name": name,
        "version": version,
        "purl": f"pkg:pypi/{name}@{quote(version, safe='.')}",
    }
    license_id = license_expression(distribution)
    if license_id:
        component["licenses"] = [{"expression": license_id}]
    references = project_references(distribution)
    if references:
        component["externalReferences"] = references
    return component


def timestamp() -> str:
    """Use SOURCE_DATE_EPOCH when supplied, otherwise the current UTC time."""
    source_date_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if source_date_epoch:
        return datetime.fromtimestamp(
            int(source_date_epoch),
            tz=UTC,
        ).isoformat()
    return datetime.now(UTC).isoformat()


def create_bom(
    root_name: str,
    *,
    direct_only: bool,
) -> dict[str, object]:
    root, distributions = collect_distributions(
        root_name,
        direct_only=direct_only,
    )
    root_component = component_for(root)
    root_component["type"] = "application"
    components = [
        component_for(distributions[name])
        for name in sorted(distributions)
    ]
    references = [
        str(root_component["bom-ref"]),
        *(str(component["bom-ref"]) for component in components),
    ]
    reference_by_name = {
        canonicalize_name(str(component["name"])): str(component["bom-ref"])
        for component in components
    }

    dependency_entries = []
    root_dependencies = [
        reference_by_name[name]
        for name in active_requirements(root)
        if name in reference_by_name
    ]
    dependency_entries.append(
        {
            "ref": str(root_component["bom-ref"]),
            "dependsOn": sorted(root_dependencies),
        }
    )
    for name in sorted(distributions):
        dependency_entries.append(
            {
                "ref": reference_by_name[name],
                "dependsOn": sorted(
                    reference_by_name[dependency]
                    for dependency in active_requirements(distributions[name])
                    if dependency in reference_by_name
                ),
            }
        )

    serial = uuid.uuid5(uuid.NAMESPACE_URL, "\n".join(sorted(references)))
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp(),
            "tools": {
                "components": [
                    {
                        "type": "application",
                        "name": "ecg-guard-sbom-generator",
                        "version": "1.0",
                    }
                ]
            },
            "component": root_component,
            "properties": [
                {
                    "name": "ecg-guard:dependency-scope",
                    "value": "direct" if direct_only else "runtime-transitive",
                }
            ],
        },
        "components": components,
        "dependencies": dependency_entries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ecg-guard")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Include direct runtime dependencies but not their dependencies.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bom = create_bom(args.root, direct_only=args.direct_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bom, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"sbom={args.output} components={len(bom['components'])} "
        f"scope={'direct' if args.direct_only else 'runtime-transitive'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
