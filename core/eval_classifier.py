"""Evaluación del clasificador unificado contra el golden set anotado.

Uso:
    # AHORA (sin tocar nodes.py): valida el golden set por errores de anotación
    python -m core.eval_classifier --validate-only

    # DESPUÉS (con graph/nodes.py actualizado): evalúa el clasificador real
    python -m core.eval_classifier
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import pandas as pd

try:
    from core.contracts import VALID_LABELS, VALID_ROUTES
except ImportError:  # ejecución directa: python core/eval_classifier.py
    from contracts import VALID_LABELS, VALID_ROUTES

GOLDEN_PATH = Path("tests/golden_sentiment.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ClassifierFn = Callable[[str], dict]  # query -> {"sentiment": ..., ..., "route": ...}
LABEL_FIELDS = ("sentiment", "urgency", "intent", "category")
FIELDS = LABEL_FIELDS + ("route",)


# --------------------------------------------------------------------------
# Regla oficial de routing (debe ser IDÉNTICA a compute_route() del nodo)
# --------------------------------------------------------------------------
def expected_route(labels: dict) -> str:
    """Regla v4: intent decide; el sentimiento solo modula tono."""
    if labels["intent"] == "fuera_de_dominio":
        return "respuesta_fuera_dominio"
    if labels["intent"] == "hablar_humano":
        return "handoff_humano"
    return labels["intent"]



def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["examples"]


# --------------------------------------------------------------------------
# Validación del golden (label_validity + consistencia ruta↔etiquetas)
# --------------------------------------------------------------------------
def validate_golden(golden: list[dict]) -> list[str]:
    errors: list[str] = []
    ids = Counter()
    for ex in golden:
        _id = ex.get("id")
        ids[_id] += 1
        exp = ex.get("expected", {})
        for field in LABEL_FIELDS:
            if exp.get(field) not in VALID_LABELS[field]:
                errors.append(f"id={_id}: {field}='{exp.get(field)}' no es etiqueta válida")
        if exp.get("route") not in VALID_ROUTES:
            errors.append(f"id={_id}: route='{exp.get('route')}' no es una ruta válida")
        elif exp.get("route") != expected_route(exp):
            errors.append(
                f"id={_id}: route='{exp['route']}' incosistente con la regla "
                f"oficial (debiera ser '{expected_route(exp)}')"
            )
    dup = [i for i, c in ids.items() if c > 1]
    if dup:
        errors.append(f"ids duplicados: {dup}")
    return errors


# --------------------------------------------------------------------------
# Evaluación del clasificador
# --------------------------------------------------------------------------
def evaluate(classifier: ClassifierFn, golden: list[dict]) -> pd.DataFrame:
    rows = []
    for ex in golden:
        pred = classifier(ex["query"])
        row = {"id": ex["id"], "group": ex["group"], "query": ex["query"]}
        for field in FIELDS:
            row[f"pred_{field}"] = pred.get(field)
            row[f"{field}_acc"] = pred.get(field) == ex["expected"][field]
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    acc_cols = [f"{f}_acc" for f in FIELDS]
    print("\n" + "=" * 70)
    print("ACCURACY GLOBAL POR ETIQUETA")
    print("=" * 70)
    print(df[acc_cols].mean().round(3).to_string())

    print("\nACCURACY POR GRUPO (media de todas las etiquetas):")
    by_group = df.groupby("group")[acc_cols].mean().mean(axis=1).round(3)
    print(by_group.sort_values().to_string())

    print("\nMATRIZ DE CONFUSIÓN — intent:")
    print(pd.crosstab(df["pred_intent"], [ex["expected"]["intent"] for ex in load_golden()],
                      rownames=["pred"], colnames=["gold"], dropna=False))
    print("=" * 70)

    df.to_csv(RESULTS_DIR / "classifier_report.csv", index=False, encoding="utf-8-sig")
    print(f"Reporte guardado en {RESULTS_DIR / 'classifier_report.csv'}")


def _load_default_classifier() -> ClassifierFn:
    """Importa el nodo real una vez actualizado graph/nodes.py (fase posterior)."""
    from graph.nodes import analyze_sentiment  # noqa: import diferido a propósito
    return lambda query: analyze_sentiment({"query": query})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--validate-only", action="store_true",
                        help="Solo valida el golden set; no invoca el LLM.")
    args = parser.parse_args()

    golden = load_golden(args.golden)
    errors = validate_golden(golden)
    if errors:
        print(f"❌ Golden set con {len(errors)} problema(s):")
        print("\n".join(f"  - {e}" for e in errors))
        raise SystemExit(1)
    print(f"✅ Golden set válido: {len(golden)} ejemplos, etiquetas y rutas consistentes.")

    if args.validate_only:
        return

    classifier = _load_default_classifier()
    report(evaluate(classifier, golden))


if __name__ == "__main__":
    main()
