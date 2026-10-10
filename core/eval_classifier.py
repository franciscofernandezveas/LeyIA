"""Evaluación del clasificador unificado contra el golden set anotado.

v2 — Alineado con contracts.py v7 y prompts.yaml v3.0.0:
     + Regla de routing v7 compuesta (reproduce el 100% del golden v1.0:
       intent decide; negativo + urgencia≥media + agendar_asesoria → handoff).
     + Columnas gold_* en el DataFrame: matrices de confusión y CSV
       autocontenidos. Corrige el bug de report() que releía el golden por
       defecto ignorando --golden.
     + Métricas por clase (intent/sentiment), recall/precision de escalamiento,
       exact-match, baseline trivial, tasa de fallback y latencia por ejemplo.
     + Resiliencia: excepciones del LLM por ejemplo no matan la corrida.
     + Validación de meta del golden (warnings, no bloquean) y de llaves
       requeridas por ejemplo.
     + Chequeo route_rule_ok: detecta si el nodo viola su propia regla de
       routing con sus propias etiquetas (bug de cableado del nodo).

Uso:
    python -m core.eval_classifier --validate-only   # valida el golden (sin LLM)
    python -m core.eval_classifier                    # evaluación completa
"""
from __future__ import annotations

import argparse
import json
import time
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

# ⚠️ Calce triple: esta constante ↔ golden.meta.rule_version ↔ compute_route()
# del nodo. El dueño único de la regla debe ser contracts.compute_route
# (pendiente); hasta entonces, cualquier cambio aquí se replica allá.
RULE_VERSION = "v7"
ROUTE_ESCALAMIENTO = "handoff_humano"

ClassifierFn = Callable[[str], dict]
LABEL_FIELDS = ("sentiment", "urgency", "intent", "category")
FIELDS = LABEL_FIELDS + ("route",)
REQUIRED_KEYS = ("id", "query", "group", "expected")


# --------------------------------------------------------------------------
# Regla oficial de routing v7 (temporalmente aquí; dueño final: contracts)
# --------------------------------------------------------------------------
def expected_route(labels: dict) -> str:
    """v7 compuesta:
      fuera_de_dominio → respuesta_fuera_dominio (el sentimiento NUNCA saca
                          de dominio)
      hablar_humano    → handoff_humano (como el golden; la semántica
                          contracts-v6 'hablar_humano → intake_lead' quedó
                          pendiente: si el nodo v9 la aplica, hay que migrar
                          ids 51–55 del golden a route=intake_lead)
      negativo + urgencia≥media + agendar_asesoria → handoff_humano
                          (escalamiento por emoción EXIGE coacción corroborante;
                          el enojo solo no deriva a nadie)
      resto            → intent
    """
    if labels.get("intent") == "fuera_de_dominio":
        return "respuesta_fuera_dominio"
    if labels.get("intent") == "hablar_humano":
        return ROUTE_ESCALAMIENTO
    if (labels.get("sentiment") == "negativo"
            and labels.get("urgency") in {"media", "alta"}
            and labels.get("intent") == "agendar_asesoria"):
        return ROUTE_ESCALAMIENTO
    return labels.get("intent")


def load_golden(path: Path = GOLDEN_PATH) -> tuple[dict, list[dict]]:
    """Retorna (meta, examples)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("meta", {}), data["examples"]


# --------------------------------------------------------------------------
# Validaciones
# --------------------------------------------------------------------------
def validate_meta(meta: dict, golden: list[dict]) -> list[str]:
    """Inconsistencias meta ↔ contenido. WARNINGS: informativos, no bloquean."""
    warnings: list[str] = []
    if (n := meta.get("n_examples")) is not None and n != len(golden):
        warnings.append(f"meta.n_examples={n} ≠ {len(golden)} ejemplos reales")
    for field, declared in (meta.get("distribution") or {}).items():
        actual = Counter(ex["expected"].get(field) for ex in golden)
        for k, v in declared.items():
            if actual.get(k, 0) != v:
                warnings.append(
                    f"meta.distribution.{field}.{k}: declarado={v}, real={actual.get(k, 0)}"
                )
        if (extra := sorted(set(actual) - set(declared))):
            warnings.append(f"meta.distribution.{field}: clases no declaradas: {extra}")
    if (rv := meta.get("rule_version")) is None:
        warnings.append(f"meta.rule_version ausente; este eval asume {RULE_VERSION}")
    elif rv != RULE_VERSION:
        warnings.append(
            f"meta.rule_version='{rv}' ≠ eval '{RULE_VERSION}': "
            f"¿golden anotado con otra regla de routing?"
        )
    return warnings


def validate_golden(golden: list[dict]) -> list[str]:
    """Errores bloqueantes: llaves faltantes, etiquetas inválidas, ids
    duplicados y consistencia ruta↔etiquetas contra la regla oficial."""
    errors: list[str] = []
    ids = Counter()
    for ex in golden:
        _id = ex.get("id")
        if missing := [k for k in REQUIRED_KEYS if k not in ex]:
            errors.append(f"id={_id}: faltan llaves requeridas {missing}")
            continue  # sin 'expected' no hay nada más que validar
        ids[_id] += 1
        exp = ex["expected"]
        for field in LABEL_FIELDS:
            if exp.get(field) not in VALID_LABELS[field]:
                errors.append(
                    f"id={_id}: {field}='{exp.get(field)}' no es etiqueta válida"
                )
        route = exp.get("route")
        if route not in VALID_ROUTES:
            errors.append(f"id={_id}: route='{route}' no es una ruta válida")
        elif route != expected_route(exp):
            errors.append(
                f"id={_id}: route='{route}' inconsistente con regla {RULE_VERSION} "
                f"(debiera ser '{expected_route(exp)}')"
            )
    if dup := [i for i, c in ids.items() if c > 1 and i is not None]:
        errors.append(f"ids duplicados: {dup}")
    return errors


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
def evaluate(classifier: ClassifierFn, golden: list[dict]) -> pd.DataFrame:
    rows = []
    for i, ex in enumerate(golden, 1):
        t0 = time.perf_counter()
        try:
            pred = classifier(ex["query"]) or {}
            error = None
        except Exception as e:  # timeout/429/500 del LLM: no matar la corrida
            pred, error = {}, f"{type(e).__name__}: {e}"
        latency_ms = int((time.perf_counter() - t0) * 1000)

        row = {
            "id": ex["id"], "group": ex["group"], "query": ex["query"],
            "notes": ex.get("notes", ""),
            "clf_reason": pred.get("clf_reason"),
            "latency_ms": latency_ms, "error": error,
        }
        for field in FIELDS:
            row[f"gold_{field}"] = ex["expected"].get(field)
            row[f"pred_{field}"] = pred.get(field)
            row[f"{field}_acc"] = pred.get(field) == ex["expected"].get(field)
        # ¿El clasificador cumple su propia regla con sus propias etiquetas?
        if all(pred.get(f) for f in LABEL_FIELDS) and pred.get("route"):
            row["route_rule_ok"] = pred.get("route") == expected_route(pred)
        else:
            row["route_rule_ok"] = None
        rows.append(row)
        if i % 10 == 0:
            print(f"  ... {i}/{len(golden)}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reporte
# --------------------------------------------------------------------------
def _per_class_metrics(df: pd.DataFrame, field: str) -> pd.DataFrame:
    labels = sorted(
        set(df[f"gold_{field}"].dropna()) | set(df[f"pred_{field}"].dropna())
    )
    rows = []
    for lab in labels:
        tp = int(((df[f"gold_{field}"] == lab) & (df[f"pred_{field}"] == lab)).sum())
        fp = int(((df[f"gold_{field}"] != lab) & (df[f"pred_{field}"] == lab)).sum())
        fn = int(((df[f"gold_{field}"] == lab) & (df[f"pred_{field}"] != lab)).sum())
        rows.append({
            "clase": lab, "soporte": tp + fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else float("nan"),
            "recall": round(tp / (tp + fn), 3) if tp + fn else float("nan"),
        })
    return pd.DataFrame(rows).set_index("clase")


def report(df: pd.DataFrame) -> None:
    acc_cols = [f"{f}_acc" for f in FIELDS]

    print("\n" + "=" * 72)
    print(f"EVALUACIÓN CLASIFICADOR — regla {RULE_VERSION} — {len(df)} ejemplos")
    print("=" * 72)

    print("\n■ ACCURACY GLOBAL POR ETIQUETA")
    print(df[acc_cols].mean().round(3).to_string())
    print(f"\nexact-match (las 5 correctas): {df[acc_cols].all(axis=1).mean():.3f}")

    print("\n■ BASELINE TRIVIAL (predecir siempre la clase mayoritaria) — contexto:")
    for f in LABEL_FIELDS + ("route",):
        vc = df[f"gold_{f}"].value_counts(normalize=True)
        print(f"  {f}: siempre '{vc.idxmax()}' → {vc.iloc[0]:.3f}")

    print("\n■ MÉTRICAS POR CLASE — intent:")
    print(_per_class_metrics(df, "intent").to_string())
    print("\n■ MÉTRICAS POR CLASE — sentiment:")
    print(_per_class_metrics(df, "sentiment").to_string())

    gold_esc = df["gold_route"] == ROUTE_ESCALAMIENTO
    tp = int(((df["pred_route"] == ROUTE_ESCALAMIENTO) & gold_esc).sum())
    n_pred = int((df["pred_route"] == ROUTE_ESCALAMIENTO).sum())
    print(f"\n■ ESCALAMIENTO ({ROUTE_ESCALAMIENTO}): "
          f"precision={tp / n_pred:.3f} ({tp}/{n_pred}), "
          f"recall={tp / max(1, int(gold_esc.sum())):.3f} ({tp}/{int(gold_esc.sum())} casos gold)")

    print("\n■ ACCURACY POR GRUPO (media de las 5 etiquetas):")
    by_group = df.groupby("group")[acc_cols].mean().mean(axis=1).round(3)
    print(by_group.sort_values().to_string())

    print("\n■ MATRIZ DE CONFUSIÓN — intent:")
    print(pd.crosstab(df["pred_intent"], df["gold_intent"],
                      rownames=["pred"], colnames=["gold"], dropna=False))
    print("\n■ MATRIZ DE CONFUSIÓN — sentiment:")
    print(pd.crosstab(df["pred_sentiment"], df["gold_sentiment"],
                      rownames=["pred"], colnames=["gold"], dropna=False))
    print("\n■ MATRIZ DE CONFUSIÓN — route:")
    print(pd.crosstab(df["pred_route"], df["gold_route"],
                      rownames=["pred"], colnames=["gold"], dropna=False))

    violations = df[df["route_rule_ok"] == False]  # noqa: E712
    if len(violations):
        print(f"\n⚠️ VIOLACIONES DE REGLA DE ROUTING POR EL NODO: {len(violations)} "
              f"(el nodo contradijo su propia regla — bug de cableado, no de clasificación)")
        print(violations[["id", "pred_intent", "pred_route"]].to_string(index=False))

    fb_rate = df["clf_reason"].astype(str).str.startswith("fallback por").mean()
    n_err = int(df["error"].notna().sum())
    print(f"\n■ SALUD DE CORRIDA: fallback={fb_rate:.1%} · errores LLM={n_err} · "
          f"latencia p50={int(df['latency_ms'].median())}ms "
          f"p95={int(df['latency_ms'].quantile(0.95))}ms")

    fail = df[~df[acc_cols].all(axis=1)]
    print(f"\n■ FALLOS ({len(fail)}/{len(df)}):")
    for _, r in fail.iterrows():
        diffs = "; ".join(
            f"{f}: gold={r[f'gold_{f}']!r} → pred={r[f'pred_{f}']!r}"
            for f in FIELDS if not r[f"{f}_acc"]
        )
        note = f"\n    nota: {r['notes']}" if r["notes"] else ""
        err = f"\n    ERROR: {r['error']}" if r["error"] else ""
        reason = f"\n    reason: {r['clf_reason']}" if r["clf_reason"] else ""
        print(f"  id={r['id']} [{r['group']}] {r['query'][:90]}\n    {diffs}{reason}{note}{err}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "classifier_report.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("=" * 72)
    print(f"Reporte guardado en {out}")


def _load_default_classifier() -> ClassifierFn:
    """Importa el nodo real una vez actualizado graph/nodes.py."""
    from graph.nodes import analyze_sentiment  # noqa: import diferido a propósito
    return lambda query: analyze_sentiment({"query": query})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    parser.add_argument("--validate-only", action="store_true",
                        help="Solo valida el golden set; no invoca el LLM.")
    args = parser.parse_args()

    meta, golden = load_golden(args.golden)

    if warnings := validate_meta(meta, golden):
        print(f"⚠️ Meta del golden con {len(warnings)} inconsistencia(s) "
              f"(informativas, no bloquean):")
        print("\n".join(f"  - {w}" for w in warnings))

    errors = validate_golden(golden)
    if errors:
        print(f"❌ Golden set con {len(errors)} problema(s):")
        print("\n".join(f"  - {e}" for e in errors))
        raise SystemExit(1)
    print(f"✅ Golden set válido: {len(golden)} ejemplos, "
          f"etiquetas y rutas consistentes (regla {RULE_VERSION}).")

    if args.validate_only:
        return

    classifier = _load_default_classifier()
    report(evaluate(classifier, golden))


if __name__ == "__main__":
    main()
