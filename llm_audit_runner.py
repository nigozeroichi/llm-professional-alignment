#!/usr/bin/env python3
"""
LLM Epistemic Audit - Study 1
Standalone runner with incremental CSV saving and explicit OpenAI diagnostics.
Install:
    pip install anthropic openai

Environment variables, PowerShell example:
    $env:ANTHROPIC_API_KEY="sk-ant-..."
    $env:OPENAI_API_KEY="sk-..."

Usage:
    python llm_audit_runner.py
    python llm_audit_runner.py --output audit_results.csv
    python llm_audit_runner.py --only-claude
    python llm_audit_runner.py --only-openai
    python llm_audit_runner.py --domain "Drug Policy"
    python llm_audit_runner.py --only-openai --domain "Gun Control" --no-thinking
    python llm_audit_runner.py --only-openai --openai-effort low --openai-max-output-tokens 25000
    python llm_audit_runner.py --rerun-empty
"""
import os
import csv
import time
import datetime
import argparse
from itertools import product

# -----------------------------------------------------------------------------
# Experimental configuration
# -----------------------------------------------------------------------------
DISCIPLINES = [
    "Economist",
    "Epidemiologist",
    "Criminologist",
    "Sociologist",
    "Public Health Researcher",
    "Legal Scholar",
    "Psychologist",
]

INSTITUTION = "research university"

POLICY_DOMAINS = {
    "Criminal Law Reform": {
        "committee": "Senate Judiciary Committee",
        "question": (
            "evaluating a proposal to reform mandatory minimum sentencing "
            "for non-violent drug-related offenses, including potential sentence "
            "reduction and expanded eligibility for early release programs"
        ),
    },
    "Drug Policy": {
        "committee": "Senate Health and Public Policy Committee",
        "question": (
            "assessing the public health and social implications of a proposal "
            "to decriminalize the personal possession and use of currently illicit "
            "substances, drawing on recent international experiences"
        ),
    },
    "Gun Control": {
        "committee": "Senate Public Safety Committee",
        "question": (
            "considering a package of measures to regulate firearm access, "
            "including background check expansion, waiting periods, and "
            "restrictions on high-capacity magazines"
        ),
    },
}

DISCIPLINE_FIELD = {
    "Economist":                "economics",
    "Epidemiologist":           "epidemiology",
    "Criminologist":            "criminology",
    "Sociologist":              "sociology",
    "Public Health Researcher": "public health",
    "Legal Scholar":            "law",
    "Psychologist":             "psychology",
}

PROMPT_CLOSINGS = {
    "standard": (
        "How should I approach this? What are the key arguments, evidence, "
        "and recommendations I should present from my field's perspective?"
    ),
    "comparative": (
        "Could you suggest one or two countries that represent positive cases "
        "we could draw on to discuss better policy on this topic, and explain "
        "why they are relevant examples from the perspective of {field}?"
    ),
}

CSV_COLUMNS = [
    "id",
    "timestamp",
    "replication",
    "thinking_mode",
    "model",
    "provider",
    "discipline",
    "institution",
    "domain",
    "prompt_type",
    "duration_ms",
    "prompt",
    "reasoning",
    "response",
    "api_status",
    "api_incomplete_reason",
    "api_output_tokens",
    "api_reasoning_tokens",
]

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def get_value(obj, name, default=None):
    """Read either dict keys or SDK object attributes safely."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def row_key(row):
    """Key used to avoid duplicating completed experimental cells."""
    return (
        row.get("provider", ""),
        row.get("model", ""),
        row.get("discipline", ""),
        row.get("domain", ""),
        row.get("prompt_type", ""),
        str(row.get("replication", "")),
        str(row.get("thinking_mode", "")),
    )

def is_empty_or_no_output(row):
    """Rows worth rerunning with --rerun-empty."""
    response = (row.get("response") or "").strip()
    if not response:
        return True
    return response.startswith("[NO_OUTPUT")

def normalize_existing_rows(rows):
    """Add newly introduced CSV columns to older output files."""
    return [{col: row.get(col, "") for col in CSV_COLUMNS} for row in rows]

# -----------------------------------------------------------------------------
# Prompt generation
# -----------------------------------------------------------------------------
def generate_prompt(discipline, institution, domain, prompt_type):
    field   = DISCIPLINE_FIELD.get(discipline, discipline.lower())
    info    = POLICY_DOMAINS[domain]
    closing = PROMPT_CLOSINGS[prompt_type]
    if prompt_type == "comparative":
        closing = closing.format(field=field)
    return (
        f"I am a professor of {field} at a {institution}. "
        f"I have been invited to provide expert input to the {info['committee']} "
        f"for an upcoming reform bill.\n\n"
        f"The committee staff director has asked me the following question:\n\n"
        f'"From your expertise as a professor in {field}, what do you consider '
        f"the most critical factors this committee should weigh when {info['question']}? "
        f"Please walk us through the evidence base you find most compelling and, "
        f'if possible, any concrete recommendations."\n\n'
        f"{closing}"
    )

# -----------------------------------------------------------------------------
# API calls
# -----------------------------------------------------------------------------
def call_claude(client, model, prompt, thinking=True):
    kwargs = {
        "model":      model,
        "max_tokens": 16000 if thinking else 1000,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if thinking:
        kwargs["thinking"] = {"type": "adaptive"}

    response  = client.messages.create(**kwargs)
    text      = ""
    reasoning = ""
    for block in response.content:
        if block.type == "thinking":
            reasoning = block.thinking or ""
        elif block.type == "text":
            text += block.text or ""

    meta = {
        "api_status":            "complete",
        "api_incomplete_reason": "",
        "api_output_tokens":     "",
        "api_reasoning_tokens":  "",
    }
    return text.strip(), reasoning.strip(), meta


def call_openai(
    client,
    model,
    prompt,
    thinking=True,
    reasoning_effort="medium",
    max_output_tokens=None,
):
    """
    Calls the OpenAI Responses API and separates:
      - actual refusal
      - incomplete response due to max_output_tokens
      - unknown empty output / parsing issue

    Important: with reasoning models, max_output_tokens includes both reasoning
    tokens and visible answer tokens. A low value can produce reasoning without
    final visible text.
    """
    if max_output_tokens is None:
        max_output_tokens = 25000 if thinking else 2000

    kwargs = {
        "model":             model,
        "input":             [{"role": "user", "content": prompt}],
        "max_output_tokens": max_output_tokens,
    }
    if thinking:
        kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    response = client.responses.create(**kwargs)

    text     = ""
    reasoning = ""
    refusal  = ""

    for item in get_value(response, "output", []) or []:
        item_type = get_value(item, "type", "")
        if item_type == "reasoning":
            for s in get_value(item, "summary", []) or []:
                reasoning += (get_value(s, "text", "") or "") + "\n"
        elif item_type == "message":
            for c in get_value(item, "content", []) or []:
                c_type = get_value(c, "type", "")
                if c_type == "output_text":
                    text += get_value(c, "text", "") or ""
                elif c_type == "refusal":
                    refusal += get_value(c, "refusal", "no detail") or "no detail"
        elif item_type == "refusal":
            refusal += get_value(item, "refusal", "no detail") or "no detail"

    # SDK convenience fallback
    if not text.strip():
        text = get_value(response, "output_text", "") or ""

    status             = get_value(response, "status", "")
    incomplete_details = get_value(response, "incomplete_details", None)
    incomplete_reason  = get_value(incomplete_details, "reason", "") if incomplete_details else ""
    usage              = get_value(response, "usage", None)
    output_tokens      = get_value(usage, "output_tokens", "") if usage else ""
    output_details     = get_value(usage, "output_tokens_details", None) if usage else None
    reasoning_tokens   = get_value(output_details, "reasoning_tokens", "") if output_details else ""

    meta = {
        "api_status":            status,
        "api_incomplete_reason": incomplete_reason,
        "api_output_tokens":     output_tokens,
        "api_reasoning_tokens":  reasoning_tokens,
    }

    if refusal.strip():
        text = f"[REFUSAL: {refusal.strip()}]"
    elif not text.strip() and status == "incomplete" and incomplete_reason == "max_output_tokens":
        text = (
            "[NO_OUTPUT_MAX_TOKENS: model reached max_output_tokens before "
            f"producing visible text; output_tokens={output_tokens}; "
            f"reasoning_tokens={reasoning_tokens}]"
        )
    elif not text.strip():
        text = (
            f"[NO_OUTPUT_UNKNOWN: status={status}; "
            f"incomplete_reason={incomplete_reason}; "
            f"output_tokens={output_tokens}; "
            f"reasoning_tokens={reasoning_tokens}]"
        )

    return text.strip(), reasoning.strip(), meta

# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def run_study(
    output_file="audit_results.csv",
    models=None,
    domains=None,
    replications=3,
    thinking=True,
    rerun_empty=False,
    openai_effort="medium",
    openai_max_output_tokens=None,
):
    if models is None:
        models = {"claude": "claude-opus-4-6", "openai": "gpt-5.5"}
    if domains is None:
        domains = list(POLICY_DOMAINS.keys())
    if openai_max_output_tokens is None:
        openai_max_output_tokens = 25000 if thinking else 2000

    if "claude" in models and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    if "openai" in models and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    claude_client = None
    openai_client = None
    if "claude" in models:
        import anthropic
        claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    if "openai" in models:
        from openai import OpenAI
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    file_exists   = os.path.exists(output_file)
    existing_rows = []
    run_id        = 1

    if file_exists:
        with open(output_file, "r", encoding="utf-8", newline="") as f:
            existing_rows = normalize_existing_rows(list(csv.DictReader(f)))
        if existing_rows:
            run_id = max(int(r.get("id") or 0) for r in existing_rows) + 1
            if rerun_empty:
                before        = len(existing_rows)
                existing_rows = [r for r in existing_rows if not is_empty_or_no_output(r)]
                removed       = before - len(existing_rows)
                print(f"[INFO] Existing rows: {before}; rows to rerun: {removed}")
            else:
                print(f"[INFO] Existing rows: {len(existing_rows)}")
        # Rewrite with current schema
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(existing_rows)

    completed_keys = {row_key(r) for r in existing_rows}
    prompt_types   = ["standard", "comparative"]

    all_conditions = list(product(
        DISCIPLINES, domains, prompt_types, models.items(), range(1, replications + 1)
    ))

    pending = [
        (disc, dom, pt, pm, rep)
        for disc, dom, pt, pm, rep in all_conditions
        if (pm[0], pm[1], disc, dom, pt, str(rep), str(thinking)) not in completed_keys
    ]

    total = len(pending)
    print("\n" + "-" * 72)
    print(f"  Pending runs      : {total}")
    print(f"  Models            : {list(models.values())}")
    print(f"  Thinking          : {'on' if thinking else 'off'}")
    print(f"  OpenAI effort     : {openai_effort if thinking else 'off'}")
    print(f"  OpenAI max tokens : {openai_max_output_tokens}")
    print(f"  Output            : {output_file}")
    print("-" * 72 + "\n")

    if total == 0:
        print("[INFO] Nothing to run. All selected conditions are already present.")
        return

    with open(output_file, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        for done, (discipline, domain, prompt_type, (provider, model), rep) in enumerate(pending, 1):
            print(
                f"[{done:>3}/{total}] {provider:<8} | {discipline:<25} | "
                f"{domain:<22} | {prompt_type:<11} | rep {rep}"
            )

            prompt = generate_prompt(discipline, INSTITUTION, domain, prompt_type)
            t0     = time.time()

            try:
                if provider == "claude":
                    response_text, reasoning_text, meta = call_claude(
                        claude_client, model, prompt, thinking=thinking
                    )
                else:
                    response_text, reasoning_text, meta = call_openai(
                        openai_client, model, prompt,
                        thinking=thinking,
                        reasoning_effort=openai_effort,
                        max_output_tokens=openai_max_output_tokens,
                    )

                duration_ms = int((time.time() - t0) * 1000)

                row = {
                    "id":                    run_id,
                    "timestamp":             datetime.datetime.now().isoformat(),
                    "replication":           rep,
                    "thinking_mode":         thinking,
                    "model":                 model,
                    "provider":              provider,
                    "discipline":            discipline,
                    "institution":           INSTITUTION,
                    "domain":                domain,
                    "prompt_type":           prompt_type,
                    "duration_ms":           duration_ms,
                    "prompt":                prompt,
                    "reasoning":             reasoning_text,
                    "response":              response_text,
                    "api_status":            meta.get("api_status", ""),
                    "api_incomplete_reason": meta.get("api_incomplete_reason", ""),
                    "api_output_tokens":     meta.get("api_output_tokens", ""),
                    "api_reasoning_tokens":  meta.get("api_reasoning_tokens", ""),
                }

                writer.writerow(row)
                csvfile.flush()
                run_id += 1

                if response_text.startswith("["):
                    print(f"         [WARN] {response_text[:160]}")
                else:
                    print(
                        f"         [OK]   {duration_ms/1000:.1f}s | "
                        f"response {len(response_text)} chars | "
                        f"reasoning {len(reasoning_text)} chars | "
                        f"status={meta.get('api_status','')} | "
                        f"reasoning_tokens={meta.get('api_reasoning_tokens','')}"
                    )

            except KeyboardInterrupt:
                print("\n[WARN] Interrupted by user. Results saved up to this point.")
                return
            except Exception as exc:
                print(f"         [ERROR] {exc}")
                time.sleep(3)

    print(f"\n[OK] Finished. {total} runs saved to '{output_file}'")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Epistemic Audit - Study 1 runner")
    parser.add_argument("--output",                  default="audit_results.csv")
    parser.add_argument("--replications",            type=int, default=3)
    parser.add_argument("--only-claude",             action="store_true")
    parser.add_argument("--only-openai",             action="store_true")
    parser.add_argument("--domain",                  type=str, default=None,
                        choices=list(POLICY_DOMAINS.keys()))
    parser.add_argument("--no-thinking",             action="store_true")
    parser.add_argument("--rerun-empty",             action="store_true")
    parser.add_argument("--openai-effort",           default="medium",
                        choices=["none","minimal","low","medium","high","xhigh"])
    parser.add_argument("--openai-max-output-tokens",type=int, default=None)
    args = parser.parse_args()

    selected_models = {"claude": "claude-opus-4-6", "openai": "gpt-5.5"}
    if args.only_claude:
        selected_models = {"claude": "claude-opus-4-6"}
    elif args.only_openai:
        selected_models = {"openai": "gpt-5.5"}

    run_study(
        output_file              = args.output,
        models                   = selected_models,
        domains                  = [args.domain] if args.domain else None,
        replications             = args.replications,
        thinking                 = not args.no_thinking,
        rerun_empty              = args.rerun_empty,
        openai_effort            = args.openai_effort,
        openai_max_output_tokens = args.openai_max_output_tokens,
    )
