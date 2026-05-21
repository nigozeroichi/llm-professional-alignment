#!/usr/bin/env python3
"""
LLM Epistemic Audit — Study 1
Runner standalone com salvamento incremental linha a linha.

Instalação:
    pip install anthropic openai

Uso no PowerShell:
    # Defina as chaves de API como variáveis de ambiente:
    $env:ANTHROPIC_API_KEY="sk-ant-..."
    $env:OPENAI_API_KEY="sk-..."

    # Rode o script:
    python .\llm_audit_runner.py

    # Rodar apenas OpenAI:
    python .\llm_audit_runner.py --only-openai

    # Rodar apenas Claude:
    python .\llm_audit_runner.py --only-claude

    # Rodar apenas um domínio:
    python .\llm_audit_runner.py --domain "Gun Control"

    # Teste pequeno:
    python .\llm_audit_runner.py --only-openai --domain "Gun Control" --replications 1 --output teste_openai.csv

    # Re-rodar linhas vazias ou com NO_OUTPUT:
    python .\llm_audit_runner.py --rerun-empty

Observação importante:
    Em modelos GPT com reasoning, max_output_tokens inclui tanto os tokens de raciocínio
    quanto os tokens da resposta final visível. Se o orçamento for baixo, o modelo pode
    gastar tudo raciocinando e não produzir texto final.
"""

import os
import csv
import time
import datetime
import argparse
from itertools import product


# ─────────────────────────────────────────────
# Configuração experimental
# ─────────────────────────────────────────────

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
    "Economist": "economics",
    "Epidemiologist": "epidemiology",
    "Criminologist": "criminology",
    "Sociologist": "sociology",
    "Public Health Researcher": "public health",
    "Legal Scholar": "law",
    "Psychologist": "psychology",
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

    # Diagnóstico OpenAI
    "openai_response_id",
    "openai_status",
    "openai_incomplete_reason",
    "openai_input_tokens",
    "openai_output_tokens",
    "openai_total_tokens",
    "openai_reasoning_tokens",
    "openai_output_types",
    "diagnostic",
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_value(obj, key, default=None):
    """
    Acessa tanto atributos de objetos do SDK quanto chaves de dicionário.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def response_is_empty_or_no_output(row):
    """
    Decide se uma linha antiga deve ser re-rodada com --rerun-empty.
    Considera vazios reais e marcadores de falha operacional.
    """
    response = (row.get("response") or "").strip()

    if not response:
        return True

    no_output_markers = [
        "[NO_OUTPUT",
        "[NO_OUTPUT_MAX_TOKENS",
        "[NO_OUTPUT_UNKNOWN",
    ]

    return any(response.startswith(marker) for marker in no_output_markers)


def normalize_existing_row(row):
    """
    Garante que linhas antigas continuem compatíveis quando adicionamos colunas novas.
    """
    return {column: row.get(column, "") for column in CSV_COLUMNS}


# ─────────────────────────────────────────────
# Geração de prompt
# ─────────────────────────────────────────────

def generate_prompt(discipline, institution, domain, prompt_type):
    field = DISCIPLINE_FIELD.get(discipline, discipline.lower())
    info = POLICY_DOMAINS[domain]
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


# ─────────────────────────────────────────────
# Chamada Claude
# ─────────────────────────────────────────────

def call_claude(client, model, prompt, thinking=True):
    kwargs = {
        "model": model,
        "max_tokens": 16000 if thinking else 2000,
        "messages": [{"role": "user", "content": prompt}],
    }

    if thinking:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": 8000,
        }

    response = client.messages.create(**kwargs)

    text = ""
    reasoning = ""

    for block in response.content:
        if block.type == "thinking":
            reasoning = block.thinking or ""
        elif block.type == "text":
            text += block.text or ""

    meta = {
        "openai_response_id": "",
        "openai_status": "",
        "openai_incomplete_reason": "",
        "openai_input_tokens": "",
        "openai_output_tokens": "",
        "openai_total_tokens": "",
        "openai_reasoning_tokens": "",
        "openai_output_types": "",
        "diagnostic": "claude_ok" if text.strip() else "claude_empty_response",
    }

    return text.strip(), reasoning.strip(), meta


# ─────────────────────────────────────────────
# Chamada OpenAI
# ─────────────────────────────────────────────

def extract_openai_usage(response):
    usage = get_value(response, "usage")

    input_tokens = get_value(usage, "input_tokens", "")
    output_tokens = get_value(usage, "output_tokens", "")
    total_tokens = get_value(usage, "total_tokens", "")

    output_details = get_value(usage, "output_tokens_details")
    reasoning_tokens = get_value(output_details, "reasoning_tokens", "")

    return {
        "openai_input_tokens": input_tokens,
        "openai_output_tokens": output_tokens,
        "openai_total_tokens": total_tokens,
        "openai_reasoning_tokens": reasoning_tokens,
    }


def call_openai(
    client,
    model,
    prompt,
    thinking=True,
    openai_max_output_tokens=25000,
    openai_reasoning_effort="medium",
):
    kwargs = {
        "model": model,
        "input": [{"role": "user", "content": prompt}],
        "max_output_tokens": openai_max_output_tokens if thinking else 2000,
    }

    if thinking:
        kwargs["reasoning"] = {
            "effort": openai_reasoning_effort,
            "summary": "auto",
        }

    response = client.responses.create(**kwargs)

    text = ""
    reasoning = ""
    refusal = ""
    output_types = []

    for item in response.output:
        item_type = get_value(item, "type", "")
        output_types.append(item_type)

        if item_type == "reasoning":
            for summary_item in (get_value(item, "summary", []) or []):
                reasoning += get_value(summary_item, "text", "") + "\n"

        elif item_type == "message":
            for content_item in (get_value(item, "content", []) or []):
                content_type = get_value(content_item, "type", "")

                if content_type == "output_text":
                    text += get_value(content_item, "text", "")

                elif content_type == "refusal":
                    refusal += get_value(content_item, "refusal", "no detail")

        elif item_type == "refusal":
            refusal += get_value(item, "refusal", "no detail")

    # Fallback de conveniência do SDK
    if not text.strip():
        text = get_value(response, "output_text", "") or ""

    response_id = get_value(response, "id", "")
    status = get_value(response, "status", "")
    incomplete_details = get_value(response, "incomplete_details")
    incomplete_reason = get_value(incomplete_details, "reason", "")

    usage_meta = extract_openai_usage(response)
    reasoning_tokens = usage_meta.get("openai_reasoning_tokens", "")

    diagnostic = "openai_ok"

    if refusal.strip():
        text = f"[REFUSAL: {refusal.strip()}]"
        diagnostic = "openai_refusal"

    elif not text.strip() and status == "incomplete" and incomplete_reason == "max_output_tokens":
        text = (
            "[NO_OUTPUT_MAX_TOKENS: model reached max_output_tokens before producing "
            f"visible text; reasoning_tokens={reasoning_tokens}]"
        )
        diagnostic = "openai_no_output_max_tokens"

    elif not text.strip():
        text = (
            f"[NO_OUTPUT_UNKNOWN: status={status}; "
            f"incomplete_reason={incomplete_reason}; "
            f"reasoning_tokens={reasoning_tokens}]"
        )
        diagnostic = "openai_no_output_unknown"

    meta = {
        "openai_response_id": response_id,
        "openai_status": status,
        "openai_incomplete_reason": incomplete_reason,
        "openai_output_types": "|".join(output_types),
        "diagnostic": diagnostic,
    }

    meta.update(usage_meta)

    return text.strip(), reasoning.strip(), meta


# ─────────────────────────────────────────────
# Runner principal
# ─────────────────────────────────────────────

def run_study(
    output_file="audit_results.csv",
    models=None,
    domains=None,
    replications=3,
    thinking=True,
    rerun_empty=False,
    openai_max_output_tokens=25000,
    openai_reasoning_effort="medium",
):
    if models is None:
        models = {
            "claude": "claude-opus-4-6",
            "openai": "gpt-5.5",
        }

    if domains is None:
        domains = list(POLICY_DOMAINS.keys())

    prompt_types = ["standard", "comparative"]

    # Inicializa clientes apenas para os providers necessários
    claude_client = None
    openai_client = None

    if "claude" in models:
        import anthropic

        claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not claude_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY não encontrada nas variáveis de ambiente.")

        claude_client = anthropic.Anthropic(api_key=claude_api_key)

    if "openai" in models:
        from openai import OpenAI

        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY não encontrada nas variáveis de ambiente.")

        openai_client = OpenAI(api_key=openai_api_key)

    # Retoma de onde parou se o arquivo existir
    file_exists = os.path.exists(output_file)
    run_id = 1

    if file_exists:
        with open(output_file, "r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))

        if existing:
            run_id = max(int(r["id"]) for r in existing if str(r.get("id", "")).isdigit()) + 1

            if rerun_empty:
                empty_ids = {
                    int(r["id"])
                    for r in existing
                    if str(r.get("id", "")).isdigit() and response_is_empty_or_no_output(r)
                }

                print(
                    f"↩  {len(existing)} resultados existentes · "
                    f"{len(empty_ids)} vazios/NO_OUTPUT serão re-rodados"
                )

                keep = [
                    normalize_existing_row(r)
                    for r in existing
                    if int(r["id"]) not in empty_ids
                ]

                with open(output_file, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=CSV_COLUMNS,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                    writer.writerows(keep)

                file_exists = True

            else:
                print(f"↩  Retomando do run #{run_id} ({len(existing)} resultados existentes)")

    total = (
        len(DISCIPLINES)
        * len(domains)
        * len(prompt_types)
        * len(models)
        * replications
    )

    done = 0

    print(f"\n{'─' * 72}")
    print(f"  Total de runs: {total}")
    print(f"  Modelos: {list(models.values())}")
    print(f"  Thinking: {'on' if thinking else 'off'}")
    print(f"  OpenAI reasoning effort: {openai_reasoning_effort if thinking else 'off'}")
    print(f"  OpenAI max_output_tokens: {openai_max_output_tokens if thinking else 2000}")
    print(f"  Output: {output_file}")
    print(f"{'─' * 72}\n")

    with open(output_file, "a", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=CSV_COLUMNS,
            extrasaction="ignore",
        )

        if not file_exists:
            writer.writeheader()

        for discipline, domain, prompt_type, (provider, model), rep in product(
            DISCIPLINES,
            domains,
            prompt_types,
            models.items(),
            range(1, replications + 1),
        ):
            done += 1

            print(
                f"[{done:>3}/{total}] {provider:<8} · {discipline:<25} "
                f"· {domain:<22} · {prompt_type:<11} · rep {rep}"
            )

            prompt = generate_prompt(
                discipline=discipline,
                institution=INSTITUTION,
                domain=domain,
                prompt_type=prompt_type,
            )

            t0 = time.time()

            try:
                if provider == "claude":
                    response, reasoning, meta = call_claude(
                        client=claude_client,
                        model=model,
                        prompt=prompt,
                        thinking=thinking,
                    )

                elif provider == "openai":
                    response, reasoning, meta = call_openai(
                        client=openai_client,
                        model=model,
                        prompt=prompt,
                        thinking=thinking,
                        openai_max_output_tokens=openai_max_output_tokens,
                        openai_reasoning_effort=openai_reasoning_effort,
                    )

                else:
                    raise ValueError(f"Provider desconhecido: {provider}")

                duration_ms = int((time.time() - t0) * 1000)

                row = {
                    "id": run_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "replication": rep,
                    "thinking_mode": thinking,
                    "model": model,
                    "provider": provider,
                    "discipline": discipline,
                    "institution": INSTITUTION,
                    "domain": domain,
                    "prompt_type": prompt_type,
                    "duration_ms": duration_ms,
                    "prompt": prompt,
                    "reasoning": reasoning,
                    "response": response,
                }

                row.update(meta)

                writer.writerow(row)
                csvfile.flush()

                run_id += 1

                resp_chars = len(response)
                reas_chars = len(reasoning)

                if response.startswith("[REFUSAL"):
                    print(f"         ⚠  REFUSAL · {response[:120]}")

                elif response.startswith("[NO_OUTPUT"):
                    print(f"         ⚠  {response[:160]}")

                elif not response.strip():
                    print("         ⚠  RESPOSTA VAZIA — verifique a API antes de continuar")

                else:
                    extra = ""
                    if provider == "openai":
                        extra = (
                            f" | status {meta.get('openai_status')} "
                            f"| reasoning_tokens {meta.get('openai_reasoning_tokens')}"
                        )

                    print(
                        f"         ✓  {duration_ms / 1000:.1f}s "
                        f"| resposta {resp_chars} chars "
                        f"| raciocínio {reas_chars} chars"
                        f"{extra}"
                    )

            except KeyboardInterrupt:
                print("\n\n⚠  Interrompido pelo usuário. Resultados salvos até aqui.")
                return

            except Exception as e:
                print(f"         ✗  Erro: {e}")
                time.sleep(3)

    print(f"\n✓  Concluído. {done} runs salvas em '{output_file}'")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Epistemic Audit — Study 1 runner"
    )

    parser.add_argument(
        "--output",
        default="audit_results.csv",
        help="Arquivo CSV de saída. Padrão: audit_results.csv",
    )

    parser.add_argument(
        "--replications",
        type=int,
        default=3,
        help="Réplicas por condição. Padrão: 3",
    )

    parser.add_argument(
        "--only-claude",
        action="store_true",
        help="Rodar apenas Claude.",
    )

    parser.add_argument(
        "--only-openai",
        action="store_true",
        help="Rodar apenas OpenAI.",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        choices=list(POLICY_DOMAINS.keys()),
        help="Rodar apenas um domínio de política.",
    )

    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Desativar extended thinking/reasoning.",
    )

    parser.add_argument(
        "--rerun-empty",
        action="store_true",
        help="Re-rodar runs que vieram vazios ou com marcador NO_OUTPUT.",
    )

    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=25000,
        help=(
            "Budget de max_output_tokens para OpenAI quando thinking está ligado. "
            "Padrão: 25000."
        ),
    )

    parser.add_argument(
        "--openai-reasoning-effort",
        type=str,
        default="medium",
        choices=["minimal", "low", "medium", "high"],
        help="Nível de reasoning effort para OpenAI. Padrão: medium.",
    )

    args = parser.parse_args()

    models = {
        "claude": "claude-opus-4-6",
        "openai": "gpt-5.5",
    }

    if args.only_claude:
        models = {"claude": "claude-opus-4-6"}

    elif args.only_openai:
        models = {"openai": "gpt-5.5"}

    domains = [args.domain] if args.domain else None

    run_study(
        output_file=args.output,
        models=models,
        domains=domains,
        replications=args.replications,
        thinking=not args.no_thinking,
        rerun_empty=args.rerun_empty,
        openai_max_output_tokens=args.openai_max_output_tokens,
        openai_reasoning_effort=args.openai_reasoning_effort,
    )