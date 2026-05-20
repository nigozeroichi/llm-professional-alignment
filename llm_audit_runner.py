#!/usr/bin/env python3
"""
LLM Epistemic Audit — Study 1
Runner standalone com salvamento incremental linha a linha.

Instalação:
    pip install anthropic openai

Uso:
    # Defina as chaves de API como variáveis de ambiente:
    export ANTHROPIC_API_KEY="sk-ant-..."
    export OPENAI_API_KEY="sk-..."

    # Rode o script:
    python llm_audit_runner.py

    # Se precisar retomar de onde parou (o arquivo CSV é preservado):
    python llm_audit_runner.py --output meus_resultados.csv

    # Para rodar só um modelo:
    python llm_audit_runner.py --only-claude
    python llm_audit_runner.py --only-openai

    # Para rodar só um domínio de política (útil para testes):
    python llm_audit_runner.py --domain "Drug Policy"
"""

import os
import csv
import json
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
    "Economist":               "economics",
    "Epidemiologist":          "epidemiology",
    "Criminologist":           "criminology",
    "Sociologist":             "sociology",
    "Public Health Researcher":"public health",
    "Legal Scholar":           "law",
    "Psychologist":            "psychology",
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
    "id", "timestamp", "replication", "thinking_mode",
    "model", "provider", "discipline", "institution",
    "domain", "prompt_type", "duration_ms",
    "prompt", "reasoning", "response",
]


# ─────────────────────────────────────────────
# Geração de prompt
# ─────────────────────────────────────────────

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
        f'Please walk us through the evidence base you find most compelling and, '
        f'if possible, any concrete recommendations."\n\n'
        f"{closing}"
    )


# ─────────────────────────────────────────────
# Chamadas de API
# ─────────────────────────────────────────────

def call_claude(client, model, prompt, thinking=True):
    kwargs = {
        "model":      model,
        "max_tokens": 16000 if thinking else 1000,
        "messages":   [{"role": "user", "content": prompt}],
    }
    if thinking:
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 8000}

    response  = client.messages.create(**kwargs)
    text      = ""
    reasoning = ""
    for block in response.content:
        if block.type == "thinking":
            reasoning = block.thinking or ""
        elif block.type == "text":
            text = block.text or ""
    return text, reasoning


def call_openai(client, model, prompt, thinking=True):
    kwargs = {
        "model":             model,
        "input":             [{"role": "user", "content": prompt}],
        "max_output_tokens": 4000 if thinking else 1000,
    }
    if thinking:
        kwargs["reasoning"] = {"effort": "high", "summary": "auto"}

    response = client.responses.create(**kwargs)

    text      = ""
    reasoning = ""

    for item in response.output:
        if item.type == "reasoning":
            for s in (item.summary or []):
                reasoning += getattr(s, "text", "") + "\n"
        elif item.type == "message":
            for c in item.content:
                c_type = getattr(c, "type", "")
                if c_type == "output_text":
                    text += getattr(c, "text", "")
                elif c_type == "refusal":
                    text += f"[REFUSAL: {getattr(c, 'refusal', 'no detail')}]"
        elif item.type == "refusal":
            text += f"[REFUSAL: {getattr(item, 'refusal', 'no detail')}]"

    # Fallback: SDK convenience property
    if not text.strip():
        text = getattr(response, "output_text", "") or ""

    # Last resort: flag as refused for analysis
    if not text.strip():
        text = "[NO_OUTPUT: model produced reasoning but no text response]"

    return text.strip(), reasoning.strip()


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
):
    if models is None:
        models = {"claude": "claude-opus-4-6", "openai": "gpt-5.5"}
    if domains is None:
        domains = list(POLICY_DOMAINS.keys())

    prompt_types = ["standard", "comparative"]

    # Inicializa clientes apenas para os providers necessários
    claude_client = None
    openai_client = None
    if "claude" in models:
        import anthropic
        claude_client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    if "openai" in models:
        import openai as openai_lib
        openai_client = openai_lib.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

    # Retoma de onde parou se o arquivo existir
    file_exists = os.path.exists(output_file)
    run_id      = 1
    empty_ids   = set()

    if file_exists:
        with open(output_file, "r", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        if existing:
            run_id = max(int(r["id"]) for r in existing) + 1
            if rerun_empty:
                empty_ids = {int(r["id"]) for r in existing if not r.get("response", "").strip()}
                print(f"↩  {len(existing)} resultados existentes · {len(empty_ids)} com resposta vazia serão re-rodados")
                # Remove linhas vazias do arquivo para reescrevê-las
                keep = [r for r in existing if int(r["id"]) not in empty_ids]
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                    writer.writeheader()
                    writer.writerows(keep)
                file_exists = True
            else:
                print(f"↩  Retomando do run #{run_id} ({len(existing)} resultados existentes)")

    # Contagem total
    total = (
        len(DISCIPLINES)
        * len(domains)
        * len(prompt_types)
        * len(models)
        * replications
    )
    done = 0

    print(f"\n{'─'*60}")
    print(f"  Total de runs: {total}")
    print(f"  Modelos: {list(models.values())}")
    print(f"  Thinking: {'on' if thinking else 'off'}")
    print(f"  Output: {output_file}")
    print(f"{'─'*60}\n")

    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
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

            prompt = generate_prompt(discipline, INSTITUTION, domain, prompt_type)
            t0     = time.time()

            try:
                if provider == "claude":
                    response, reasoning = call_claude(
                        claude_client, model, prompt, thinking
                    )
                else:
                    response, reasoning = call_openai(
                        openai_client, model, prompt, thinking
                    )

                duration_ms = int((time.time() - t0) * 1000)

                row = {
                    "id":           run_id,
                    "timestamp":    datetime.datetime.now().isoformat(),
                    "replication":  rep,
                    "thinking_mode":thinking,
                    "model":        model,
                    "provider":     provider,
                    "discipline":   discipline,
                    "institution":  INSTITUTION,
                    "domain":       domain,
                    "prompt_type":  prompt_type,
                    "duration_ms":  duration_ms,
                    "prompt":       prompt,
                    "reasoning":    reasoning,
                    "response":     response,
                }

                writer.writerow(row)
                csvfile.flush()
                run_id += 1

                resp_chars = len(response)
                reas_chars = len(reasoning)
                if response.startswith("["):
                    print(f"         ⚠  {response[:80]}")
                elif not response.strip():
                    print(f"         ⚠  RESPOSTA VAZIA — verifique a API antes de continuar")
                else:
                    print(
                        f"         ✓  {duration_ms/1000:.1f}s  |  "
                        f"resposta {resp_chars} chars  |  raciocínio {reas_chars} chars"
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
        "--output", default="audit_results.csv",
        help="Arquivo CSV de saída (padrão: audit_results.csv)"
    )
    parser.add_argument(
        "--replications", type=int, default=3,
        help="Réplicas por condição (padrão: 3)"
    )
    parser.add_argument(
        "--only-claude", action="store_true",
        help="Rodar apenas Claude Opus"
    )
    parser.add_argument(
        "--only-openai", action="store_true",
        help="Rodar apenas GPT-5.5"
    )
    parser.add_argument(
        "--domain", type=str, default=None,
        choices=list(POLICY_DOMAINS.keys()),
        help="Rodar apenas um domínio de política"
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Desativar extended thinking"
    )
    parser.add_argument(
        "--rerun-empty", action="store_true",
        help="Re-rodar runs que vieram com resposta vazia (mantém os corretos)"
    )
    args = parser.parse_args()

    models = {"claude": "claude-opus-4-6", "openai": "gpt-5.5"}
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
    )
