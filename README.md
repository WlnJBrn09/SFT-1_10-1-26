# SFT-T

Private supervised-fine-tuning dataset built from this machine's coding-agent sessions (Claude Code, Codex, Grok Build, Cursor, OpenCode).

Raw archives live in [Formatron](https://github.com/WlnJBrn09/Formatron). This repo is the **train-ready JSONL** rewrite.

Conversations were compacted from native session logs, then formatted by **Xiaomi MiMo-V2.6-Pro** (`xiaomi/mimo-v2.6-pro` on OpenRouter) into OpenAI-style chat examples.

## Counts

- **595** examples (`data/all.jsonl`)
- **559** train / **36** valid (hash split, ~94/6)
- Median example ~4k characters; none over 48k
- 443 chunks formatted by MiMo; 125 used a deterministic fallback when the model output was not parseable JSONL

## Layout

| Path | Contents |
| --- | --- |
| `data/train.jsonl` | Training split (~95%) |
| `data/valid.jsonl` | Held-out split (~5%) |
| `data/all.jsonl` | Full set, same schema |
| `data/stats.json` | Counts, token-ish length buckets, source mix |

## Schema

One JSON object per line, UTF-8, no BOM:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

This is the format used by OpenAI fine-tuning, Axolotl (`chat_template`), Unsloth, Llama-Factory, and TRL `SFTTrainer` with a chat template.

Rules enforced:

- `messages` is a list of `{role, content}` objects
- `role` is only `system`, `user`, or `assistant`
- every example has at least one `user` and ends on `assistant`
- `content` is a non-empty string
- obvious secrets (API keys, tokens, passwords) are replaced with `<REDACTED>`

## Train with it

**Axolotl** (YAML snippet):

```yaml
datasets:
  - path: data/train.jsonl
    type: chat_template
    field_messages: messages
    message_property_mappings:
      role: role
      content: content
val_set_size: 0
```

**OpenAI-compatible FT:**

```bash
openai api fine_tuning.jobs.create -m gpt-4.1-mini -f data/train.jsonl
```

**TRL / Hugging Face:**

```python
from datasets import load_dataset
ds = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/valid.jsonl"})
# tokenizer.apply_chat_template(ex["messages"], tokenize=False)
```

Packing, max length, and chat template are trainer-side. 8k–32k context is a reasonable starting cutoff; drop or chunk examples that exceed your model's limit.

## Provenance

| Source | Local store |
| --- | --- |
| Claude Code | `~/.claude/projects/` |
| Codex CLI | `~/.codex/sessions/` |
| Grok Build | `~/.grok/sessions/` |
| Cursor | `~/.cursor/projects/*/agent-transcripts/` |
| OpenCode | `~/.local/share/opencode/opencode.db` |

Formatter: MiMo-V2.6-Pro. Failed or unparseable model outputs fall back to a deterministic user/assistant collapse so the split stays complete.

Lock files, WAL/SHM sidecars, and credential stores were never copied in.
