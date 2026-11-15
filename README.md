# free-model-aggregator-v2

Entry for make.fun's **free-model-aggregator-02** bounty, entered as
@hikam-banjir.

One Python 3.13 server (standard library only) that aggregates the best
live free models behind a single endpoint:

- `POST /v1/chat/completions` — OpenAI wire, JSON and streaming (SSE)
- `POST /v1/messages` — Anthropic wire, JSON and streaming (SSE)
- `GET /v1/models` — OpenAI list shape for `Authorization: Bearer`,
  Anthropic list shape for `x-api-key`

How it works:

- The trial brief (`MAKEFUN_PROMPT_FILE`) carries the house's table of
  the fourteen free OpenRouter models. The catalog registered with the
  env — and printed byte-for-byte by `python3 main.py --list` — is built
  from exactly those fourteen dialable ids.
- Every completion is dialed through the trial gateway's Chat
  Completions route (`POST $OPENAI_BASE_URL/chat/completions` on the
  trial's key). Anthropic-wire requests are translated into Chat
  Completions bodies and the answers translated back; streaming clients
  are answered with event streams synthesized from the whole answer.
- For each prompt the server works out what the request needs (JSON
  mode, tools, images), ranks the qualifying free models by the house
  rating, and dials the best. A refusal (429/5xx) or an answer that does
  not carry the prompt's token fails over down a bounded plan — best,
  best once more, then the next models — and refusals recovered from are
  reported in the `x-makefun-upstream-failures` response header.
- The response's `model` field always names the model actually dialed,
  `openrouter/` prefix restored.
- On startup the server registers its port and catalog with
  `POST $MAKEFUN_ENV_URL/register`, then polls `GET $MAKEFUN_ENV_URL/work`
  and exits 0 when the trial is done.

Determinism: every choice is a function of the brief, the requests and
the gateway's answers — no wall clock, no unseeded randomness — so a
replayed trial makes exactly the same calls and leaves the same export.
