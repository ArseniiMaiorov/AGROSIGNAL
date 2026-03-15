# Release Verification Matrix

Run from project root after `docker compose up -d --build`.

## Automated

Backend regression:
```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/pytest backend/tests -q
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/pytest backend/tests --cov=backend --cov-report=term-missing:skip-covered
```

Frontend regression:
```bash
npm --prefix frontend run build
npm --prefix frontend run test:unit
npm --prefix frontend run test:coverage
npm --prefix frontend run test:e2e
```

Release API smoke:
```bash
./.venv/bin/python scripts/release_smoke.py
```

Russia QA matrix:
```bash
./.venv/bin/python backend/training/scripts/validate_release_qa_matrix.py backend/training/release_russia_qa_matrix.json
./.venv/bin/python backend/training/scripts/run_release_qa_matrix.py --limit 2
```

## Manual release checks

- Startup: `bootstrap -> login -> logout -> refresh token`
- Detection: `submit -> stage-aware progress -> result -> stale recovery`
- Field ops: `manual create -> merge -> split`
- Analytics: `dashboard -> prediction -> scenario -> archive`
- UX: `weather freshness -> status freshness -> wind streak overlay`
- Recovery: `refresh page during polling -> repeat submit guard -> retry after failed run`

## Release blockers

- Any raw `500` in an expected business flow
- Any silent hang without heartbeat/status progression
- Any write flow that cannot recover after refresh/restart
- Any prediction/scenario result that hides low-confidence applicability
