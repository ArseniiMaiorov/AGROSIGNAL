#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'backend'
FRONTEND = ROOT / 'frontend'


def check(name: str, ok: bool, detail: str) -> dict[str, str | bool]:
    return {'name': name, 'ok': ok, 'detail': detail}


def main() -> int:
    results = []
    results.append(check('backend_main', (BACKEND / 'main.py').exists(), str(BACKEND / 'main.py')))
    results.append(check('frontend_app', (FRONTEND / 'src/App.vue').exists(), str(FRONTEND / 'src/App.vue')))
    results.append(check('docker_compose', (ROOT / 'docker-compose.yml').exists(), str(ROOT / 'docker-compose.yml')))
    results.append(check('classifier_model', (BACKEND / 'models/object_classifier.pkl').exists(), str(BACKEND / 'models/object_classifier.pkl')))
    results.append(check('boundary_model', (BACKEND / 'models/boundary_unet_v2.onnx').exists(), str(BACKEND / 'models/boundary_unet_v2.onnx')))
    results.append(check('env_example', (ROOT / '.env.example').exists(), str(ROOT / '.env.example')))

    try:
        subprocess.run(['docker', 'compose', 'config'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        results.append(check('docker_compose_config', True, 'docker compose config'))
    except Exception as exc:
        results.append(check('docker_compose_config', False, str(exc)))

    failed = [item for item in results if not item['ok']]
    print(json.dumps({'ok': not failed, 'results': results}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
