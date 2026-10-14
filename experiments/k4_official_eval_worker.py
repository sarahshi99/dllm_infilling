"""Run unchanged enabled pinned evaluator inside an isolated filesystem/network."""
import json
from pathlib import Path
import sys

source=Path('/input/execution.py').read_text()
namespace={'__name__':'pinned_evaluator_enabled','__file__':'/input/execution.py'}
exec(compile(source,'/input/execution.py','exec'),namespace)
payload=json.loads(Path('/input/payload.json').read_text())
result=namespace['check_correctness'](payload['problem'],payload['output'],3.0,0)
print(json.dumps({'official_result':result,'interpreter':sys.version},ensure_ascii=False))
