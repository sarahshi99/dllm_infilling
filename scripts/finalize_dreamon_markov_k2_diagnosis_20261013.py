#!/usr/bin/env python3
"""Publish only the authorized diagnosis summary after the background job exits."""
import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1"
if subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip() != "codex/dreamon-markov-k2-diagnosis-v1":
    raise RuntimeError("finalization branch changed; leave results local")
status = int(sys.argv[1])
manifest = json.loads((OUT / "run_manifest.json").read_text())
audit = json.loads((OUT / "integrity_audit.json").read_text())
completed = status == 0 and audit["status"] == "passed"
when = datetime.datetime.now(datetime.timezone.utc).isoformat()
heading = "完成并停止" if completed else "停止并保存阻塞"
section = (
    f"## {when} K2 诊断后台任务：{heading}\n\n"
    f"- 用户后续授权共享 H200；十配置串行、按行轮换，GPU 背景负载另存。共享负载可能影响纯生成耗时，不能称空卡测量。\n"
    f"- 状态 `{manifest['status']}`，退出码 `{status}`；成功行 `{audit['row_count']}/10330`，"
    f"smoke `{audit['smoke_status']}`，完整性 `{audit['status']}`，GPU 作业累计 `{manifest.get('gpu_session_sec', 0) / 3600:.4f}h`（上限12h，排队等待不计）。\n"
    f"- 代码 `{manifest['code_commit']}`；报告 `analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/report.zh.md`，全部配对结论见报告。\n"
    f"- {'本轮全部完成；没有后续 GPU 作业。方法建议只写入报告，不自动运行。' if completed else '阻塞：' + manifest.get('blocker', '详见日志和完整性审计') + '。保留逐行数据库和失败日志，未自动改范围或参数。'}\n\n"
)
paths = []
for name in ("pause_checkpoint.current.md", "paper_agent_dashboard.zh.md", "experiment_queue.md", "current_action.md"):
    path = REPO / "docs/paper_agent" / name
    first, rest = path.read_text().split("\n", 1)
    path.write_text(first + "\n\n" + section + rest.lstrip("\n"))
    paths.append(str(path.relative_to(REPO)))
ledger = REPO / "docs/paper_agent/activity_ledger.zh.md"
ledger.write_text(ledger.read_text().rstrip() + "\n\n" + section.rstrip() + "\n")
paths.append(str(ledger.relative_to(REPO)))
for name in ("run_manifest.json", "integrity_audit.json", "report.zh.md", "summary.csv", "paired_comparisons.csv", "smoke_checks.json"):
    path = OUT / name
    if path.exists():
        paths.append(str(path.relative_to(REPO)))
(OUT / "background_exit.json").write_text(json.dumps({"finished_utc": when, "exit_code": status,
    "completed": completed, "status": manifest["status"], "row_count": audit["row_count"]}, indent=2) + "\n")
paths.append(str((OUT / "background_exit.json").relative_to(REPO)))
print("Finalization staged paths:\n" + "\n".join(paths), flush=True)
subprocess.run(["git", "diff", "--check"], cwd=REPO, check=True)
subprocess.run(["git", "add", "--", *paths], cwd=REPO, check=True)
if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode:
    subprocess.run(["git", "commit", "--only", "-m", "results: finalize shared H200 Markov K2 diagnosis", "--", *paths], cwd=REPO, check=True)
subprocess.run(["git", "push", "origin", "codex/dreamon-markov-k2-diagnosis-v1"], cwd=REPO, check=True)
print("Background diagnosis finalized; no further experiments scheduled.", flush=True)
