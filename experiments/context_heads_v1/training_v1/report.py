"""Verify completed pilot checkpoints, export small evidence, publish separate ready."""
import argparse
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..contract import CONTRACT_ID, HeadConfig
from ..heads import ContextHead, MODEL_NAMES
from ..integration_v1.verify_cpu import check_original_payload
from .data import atomic_json, sha
from .run import DEFAULT_RUN, ROOT

MODULE = Path(__file__).resolve().parent
BRANCH = "codex/dreamon-context-heads-20261015-v1"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_report(run):
    summary = json.loads((run/"summary.json").read_text())
    end = json.loads((run/"exit.json").read_text())
    data = json.loads((run/"data/manifest.json").read_text())
    if summary["status"] != "complete" or end["exit_code"] != 0:
        raise RuntimeError("pilot did not finish successfully")
    if [r["model"] for r in summary["results"]] != list(MODEL_NAMES):
        raise ValueError("seven-model queue incomplete")
    torch.set_num_threads(4)
    inventory = []
    for result in summary["results"]:
        if result["optimizer_steps"] != math.ceil(data["splits"]["train"]/summary["plan"]["batch_size"]) or result["examples_seen"] != data["splits"]["train"]:
            raise ValueError("unequal training budget")
        for kind in ("last", "best"):
            path = run/"checkpoints"/result["model"]/(kind+".pt")
            state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            if state["model"] != result["model"] or state["plan"] != summary["plan"]:
                raise ValueError("checkpoint model/plan mismatch")
            with torch.device("meta"):
                meta = ContextHead(result["model"], HeadConfig(**state["config"]))
            expected = meta.state_dict()
            if state["head"].keys() != expected.keys():
                raise ValueError("checkpoint parameter keys mismatch")
            for name, tensor in state["head"].items():
                if tensor.shape != expected[name].shape or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                    raise ValueError("nonfinite/incorrect checkpoint tensor")
            if kind == "last":
                if state["steps"] != result["optimizer_steps"] or state["cursor"] != data["splits"]["train"]:
                    raise ValueError("last checkpoint cursor mismatch")
                for values in state["optimizer"]["state"].values():
                    if int(values["step"]) != state["steps"]:
                        raise ValueError("Adam step mismatch")
                    for key in ("exp_avg", "exp_avg_sq"):
                        if not bool(torch.isfinite(values[key]).all()):
                            raise ValueError("nonfinite Adam checkpoint")
                step = state["steps"]
            else:
                step = state["step"]
                if step != result["best_step"]:
                    raise ValueError("best checkpoint selection mismatch")
            inventory.append(dict(model=result["model"], kind=kind, path=str(path.relative_to(run)),
                                  size_bytes=path.stat().st_size, sha256=sha(path),
                                  step=step, trained=step > 0, tensors_finite=True))
            del state, meta
    # Portable summary uses relative checkpoint paths; raw runtime remains external.
    for result in summary["results"]:
        for key in ("checkpoint_best", "checkpoint_last"):
            result[key] = str(Path(result[key]).relative_to(run))
    out = MODULE/"artifacts"
    atomic_json(out/"training_summary.json", summary)
    atomic_json(out/"checkpoint_inventory.json", dict(checkpoints=inventory,
                verification="CPU mmap, exact shapes/FP32/finite head tensors and Adam moments; plan/cursors checked",
                storage="external run cache; weights and optimizer states are not in git",
                reason_for_hashes="Git tracks code, not these external best/last files. Same-shaped checkpoints at a reused path cannot identify exact trained weights without a payload digest."))
    atomic_json(out/"completion.json", dict(status="complete", models=7,
                optimizer_steps=sum(r["optimizer_steps"] for r in summary["results"]),
                train_rows=data["splits"]["train"], validation_rows=data["splits"]["validation"],
                source_groups=data["source_groups"], group_overlap=0,
                training_source_commit=summary["source_commit"], exit_code=0,
                worker_wall_seconds=summary["worker_wall_seconds"],
                gpu_worker_exited=True, original_B_payload=check_original_payload(),
                checkpoint_files_verified=len(inventory), human_eval_answers_generated=False,
                backbone_loaded=False, teacher_collection_started=False))
    stale = summary["stale_validation"]["by_bucket"]["all"]
    lines = ["# B 首次七头统一 KL 训练结果（2026-10-28）", "",
             "**完成**：七头各1 epoch、144步，总1008步；train2304/validation768，源题组690/233，无组交集。",
             f"训练代码提交 `{summary['source_commit']}`；单worker共用GPU0 H200，FP32。worker墙时{summary['worker_wall_seconds']:.2f}秒，包括验证和写盘，不是推理测速。",
             "没有加载/重跑DreamOn、采新teacher或生成HumanEval答案；其他GPU进程未中断。", "",
             f"共同旧 logits：validation T1 KL={stale['kl_T1']:.6f}，TV={stale['tv_T1']:.6f}。以下全部为768状态上的相同离线教师目标，非代码通过率。", "",
             "|模型|初始KL|训练后KL|训练后TV|源题组等权KL|best step|peak allocated MiB|",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["results"]:
        initial, final = r["initial_validation"]["by_bucket"]["all"], r["final_validation"]["by_bucket"]["all"]
        lines.append(f"|{r['model']}|{initial['kl_T1']:.6f}|{final['kl_T1']:.6f}|{final['tv_T1']:.6f}|{r['final_validation']['group_mean_kl_T1']:.6f}|{r['best_step']}|{r['peak_gpu_allocated_bytes']/2**20:.0f}|")
    lines += ["", "|模型|历史1词KL|历史2词KL|历史3词KL|", "|---|---:|---:|---:|"]
    for r in summary["results"]:
        values = r["final_validation"]["by_bucket"]
        lines.append("|"+r["model"]+"|"+"|".join(f"{values[f'history_length={i}']['kl_T1']:.6f}" for i in (1, 2, 3))+"|")
    lines += ["", "H2 的本次末点KL最低；Hctx也已有明显改善，不能把上下文候选的全部收益归因于新token/多词记忆。",
              "H4 末点KL比初始更差，虽TV下降，仍按预先规定的KL选best_step=0；best.pt是未训练初始权重，真正训练后的权重在last.pt。负结果完整保留，没有加轮次、改学习率或重抽种子救结果。",
              "单seed、单epoch、单超参数pilot，未作显著性结论；H4退化不证明架构无效，H2名次也不证明更长记忆必要。历史1词相当K2，不能验证多词记忆。",
              "其他order/span/来源分桶、初始/最终逐桶KL/TV和源题组等权指标完整保存在artifacts/training_summary.json。",
              "T=.2/top-p=.9实际生成质量、完整结构候选调度及公平端到端速度尚未评测；没有在本批自动启动它们。", "",
              "14个best/last checkpoint已核验模型/计划/步数、FP32形状与有限参数/Adam状态，清单与hash见artifacts/checkpoint_inventory.json。",
              "大权重、optimizer、memmap和train.log只保留在外部cache。checkpoint路径相对此次run根目录：/home/shx/.cache/dllm_infilling/context_heads_training_20261028_v1。",
              "训练/恢复命令见README.zh.md；重新导出小型报告：`CUDA_VISIBLE_DEVICES='' /home/shx/miniconda3/envs/dllm_env/bin/python -m experiments.context_heads_v1.training_v1.report`。",
              "旧B与接入补验payload/ready保留；本次新训练结果单独manifest和B_training_done发布。完成后停止，不自动训练第二轮。"]
    (MODULE/"report.zh.md").write_text("\n".join(lines)+"\n")
    paths = sorted(p for p in MODULE.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.name != "manifest.json")
    atomic_json(MODULE/"manifest.json", dict(contract_id=CONTRACT_ID, schema_version=1,
                scope="completed one-epoch seven-head cached KL pilot", training_source_commit=summary["source_commit"],
                data=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths]))
    print("7 models, 1008 optimizer steps, 14 verified checkpoints; report exported")


def publish(run):
    if git("branch", "--show-current") != BRANCH or git("status", "--porcelain", "--", "experiments/context_heads_v1/training_v1", "docs/paper_agent/activity_ledger.zh.md"):
        raise RuntimeError("commit B training delivery before publication")
    commit = git("rev-parse", "HEAD")
    remote = subprocess.check_output(["git", "-c", "core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=12 -o ConnectionAttempts=1 -o ServerAliveInterval=5 -o ServerAliveCountMax=2", "ls-remote", "origin", "refs/heads/"+BRANCH], cwd=ROOT, text=True, timeout=35).split()
    if not remote or remote[0] != commit:
        raise RuntimeError("remote HEAD differs from local training delivery")
    check_original_payload()
    manifest = MODULE/"manifest.json"
    value = json.loads(manifest.read_text())
    for entry in value["data"]:
        if sha(ROOT/entry["path"]) != entry["sha256"]:
            raise RuntimeError("training delivery payload changed")
    completion = json.loads((MODULE/"artifacts/completion.json").read_text())
    if completion["status"] != "complete":
        raise RuntimeError("training not complete")
    common = Path(git("rev-parse", "--path-format=absolute", "--git-common-dir"))/"research_runs/dreamon_context_20261015_v1"
    ready = dict(contract_id=CONTRACT_ID, schema_version=1, status="B_training_pilot_complete",
                 commit=commit, branch=BRANCH, source_commit=commit, producer_branch=BRANCH,
                 training_source_commit=completion["training_source_commit"],
                 manifest_path=str(manifest), manifest_absolute_path=str(manifest), manifest_sha256=sha(manifest),
                 checkpoint_root_absolute_path=str(run), models=7, epochs=1, optimizer_steps=1008,
                 metrics_scope="offline validation T1 teacher KL/TV, no answer generation", H4_best_is_untrained=True,
                 created_at=datetime.now(timezone.utc).isoformat())
    target = common/"B_training_done.json"
    if target.exists():
        if json.loads(target.read_text()).get("commit") == commit:
            print(target)
            return
        raise FileExistsError("preserving existing training ready")
    temporary = common/(".B_training_done."+str(os.getpid())+".tmp")
    atomic_json(temporary, ready)
    os.link(temporary, target)
    temporary.unlink()
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    publish(args.run_dir) if args.publish else build_report(args.run_dir)
