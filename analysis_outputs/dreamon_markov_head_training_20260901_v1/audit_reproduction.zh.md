# 2026-09-08 CPU 修正复现

原始记录与权重来自训练提交 `32e80041c6b62a9ce2147d431d5b62b5bcad1760`。本次只复核与重算，未执行 GPU 推理或训练。

固定原始输入：

- [OpenCoder 原始 parquet](https://huggingface.co/datasets/OpenCoder-LLM/opc-sft-stage2/resolve/7d28f40d579edd7c24402d17d0c7639f991e6f8d/educational_instruct/train-00000-of-00001.parquet)：118,278 行；成功完整解析后，106,940 条冻结清单全部能按 source_seq_id 找回，无重复源主键。
- [HumanEval 官方单行文件](https://github.com/openai/human-eval-infilling/blob/88062ff9859c875d04db115b698ed4b0f0395170/data/HumanEval-SingleLineInfilling.jsonl.gz)：1,033 行、164 基础题。

在仓库根目录执行（前两个参数替换成本地下载文件的实际路径）：

```bash
python -W ignore::SyntaxWarning analysis/audit_markov_training_data.py --source-parquet /path/to/educational.parquet --humaneval /path/to/HumanEval-SingleLineInfilling.jsonl.gz --result-dir analysis_outputs/dreamon_markov_head_training_20260901_v1
python analysis/repair_markov_training_reports.py --result-dir analysis_outputs/dreamon_markov_head_training_20260901_v1
python -m unittest tests.test_dreamon_markov_head_training tests.test_markov_training_audit_fixes
```

服务器剩余检查（仅查询 samples 元数据与已有诊断主键，不重做测试）：

```bash
python analysis/audit_markov_bank_membership.py --result-dir analysis_outputs/dreamon_markov_head_training_20260901_v1 --bank-db /home/shx/.cache/dllm_infilling/markov_heads/dreamon_markov_head_training_20260901_v1/transition_bank.sqlite
```

不得用原 GPU finalize 命令替代 CPU 报表重算，不得覆盖已有样本库/权重/冻结清单。新代码将拒绝在原路径重建数据或将旧损失版本的检查点当作新版本续训；这是防止已发现的具体覆盖问题，不是新增实验晋级条件。

复核环境：Python 3.12、CPU PyTorch 2.14.0；语法与数据检查使用 numpy、pyarrow、zstandard。原训练环境记录不变，不能据此宣称已在原 GPU 验证数值等价或速度。

已执行验证：16 项相关单元测试通过；修改脚本语法检查通过；12 行历史训练头汇总值逐字段保持一致；两份冻结清单和四份原验证/测试诊断与原提交逐字节相同。
