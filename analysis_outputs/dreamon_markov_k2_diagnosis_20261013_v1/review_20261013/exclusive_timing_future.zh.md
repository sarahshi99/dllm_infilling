# 仅供以后执行的独占计时条件（本轮不执行）

须另行授权GPU运行；使用获准的同一H200、同精度/batch/模型/头/种子和完整1033行人口。开始及全程不应有其他计算任务；不通过kill清场。只等待资源自然可用，并记录功耗/温度/时钟和背景快照。旧共享速度不可充当独占测量。

原服务器有全部版本固定缓存时，可以先从已有manifest提取原配置（这一步本轮也未执行）：

```bash
python - <<'PYCODE'
import json
from pathlib import Path
manifest = json.loads(Path('analysis_outputs/dreamon_markov_k2_diagnosis_20261013_v1/run_manifest.json').read_text())
Path('/tmp/k2_original_config.json').write_text(json.dumps(manifest['original_config']))
PYCODE
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python experiments/run_dreamon_markov_k2_diagnosis.py \
  --original-config /tmp/k2_original_config.json \
  --output-dir analysis_outputs/dreamon_markov_k2_exclusive_future \
  --budget-gpu-hours 12
```

不传`--allow-shared-gpu`，现有runner会拒绝其他计算进程共存。输出必须是新目录，不能覆盖本轮结果。原配置中的作者机器模型/数据路径需在将来获准的执行机器按同一版本提供；这与CPU分析已经实现的相对路径可携带证据是不同用途。本轮未排队、未运行此命令。
