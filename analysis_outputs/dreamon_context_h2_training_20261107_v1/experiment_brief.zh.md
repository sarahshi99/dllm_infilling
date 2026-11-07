# H2 正式训练行动说明（2026-11-08 北京时间）

用户已授权必要实现、正式训练、固定评测、后台自动接续、提交推送；只在当前会话串行完成准备。
reviewer_gate_disabled；采用本地diff review和fresh verification。未调用子智能体。

选择：复用H1已提交9aff455正式train.py，增加head_name/run/bank参数，默认H1行为保留。
不复制训练器，不采集数据，不修改H1工作区。根目录为git_workspace；分支
codex/dreamon-context-h2-training-20261107-v1；基础包含A 43a52fd、B fca5775和H1 9aff455。

审稿动机/假设：在同一完整teacher库与日程下测量H2，不把改善直接归因于多词记忆，缺少同数据重训Hctx对照。
候选只采用已有H2-GRU full-memory；不训练last-only/reset或其它结构。

公共库：/home/shx/.cache/dllm_infilling/context_h1_training_20261107_v1。
来源OpenCoder educational_instruct，保持原分组、去重、污染排除和每源每桶最多4状态。
已核对H1 complete coverage=220000，32桶训练各6250/验证各625，源组10384/1054不交叉。
H1训练已经完成；H1生成6/1033，其余1027行因BF16隐藏状态传入FP32头失败。
修复仅在本分支KeywordHeadAdapter将输入old_hidden转FP32，匹配实际缓存定义；不改heads.py、不改变选位/采样/屏障。

指标：完整验证平均T1 KL选best，严格更低才替换；源等权/32桶/T1 TV/T=.2 top_p=.9 TV和固定u采样变化保留。
固定best生成SingleLine global K4全部1033行、seed42/batch1；基线优先复用A/H1已核验3099行。
配对区间按164题聚类；H1完整结果未发布则标待补，H2独立归档退出。

技术检查命令：
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 /home/shx/projects/dllm_infilling/.venvs/dreamon-repro/bin/python -u -m experiments.context_h2_training_v1.pipeline --check-only

正式命令：bash experiments/context_h2_training_v1/launch.sh
唯一cuda:0共享、单工作进程；训练不加载DreamOn，按原分片读取，不复制公共缓存。
独立运行目录：/home/shx/.cache/dllm_infilling/context_h2_training_20261107_v1。
日志pipeline.log；进度progress.json；退出exit.json/shell_exit.json；checkpoints/best.pt与last.pt保存服务器。

成功标准：52项CPU验证+真实库学习/逐词一致检查通过；随后自动5轮以内训练、完整验证、固定生成、评分和归档。
停止条件：公共数据损坏/实质配置差异/明确失败或72h等待期满；自身OOM减微批，最小仍失败则退出；评测OOM保留训练完成状态。
普通生成失败/超时不反复抽样；评测未完整则不宣称1033最终成绩。共享负载耗时仅观测，无公平加速主张。
输出：本目录紧凑报告与证据索引、完整服务器原始数据/检查点和准确恢复命令。无新模型/搜索/多种子/自身轨迹补训，无思维导图修改。
