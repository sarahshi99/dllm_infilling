# 2026-10-13 A：DreamCoder 来源与冻结迁移审计（追加节点）

任务：保全已完成 DreamOn 诊断、核对 v2 KL 头与普通词表、准备原始 DreamCoder 固定64辅助迁移。仅 CPU 与来源联网检查；GPU 由主控统一调度，A 不启动。研究人口是1033行/164基础题开发诊断集，seed42；无训练、无新封存测试访问。已有 DreamOn 自身结果只作为校准，原始Base单列，不能填补另两份DreamOn权重。

维护动机：隔离来源训练贡献与二次抽样路径贡献；完整词表和原生长度语义是可解释迁移的必要条件。成功标准：既有逐行证据10330条唯一主键、源头严格加载全部参数、普通词元一致、原生参考与适配器在固定16条上相同。技术错误暂停该目标；不按正收益决定是否全量。输出限定本文件、A独占适配/测试文件和 analysis_outputs/zero_training_transfer_20261013/a_dreamcoder/。遵循 karpathy-guidelines；公共接口由主控合入。

## 既有证据保全

只读原诊断 SQLite 一次，导出无损紧凑 gzip JSONL：`dreamon_reused_per_case.jsonl.gz`，包括输出、token ID、评测、seed、前向、相邻/回退/结构/头触发分母。完整10330行，十组各1033、164基础题，缺失/重复/未恢复错误均0；既有16条检查passed。旧报告和原始SQLite不改写、不重评测。新增 `dreamon_reuse_paired.json` 补D−B及其余目标配对；沿用旧clustered_difference，164题整组bootstrap10000次，seed20261013，行加权与题等权同时给出。

原始运行是共享H200，2886.023884秒 GPU作业成本=0.801673小时。该历史计时不能作为正式独占速度比；本次A审计GPU成本0。生成代码a55b1fd78795ae5f83845302a76f31780dda8cd2，用户核定诊断HEAD340170ce6ae44477027dd7e94ec7a9c0f0910384包含收尾修复。
