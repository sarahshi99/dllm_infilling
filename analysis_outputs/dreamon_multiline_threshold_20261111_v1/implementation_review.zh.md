# 公共实现本地审查

reviewer_gate_disabled；按任务要求不使用subagent。本地diff review + fresh verification。
修复旧shell硬编码READY路径，新增任务wrapper同时设置task与shared，命令记录显式带task环境；旧single-line结果目录未改动。
人口、角色完成、审计、checkpoint由spec派生；bootstrap只用实际出现基础题；跨行计时以同一127案例K1匹配。新增流水线释放顺序、同角色锁、实际span和固定采样测试。
生产解码器adapter.py未修改；真实smoke16例及旧单行4例逐token/逐轮画布/结构一致；无头与真实零修正接口一致；diagnostic开关输出不变，原始结构动作覆盖，极端expand在CPU用例覆盖。
ordinary代码错误退出2不做基础设施重试；OOM退出42立即恢复最多一次；OSError等基础设施退出75。所有质量结果逐例事务；Git按相关路径比较，纯报告提交允许恢复。
完整性审计检查计时输出分歧也阻止完成，保留缺失计数/前100键，可由spec与逐例证据复算全缺失集合，避免提交19MiB空人口缺失清单。
残余限制：单seed重复开发题、固定64预算、共享GPU争用；完整矩阵和独立计时尚未完成，不能作方法结论。
