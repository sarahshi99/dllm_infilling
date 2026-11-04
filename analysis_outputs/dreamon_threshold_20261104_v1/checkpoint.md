# dreamon_threshold_20261104_v1 checkpoint

状态：smoke通过，执行版本准备发布；质量生成0/35122、评分0/35122、统一计时0/16728。
执行版本：共享READY.json记录；原始账本与心跳：/home/shx/projects/dllm_infilling/threshold_eval_20261104_v1。
报告：analysis_outputs/dreamon_threshold_20261104_v1/report.zh.md；审计：同目录integrity_audit.json。
恢复：bash /home/shx/projects/dllm_infilling/threshold_eval_20261104_v1/start_worker.sh --role nohead / trained；pipeline见resume.md。正常在跑时只读状态，禁止重复模型实例。共享GPU，不排队、不等待外部任务退出。
本轮单seed开发评测，历史结果不混算；未完整时不声称完成。
