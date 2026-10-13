# DreamOn + Markov K2 诊断：共享 H200 后台运行

用户后续授权其他进程共存。先固定16行检查和估时，通过后自动10330行；显存不足时等待，GPU作业累计最多12小时。

历史空卡限制下的阻塞报告保存在 before_shared_gpu_authorization/。进度见 run_manifest.json、smoke_progress.json、progress.json 和后台日志。完成或阻塞后自动生成最终分析、更新记录并推送；不启动新方法。
