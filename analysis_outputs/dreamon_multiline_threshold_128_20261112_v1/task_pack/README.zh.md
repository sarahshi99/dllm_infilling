# 128预算补充任务

先读master_prompt.zh.md与planning_spec.json。目标34×5815质量结果、34×164×3独立计时。

相对于64任务只改变最大生成长度与其对应扩展次数预算为128，初始mask仍64。禁止向64账本写入。启动入口：bash /home/shx/projects/dllm_infilling/threshold_multiline128_eval_20261112_v1/start.sh。该入口单次检查容量，通过后固定smoke→READY→两个质量角色→评分→独立计时→报告和发布。显存不足返回4并记录blocked_resources，没有后台空闲轮询。
