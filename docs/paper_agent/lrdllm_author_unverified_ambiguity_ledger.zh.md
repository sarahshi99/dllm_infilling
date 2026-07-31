# LR-DLLM Author-Unverified Ambiguity Ledger and Preregistration

日期：2026-07-31 UTC

实现标签固定为：**paper-guided, author-unverified reimplementation of LR-DLLM**。

## Source search conclusion

截至本登记冻结日，arXiv `2602.07546v1`、Jia Li 公开 publication page、定向 GitHub 搜索、OpenReview、PMLR 和论文主页均未发现作者代码、commit 或 license。该结论只允许本地 paper-guided reimplementation，不支持 official、paper-faithful 或 exact reproduction 表述。

## Primary literal-Algorithm-1 choices

以下选择在任何 Pass@1、evaluator outcome 或正式机制分布之前冻结：

1. **Confidence**：对每个当前 masked position 的 raw model logits 做 softmax，计算 `sum_v p(v) log p(v)`，再对全部 masked positions 取算术平均。它是负熵，数值越接近 0 表示越高置信度。长度选择阶段不应用 temperature 或 top-p。
2. **Stage I**：严格 probe `{1,2,4,8,16,32,64,128}`；对全部八个 `(log L, A(L))` 点做 ordinary least squares，得到 `alpha,k`；不只拟合 long-span 子集。
3. **Adjusted confidence**：所有 Stage I 和 Stage II 候选均用冻结的 Stage I slope `k` 计算 `CL(L)=A(L)-k log L`。Stage II 不重新拟合 `k`。
4. **Initial choice**：在八个 Stage I probe 中选最大 `CL`。浮点分数在绝对差 `<=1e-12` 时视为平局，并选更短长度。
5. **Local search**：每个 token commit 前，在边界 `[1,128]` 内评估 `{L-1,L,L+1}`。只有候选 `CL` 严格高于当前 `CL+1e-12` 才移动；多个严格改善候选平局时选更短长度。没有严格改善即稳定。
6. **Termination guard**：每次 token commit 前最多允许 128 次长度移动。由于同一 prefix 下只接受严格改善且长度域有限，正常路径应更早稳定；触发 guard 是技术失败，不提交该行。
7. **Left-token commit**：稳定后，从稳定 canvas 最左侧 masked position 的 logits 以 `temperature=0.2, top_p=0.9` 采样一个 token，追加到生成 prefix；随后剩余长度减一，再进入下一轮局部搜索。
8. **Generation termination**：剩余长度降到 0 时结束；最多提交 128 个 token。动态长度而不是 EOS 控制中间 span 终止。
9. **Special tokens**：mask、pad、BOS、EOS 和 tokenizer 声明的其他 special token 在 top-p 采样前全部屏蔽。若屏蔽后没有有限候选则该行技术失败。
10. **Seed**：主结果只运行 seed `42`。为保证 resume/dedup 不改变随机流，每个 candidate key 使用 `SHA256("42|candidate_key")` 的低 32 bit 作为 row seed，并写入 raw；同协议 Fixed64 必须使用相同派生规则。
11. **Forward reuse**：同一已提交 prefix 下，相同候选长度的 confidence forward 允许 memoize；token commit 后 cache 清空。不同长度不共享 KV/cache，不跨 token commit 复用 hidden state。cache hit 不计 forward。
12. **Accounting**：每次实际 model call 计一个 forward；`token_forwards` 增加该 call 的 batch-size × input-sequence-length。分别登记 Stage I search、Stage II search、commit sampling和 total forward/token counts、cache hits、wall time、peak GPU memory。
13. **Canvas**：DreamCoder 使用其公开 remote-code 所需的 `BOS + committed-prefix + masks(L) + suffix + EOS` fixed canvas 和 shifted-logit alignment。Stage II 只改变 mask span与已提交 prefix，不改变每例 suffix 或 backbone 算法。
14. **No oracle/evaluator leakage**：选择器和 manifest builder不得读取 `canonical_solution`、reference middle/oracle length、tests、evaluator pass/fail 或任何历史 outcome。evaluator 只能在整条 completion 已固定后调用。冻结 test 文件不打开，`test_evaluation_count=0`。
15. **Resume/dedup**：canonical raw append-only；candidate key 唯一；成功行 resume no-op；error 只进入 failure journal；重新运行不得覆盖成功行或改变 row seed。

## Registered sensitivities, not primary

只有主实现完成全部技术 gate 后，且另行建立 immutable manifest/action brief，才可运行以下 sensitivity；不得根据主结果挑选：

- 仅用 long probes 拟合 `k`；
- confidence 使用 temperature-scaled distribution；
- Stage II 每个 token 重新拟合 slope；
- 平局时选更长长度；
- 不做 within-prefix confidence memoization。

top-p 截断后再计算 confidence 不作为当前批准 arm；其语义会把 sampling policy 混入长度置信度。

## Technical gates

- pure-unit gate：negative entropy、probe grid、OLS log-length fit、tie/boundary、expansion/contraction、termination、left-token commit、MAX_GEN、forward/token ledger、resume/dedup 和 forbidden-input guard 全覆盖。
- 12-case technical smoke：12 exact unique keys，0 missing/duplicate/error/failure；ledger 守恒；resume no-op；frozen=`sealed/0`。
- 64-case mechanism smoke：manifest 在 outcome 前冻结；0 missing/duplicate/error/failure；真实 expansion 与 contraction 都至少发生一次；无 guard/infinite-search；成本可解释；resume 幂等；不读取 accuracy 选择变体。
- full 顺序：DreamCoder SingleLine 927 → RandomSpan 1480 → MultiLine 5079；三者分别报告。DreamCoder 技术可信后才移植 LLaDA 的相同三项。每个 backbone 需要相同协议 Fixed64。

若主实现无法满足 gate，记录精确 blocker并停止 LR-DLLM phase；不把本地 heuristic 包装成 LR-DLLM，也不阻塞独立 CAL/DAEDAL 工作。
