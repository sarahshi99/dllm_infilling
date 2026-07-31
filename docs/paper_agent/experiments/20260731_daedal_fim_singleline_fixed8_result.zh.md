# CAL authors’ DAEDAL FIM adaptation — SingleLine Fixed8 control

日期：2026-07-31 UTC。

准确标签：**CAL authors’ DAEDAL FIM adaptation — same-decoder Fixed8 control on the 838-row / 143-cluster project-non-frozen SingleLine CAL-Rest subset**。这是固定长度敏感性/同 decoder 控制，不称为 DAEDAL 原始官方 FIM，也不称为 equal-compute 或 compute-matched。

完整性 gate：`838/838` exact manifest rows，missing/duplicate/error/failure=`0/0/0/0`；12-case smoke 与 resume no-op 均通过，resume `new_rows_written=0`；frozen test=`sealed`，`test_evaluation_count=0`。

冻结 grouped analyzer（10,000 次 base-function cluster bootstrap，seed=`20260731`）结果：

- row-level Pass@1：`50.8353%`；
- equal-weight 143-cluster task-macro Pass@1：`37.5859%`；
- task-macro 95% CI：`[32.6317%,42.6749%]`；
- search/decode/total forwards：`0/3817/3817`；
- token-forwards：`928151`；
- wall time：`1106.634s` total，`1.3206s/row` mean；
- peak GPU memory：`17733972992` bytes；
- selected length：全部为 `8`；expansion/contraction 均为 `0`；termination=`all_middle_positions_filled`（838 rows）。

Dynamic arm 尚未完成时，本报告不计算 paired delta、paired CI 或 help/harm。完整 grouped artifact：`analysis_outputs/daedal_fim_singleline_fixed8_grouped_20260731_v1/`。
