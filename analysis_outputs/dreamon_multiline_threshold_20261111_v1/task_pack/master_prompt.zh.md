请在 dllm_infilling 项目中实际完成 DreamOn 置信度阈值方法的官方 MultiLineInfilling 完整评测。这是从已完成 single-line 到 multi-line 的受控迁移；不要把任务改成新方法研究、训练或文献搜索。实现、测试、固定 smoke、完整生成、官方评分、独立计时、分析、报告、focused commit 和 push 核实均属于本任务；除具体资源/基础设施阻塞外，不要逐阶段等待我再发指令。

【仓库、起点与协调】
- 仓库 /home/shx/projects/dllm_infilling/git_workspace；GitHub sarahshi99/dllm_infilling。
- 本任务任务书所在 worktree：/home/shx/projects/dllm_infilling/git_workspace/.worktrees/dreamon-multiline-threshold-plan-20261111-v1。
- 阅读该 worktree 的 analysis_outputs/dreamon_multiline_threshold_20261111_v1/task_pack/README.zh.md、planning_spec.json 和 population_audit.json；共享目录也有同份任务包。
- single-line 已完成结果提交 285f8f1462481d7c73e1e6659822f45fe7b59a5e；实际生成代码提交 660f2960ff65fb216f762a1eca0183835de2afbb。以包含已验证 single-line 实现及本任务包的最新本任务计划提交为起点，不回退后续必要修补。
- 在新隔离分支/worktree 实现：建议分支 codex/dreamon-multiline-threshold-eval-20261111-v1，worktree .worktrees/dreamon-multiline-threshold-eval-20261111-v1。
- task_id=dreamon_multiline_threshold_20261111_v1；共享目录 /home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1；报告 analysis_outputs/dreamon_multiline_threshold_20261111_v1/。
- 先查本任务既有进程、登记、结果和 READY，接续，不重复启动或覆盖。主控唯一修改公共代码；其它对话只使用统一入口。不要使用 subagent，不调用 Goal 工具，不修改无关历史/旧导图。
- 读 AGENTS.md；研究协议本会话按需读一次，不重跑宽泛研究流程。当前明确任务优先于旧计划内 GPU2/3、旧 Phase 编号和逐阶段审批；与本任务无关的旧阻塞不得当成当前阻塞。

【先精准读取】
1. experiments/threshold_eval/adapter.py、common.py、runner.py、analyze.py；scripts/start_threshold_worker.sh。
2. tests/threshold_eval/；tests/context_foundation/。
3. analysis_outputs/dreamon_threshold_20261104_v1/run_spec.json、summary.csv、integrity_audit.json，以及冻结模型/头的原 manifest。
4. experiments/k4_batch_common.py 的 population('multi-line')；experiments/run_k4_multiline_stage.py 的 official_evaluate；experiments/k4_official_eval_worker.py。
5. analysis_outputs/k4_multiline_20261014_v1/C/run_manifest.json；analysis_outputs/structure_coordinate_fix_20261015_v2/report.zh.md。
6. experiments/context_foundation/adapter.py 的旧共同采样参照。

【人口：必须分清 MultiLine 名称与实际跨行】
- 用 pinned 官方正常 loader 读取 HumanEval-MultiLineInfilling，共5815唯一案例、164基础题；不能只改文件名而仍读取 single-line。
- task_id 的 Lstart_Lend 表示原始孔洞端点，span=end-start+1；不要按生成文本换行数定义人口。
- 全5815 = 单行跨度1033（164题）+ 真正跨行4782（127题）。跨行进一步分：2–3行1511（127题）、4–7行1870（108题）、至少8行1401（60题）。37道基础题没有跨行案例。
- 官方 MultiLine 的1033条单行跨度记录，在对应坐标映射下与 SingleLine 输入相同，但 task_id 不同；不直接复制旧 SingleLine 输出/评分。5815全部按本轮共同协议重新生成。
- 主表以“成功数/5815、通过率=成功数÷5815”报告全官方人口，紧接一张“成功数/4782、真正跨行通过率”主表，突出跨行结论。所有分层都给清楚分母；不能只放等权通过率。
- single-line 和 multi-line 共享基础题及部分相同输入；这是孔洞跨度/任务形态迁移实验，不是独立新题或未见数据泛化。不能打开封存 external test 或旧控制器 test lock。

【模型、采样和预算：只改评测输入形态】
- Dream-org/DreamOn-v0-7B，revision 8ccc74750e43177327f29dab9e91882ba759e194；bf16，冻结。
- 旧 rank256 v2 KL/full/best 头，float32，lambda=1，冻结；通过 checkpoint_registry.json/原 run_spec 定位，不猜路径、不换头。已登记sha256 58fd352ef12ff57fa0eec47a94047559f9dfd84b5ba4478dca3395089c87265a。
- seed42；temperature=.2；top_p=.9；top_k=null；batch1；alg=entropy；alg_temp=0。
- 保持 min_gen_len=64、max_gen_len=64、max_prompt_len/max_tokens=2048、steps256、原扩删/EOS规则。不要把固定64解释成绝不扩删，也不擅自改成128/256预算。
- 多行长孔洞可能受64预算约束；保留全部失败，报告生成长度、残留mask、step-budget耗尽，并可在独立事后诊断中统计规范答案token长度≤64/>64。参考答案长度不能用于选择生成预算、阈值、提交位置或删除案例。需要扩大预算属于后续独立实验，不在本轮悄悄混入。
- 继承已审计共同随机数：FP32分数/概率、稳定截断排序、FP64逆CDF；随机键保持原 contract/seed/case/round/original_position/purpose，不加入方法、阈值、worker、调度或本轮task_id。这里 case 使用真实 MultiLine case_key。
- 不改取词为 argmax，不增加新阈值、新模型、新头、教师重算、KV cache、K8或新数据集。

【完整矩阵】
两种选位global/left × K2/K4 × nohead/trained × 阈值off/.80/.90/.95，加两选位各自K1 nohead off。
- 34配置；每配置5815；共197710条质量结果。
- nohead角色18配置，共104670条；trained角色16配置，共93040条。
- 每配置真正跨行4782条；34配置跨行共162588条，是总质量人口的子集，不是额外新生成。
- off 全部重跑并完整报告；K1两策略分别跑。不得只跑 single-line 中表现好的配置，也不根据早期结果省略负结果。

【阈值与动作：严格保留 single-line 已验证定义】
- 无头分数z_i；合法头修正位置用z_i+delta_i。相同合法动作屏蔽后、temperature/top-p截断前，算FP32 T=1完整词表概率。
- 实际候选仍按T=.2/top-p=.9共同采样；阈值用被抽到的普通词y_i的p_i(y_i)，不是top1概率或选位负熵。仍合法的特殊词留在概率分母。
- 每轮仅原空间执行顺序的首个候选豁免普通词阈值；不是每条链都保底一个。global先按原置信度选候选，再按空间链执行；left按原左前沿。
- 其它普通候选先预测/修正，再判阈值；通过才写入画布和前驱记忆。拒绝保留mask、不重抽、不写前驱，停止该连续链剩余候选；其它独立链可继续，其链首也要过阈值；不跨间隔传前驱。无头、有头断链一致。
- 原始expand/delete按修补版只执行一次并结束整轮；仅由头修正出来的结构词保留原屏障，不直接提交。继承扩展预算、EOS广播、坐标移动、mask/no-op和特殊词记忆排除规则。
- nohead实际不调用头；零修正头只作真实接口一致性测试，不增加完整零头矩阵。off恢复对应共同采样基线；K1不使用额外词阈值。

【实现边界与版本】
- 旧源码多处写死single-line/1033/35122/18594/16528，以及164组统计。必须集中由任务spec驱动数据入口、分母、矩阵计数、角色完成条件、完整性审计、图表、报告和checkpoint，不能只修改population()。
- bootstrap按当前报告人口中实际存在的基础题成组重采样：全体164；跨行127；各跨度层164/127/108/60。不要把缺席组填0，也不要因为缺席组而不输出区间。统计量仍以逐例通过率/配对通过率差为主。
- 旧single-line共享路径、结果、完成状态不得改写。新task路径同时传到shell入口，不能继承旧SHARED常量误写旧目录。
- 参数化不得改变旧single-line采样与动作；CPU测试及固定旧案例真实模型回归比较逐token、逐轮画布和结构动作。
- 公共实现验证后提交，原子写READY.json，记录worktree与execution_commit。READY只是代码完成信号，不代表GPU空闲或额外审批。
- 结果记录实际生成执行版本；恢复以Git核实相关生成代码一致，允许无关文档提交，不应因文档HEAD变化误判生成版本。生成语义修复后，重跑受影响结果并保留旧尝试，不能混合执行版本。

【smoke与资源】
- 固定16个真实官方MultiLine案例，四个跨度层各4例，运行前只按case_key/跨度稳定选取；覆盖两策略、K2/K4、有无头、结构动作。off新旧参照逐token/画布/结构一致；无头与实际接口零修正一致；检查拒绝不重抽、不入记忆、不跨间隔、K上限、结构坐标、事务恢复不重复、诊断开关不改输出。
- 结构边界的极端情况可用针对性CPU测试，不按模型通过结果挑smoke，不人工捏造正式案例。
- 用固定smoke的实测生成时间给出全矩阵工时粗估和共享争用限制，不根据通过率或估时省略配置，不套用旧任务的过期预算上限。
- 主控只短暂加载一个smoke模型，smoke完成退出释放后再启动两个质量角色；主控不长期驻留第三个模型。
- 服务器当前仅一张H200，沿用实时可见映射，不硬编码旧GPU2/3；记录逻辑号、实际UUID、GPU负载和本轮PID。
- 本任务按两个质量角色同卡并发安排，各角色只常驻一个模型并依次处理自己的全部配置；同角色防重复锁保留，不设GPU互斥锁，不等待“GPU空闲”，不抢占/终止/改动其它任务。
- 计划时观察到唯一GPU显存139284/143771 MiB，仅余约4487 MiB，无法据此承诺可立即装入模型。执行时重新核对实际容量；容量不足就记录具体资源阻塞并完成可做的CPU实现，不启动长时间空闲等待，不把两个角色擅自改为串行。资源可运行时直接执行已授权完整矩阵。
- OOM只释放失败进程自己的资源、立即原口径恢复一次；仍失败则记录，不影响另一个正常角色。其它基础设施至多重试3次，保留尝试；普通代码错误/官方测试失败/3秒超时不重抽。

【持久执行与跨对话】
- 尽早建立run_spec.json、status.json、任务账本、心跳/进程/命令日志；共享目录内planning_spec和任务包是计划，不能把它们当已经生成的结果。
- 统一入口：bash /home/shx/projects/dllm_infilling/threshold_multiline_eval_20261111_v1/start_worker.sh --role nohead 或 --role trained。
- 两个worker可以由主控直接通过tmux启动，不依赖额外对话。若我另外打开worker对话，它们只登记/监督并调用同一入口，看到已有角色不得再加载。
- READY后主控自动检查并补启动缺席角色；其它对话晚到只接续检查。代码未就绪可用tmux小型等待器等待READY，不加载模型，不靠语言模型轮询。若主控明确报告资源阻塞，不把READY等待器改成GPU空闲轮询器。
- 展示实际命令后直接执行。tmux持久串起：完整生成→两个质量模型自然退出→官方评分→统一计时→审计/分析/报告→focused commit/push并核实远端。
- 每例事务落盘，唯一键至少覆盖task/config/case；计时另加repetition。中断只补缺失事务，尝试记录不算新样本。不要根据得分重跑失败代码。
- 使用官方已验证bwrap评测入口和每例3秒；CPU评分控制并发/线程，避免人为制造超时。

【统一计时，独立于质量人口】
- 在看结果前固定164个计时案例：每个基础题选1例；有跨行候选的127题只从跨行里选，37个只有单行孔洞的题从单行里选；在eligible案例里取stable(['multiline_timing_v1',case_key])最小者。任务包timing_cases.planned.json已按此生成，核对官方人口后使用。
- 固定34配置各生成3次，共16728次；其中真正跨行127×34×3=12954次，其余3774次是单行回退。两部分分别报告，不能把含37个单行回退的计时整体称为纯多行计时。
- 预热、同步CUDA、稳定轮换配置次序；时间包括模型/头/阈值/选位/采样/结构，排除加载、代码测试、写盘和重型轨迹。每次恢复先预热，再补缺失重复。
- 两个质量worker并发阶段的时间只作共享负载观测。统一计时等本轮质量模型自然释放后启动，不等外部任务空闲，不要求独占GPU。外部争用如实标注，不作确定加速达标判断。
- 明确计时是每题一个固定案例的子集；质量全5815与各跨度层不应被该子集时间包装成全人口时延。跨行时间与相同127题/相同案例的K1配对，禁止用164题混合K1作分母。

【主报告必须符合我的展示要求】
- 第一主表：34配置全部列出，包含所有off和两个K1，成功数/5815、逐例通过率、相对匹配K1的pp、平均/中位/p90计时、时间比、普通词/前向。不以基础题等权替代逐例通过率。
- 第二主表：34配置在真正跨行4782例上的相同质量指标，以及匹配跨行计时子集；明确其127基础题。
- 跨度1/2–3/4–7/至少8行分别列分母、成功数、通过率，所有负结果保留。总体最高点不自动叫跨行最优点。
- 相对对应K1、同方法off、同阈值另一方法的配对救回/改错/净变化、逐例通过率差与整题bootstrap95%区间（10000次）；题等权仅辅助列。
- 同时比较同阈值，以及已测点中相近“逐例通过率”下的质量—时间关系；不假定有头和无头置信度已校准。
- NFE、普通提交总数与分布、额外词接受/拒绝、链跳过、结构/no-op、头调用、生成长度、未完成率、峰值显存。K1不适用的额外词接受率写N/A，不写“未完成”。
- 预设探索标记分口径保存：沿用上一轮“基础题等权下降≤1pp且减时≥20%”用于历史定义一致；另列我关注的“逐例通过率下降≤1pp且减时≥20%”。全人口与跨行分别报告，不能悄悄改旧定义；共享争用时不宣称确定达标。
- 与single-line只做任务形态对照，不拼成独立样本或混合总通过率；不复用旧历史951/817/831等结果作本轮配对。
- 至少保存run_spec、commands、status/resume、summary.csv、crossline_summary.csv、span_summary.csv、paired_comparisons.csv、timing.csv（含人口字段）、可复算逐例证据、attempts、integrity_audit.json、质量—耗时/实际并行度曲线、report.zh.md和compact checkpoint。
- 完整性：34×5815=197710质量行；每配置1033+4782=5815；各层1033+1511+1870+1401=5815；18/16角色计数；16728独立计时行及12954跨行子集；每配置492次/跨行381次；缺失、重复、无效、版本、计时token分歧显式核对。
- 最后本地diff review + fresh verification，focused commit/push核实；大日志、权重、重型trace不进普通Git。
- 最终给真实完成数、所有34配置的主要结果、无头阈值是否仍有效、头是否有额外价值、不同跨度是否不同、共享计时限制、报告路径与提交。未完成就写实际阻塞；“后台已启动”不等于“评测已完成”。
