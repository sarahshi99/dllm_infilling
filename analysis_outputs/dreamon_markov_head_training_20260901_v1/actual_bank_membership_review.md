# 实际训练成员与 HumanEval 候选复核

此报告逐条记录实际进入 v1 full training bank 的候选。原题意、代码和测试只保留在同目录的服务器 JSON，不提交为普通 Git 原始样本。

| HumanEval | entry point | 实际训练 transition | 匹配 | 结论 |
|---|---|---:|---|---|
| SingleLineInfilling/HumanEval/114 | `minSubArraySum` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/126 | `is_sorted` | 1 | 仅 2 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/126 | `is_sorted` | 1 | 仅 2 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/126 | `is_sorted` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/24 | `largest_divisor` | 1 | 仅 2 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/24 | `largest_divisor` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/31 | `is_prime` | 2 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/31 | `is_prime` | 3 | 仅 2 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/31 | `is_prime` | 1 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/31 | `is_prime` | 24 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/45 | `triangle_area` | 2 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/47 | `median` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/55 | `fib` | 3 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/55 | `fib` | 1 | 仅 2 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/55 | `fib` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/55 | `fib` | 2 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/57 | `monotonic` | 1 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/59 | `largest_prime_factor` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/59 | `largest_prime_factor` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/59 | `largest_prime_factor` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/59 | `largest_prime_factor` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/59 | `largest_prime_factor` | 2 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/63 | `fibfib` | 1 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/70 | `strange_sort_list` | 3 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/70 | `strange_sort_list` | 3 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/70 | `strange_sort_list` | 1 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/70 | `strange_sort_list` | 1 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/70 | `strange_sort_list` | 2 | 仅 3 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/71 | `triangle_area` | 2 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/78 | `hex_key` | 1 | 仅 5 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/8 | `sum_product` | 1 | 仅 1 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/90 | `next_smallest` | 2 | 仅 4 条规范化断言 | 仅同名入口函数的部分断言匹配；未见代码 AST 相同，不能仅据此认定抄袭或直接代码重叠。 |
| SingleLineInfilling/HumanEval/13 | `euclidean_gcd` | 1 | 代码 AST 相同 | 违反既定基准排除规则：实际训练记录与对应 HumanEval 题的去文档字符串代码 AST 相同。 |
