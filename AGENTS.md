# AGENTS.md

This file provides guidance for Codex and other AI assistants when working with code in this repository.

## 构建与运行

```bash
pip install -e ".[mobile]"          # 安装依赖（mobile 可选，需 Appium 才安装）
rag chat                            # 交互式对话（精简模式，无日志）
rag chat -d                         # 调试模式（显示工具调用与详细日志）
```

## 架构

```
Agent Tool → Workflow → Service → Infra
单向依赖，禁止反向引用
```

四个核心能力，对应四个 Agent 工具：

| 工具 | 文件 | 链路 |
|------|------|------|
| `search_knowledge` | `agent/tools/search_knowledge.py` | 内部调 KB + Web，Agent 无感知 |
| `analyze_requirement` | `agent/tools/analyze_requirement.py` | 内部调 parser + reviewer + graph |
| `design_test_cases` | `agent/tools/design_test_cases.py` | → `workflows/testcase_design.py` → `services/case_generator.py` → LLM |
| `execute_scenario` | `agent/tools/execute_scenario.py` | → `workflows/execution.py` → mobile tools |

Mobile 工具独立存在，供 Agent 即时 UI 操作：`device_tool`、`screen_tool`、`action_tool`、`assertion_tool`。

## 必须遵守的规则

1. **生成和修改代码时添加中文注释**。所有新代码、修改过的代码必须包含中文注释说明意图。

2. **YAML 是 prompt 唯一来源**。所有 LLM system prompt 集中放在 `prompts/<id>.yaml` 独立文件里（如 `prompts/agent.yaml`），由 `src/core/prompt_loader.py::load_prompt_file(id)` 按 id 读取，文件内 `id:` 字段必须等于文件名。`configs/default.yaml` 里只存指针（如 `agent.system_prompt_id: agent`），并没有内嵌的 `prompts:` 块。代码中禁止硬编码 `_SYSTEM_PROMPT = """..."""`。

3. **tool_factory 只注册对 Agent 暴露的工具**。内部工具（search_kb、web_search、requirement_parser 等）不注册到 factories 字典。

4. **不做循环依赖**。services 不能 import workflows。

5. **新增 LLM 工具只需改 4 个文件**：`agent/tools/new_tool.py`、`agent/tool_factory.py`（3 行 import + factory）、`configs/default.yaml`（prompts 块 + tools 列表）。不要新建 Service、Workflow、Node 等中间层。

6. **`.ruff_cache` `.pytest_cache` `.idea`** 已在 `.gitignore`。

7. **知识库检索的几个不变量不要改坏**（详见 `docs/architecture.md` "知识库检索可靠性"一节）：
   - `search_knowledge` 的答案合成必须用 `temperature=0`（`cli.py::_CHAT_TEMPERATURE`），不要改回依赖全局 `llm.temperature`。
   - `prompts/agent.yaml` 的"禁止凭历史跳过检索 ⚠️ 最高优先级"必须保持独立小节的结构化写法，降级成普通 bullet 实测无效。
   - `react_loop.py` 里"未调用工具却出现 `[来源：片段N]`"的强制重试兜底依赖这个引用格式；改引用格式时要同步改 `_FABRICATED_CITATION_PATTERN`。
   - `KnowledgeBaseTool` / `SearchKnowledgeTool` 的默认 `top_k` 要和 `configs/default.yaml` 的 `retrieval.final_k`（当前 8）保持一致。
   - `search_kb.py` 有三条专项 boost 路径 + 一个对比模式，触发条件和优先级不要改坏：
     - **Excel boost**（始终激活）：拉 80 条 xlsx 测试用例，防止被 bug 行挤出。
     - **Bug boost**（`_BUG_TRIGGER_WORDS`：bug/缺陷/崩溃等）：拉 80 条 bug 记录并给 bug 记录优先名额。
     - **XMind boost**（`_XMIND_TRIGGER_WORDS`：小程序/插件）：拉 60 条 xmind 并给 xmind 优先名额；仅用于叭嗒 App 之外、只有 XMind 覆盖的小程序/插件场景。叭嗒 App 主体功能查询**默认优先 Excel**（Excel 是当前版本最全、逻辑最新的测试用例，其余来源可能是历史版本遗留、已过时），不要把通用"功能/需求"词加回触发词列表，否则会把主体功能查询误判为 xmind 优先。
     - **Comparison mode**（`_COMPARISON_TRIGGER_WORDS`：差异/区别/对比等）：与 xmind_query_mode 同时触发时（即对比对象里出现小程序/插件），XMind 和 Excel 各占 50% 槽位；对比查询需要双来源，不能独占。

8. **测试用例生成的设计方法论/优先级规则不要改坏**（`prompts/design_test_cases.yaml`）：
   - 优先级按**用例类别**赋值，不按功能重要性主观打分：P1=点击跳转，P2=展示逻辑（同一模块因条件不同展示不同 UI 变体的判断），P3=UI 展示（含整体页面 UI case）+ 异常情况。
   - 场景 B（新增功能）依赖 `analyze_requirement` 产出的 `regression_scope` 字段：`TestCaseGenerationWorkflow.build_regression_context()` 会据此逐项查询知识库现状，拼成 `regression_section` 注入 prompt，用于生成"影响面回归用例"；改字段名或结构要同步改这条链路。
   - `TestCaseGenerationWorkflow.run_from_analysis_graph()` 会兜底调用 `build_kb_samples()` 填充 `kb_samples`（此前是死代码，从未被调用，导致生成用例长期没有参考知识库风格）；不要再把这行删掉。
   - automation 模式代码层强制过滤 `case["type"]` 含"异常"的用例（`case_generator.py::_exclude_exception_cases`），不仅依赖 prompt 约束；automation 只做 UI + 核心功能回归，异常场景一律不生成。

9. **移动端自动化执行的健壮性修复不要改坏**：
   - `assert_text`/`tap` 的定位文案必须是真实 UI 渲染的文字，箭头/图标等非文字符号
     （如"追番表"文案旁的箭头图标）不会出现在 element 的 `text` 属性里，`prompts/design_test_cases.yaml`
     的 automation_requirements 明确禁止把这类符号拼进 text/target；生成器改动时不要删掉这条约束。
   - `ParsedScreen.bottom_overlay_elements()` / `bottom_overlay_top()` / `is_occluded()`
     （`src/mobile/screen_parser.py`）用启发式识别贴底的悬浮导航栏，并**必须把悬浮层
     元素自身从"是否被遮挡"判断里排除**（否则悬浮层会判定为"被自己遮挡"，依赖它
     收敛的微调滚动永远不会停）。`ActionTool._scroll_until_condition()` 命中停止条件
     后会调用 `_settle_module()`、`_resolve_coords()` 解析 tap 坐标时发现目标被遮挡会
     调用 `_nudge_into_view()`，两者都靠 `_perform_single_swipe` 的 `distance` 参数做
     小幅微调（屏幕高度 15%）——避免"滚动只判断标题可见，但卡片本体/按钮还压在
     底部 Tab 栏下面"导致断言看到不完整模块、或点空/误触导航栏。改
     `_perform_single_swipe` 时要保证 `distance=None` 时的默认滑动坐标
     （3/4 → 1/4 整屏）与原行为一致。**`_settle_module()` 判断"锚点下方是否被
     遮挡"时只能认定"骑跨在遮挡边界上"的元素**（`current.y1 <= el.y1 < boundary
     <= el.y2`，即顶部在边界以上、底部延伸到边界以下），不能用"锚点下方任意
     可见元素只要贴到边界就算"。**光是"骑线"还不够**：在可以无限下拉的长列表
     页面（RecyclerView）里，只要页面还能继续往下滚，就总会有下一张完全不
     相关的卡片正在滑入视野、贴着遮挡边界线，"骑线"条件几乎永远为真——必须
     再用 `_MODULE_HEIGHT_RATIO`（锚点往下的高度预算，屏幕高度的比例）把
     "属于锚点自身模块"和"下一个模块的内容"区分开，只处理预算范围内的骑线
     元素，超出范围一律视为下一模块、不触发滚动。否则会导致无法收敛、越滑
     越远，批量连续执行多条 case 时滚动量层层累积，最终把目标模块连同锚点
     标题本身一起滑出屏幕（且滚动方向不可逆，后续 case 永远找不回目标元素；
     这是实测踩过两次的坑，第一次只收窄了"骑线"范围但没加模块高度预算，
     仍然会在批量执行中把 002 也拖垮）。滑动距离要按骑线元素的实际重叠量
     精确计算（重叠量 + 安全余量），不能用固定滑动屏幕高度某个百分比的经验值
     反复试探。`_scroll_until_condition()` 的返回文案必须把 `_settle_module()`
     内部的微调次数计入"共滑动 N 次"，不能让这部分滚动"隐身"（否则日志显示
     滑动 0 次，但背后其实多滑了好几屏，问题无法排查）。**`max_nudges` 不要压得
     太低**：卡片内部常是"封面图 + 标题/副标题 + 按钮"纵向堆叠的多层结构，
     每次微调可能只把当前骑线的那一层滑出遮挡区，随即下一层又顶上边界线、
     变成新的骑线元素，需要再来一次才能把最底下的按钮（如"加追"）完全滑出
     遮挡区（实测 `max_nudges=2` 在这类三层卡片上不够，按钮长期留在遮挡区内
     导致后续 tap 报"未找到目标"）；由于范围已被 `_MODULE_HEIGHT_RATIO` 限定
     在锚点自身模块内、检测不到骑线元素就立即提前返回，适当调高（当前 4）
     不会重新引入"越滑越远"的旧问题。
   - `ActionTool._perform_single_swipe()` 对底层 `driver.swipe` 调用做了一次短暂重试
     （`_SWIPE_RETRY_ATTEMPTS` / `_SWIPE_RETRY_INTERVAL_S`）：UiAutomator2 偶发抛出
     `SecurityException: ... INJECT_EVENTS permission`，这是设备/环境层面的瞬时抖动
     （event injection 权限校验偶发失败），并非手势参数或代码逻辑问题，一次性失败
     没必要让整条 case 直接判失败；仍失败才把原始异常信息透出，不能静默吞掉。
   - `ActionTool._scroll_until_condition()` 的**第一次检查（滑动前，`attempt == 0`）
     失败时要先做几次短暂重试**（复用 `_LOOKUP_RETRY_ATTEMPTS`/`_LOOKUP_RETRY_INTERVAL_S`），
     确认真的找不到再决定滑动。根因：紧跟在 `back`/`tap` 跳转这类会触发 Activity
     切换动画的操作之后立即调用 scroll，Activity 转场动画/列表布局往往还没完全
     稳定，第一次检查很容易误判"目标不可见"——如果此时目标其实已经正常展示在
     屏幕上，误判会导致按剧本执行若干次"继续往下浏览"的滑动，把原本已经就位的
     模块滑到很远的地方（且这个方向的滑动在同一次调用里无法自我纠正，只会越滑
     越远），表现为"点击封面进入播放页、返回后模块明明展示正常，却看到手机
     一直在向下滑动、后续用例全部失败"。这个重试只加在 `attempt == 0` 上，不影响
     真正需要多次滑动才能找到目标的正常场景（那些场景每次 swipe 后的等待时长
     `duration_ms` 本身已经提供了自然的稳定窗口）。
   - **`ExecutionWorkflow._execute_step()`/`_settle_post_launch_ui()`/assertion 循环必须
     兜底捕获底层驱动抛出的未预期异常**（`_describe_driver_crash()`），不能让它们裸抛：
     UiAutomator2 服务端在设备侧偶发崩溃（`instrumentation process is not running`，
     长时间连续操作后的环境抖动，与代码逻辑无关）时，Appium 对任意指令都会抛出未捕获的
     `WebDriverException`；这类异常若不在这一层捕获，会一路冒泡穿透 `_run_case`/
     `execute_batch`，把整个批量执行（包括前面已经跑完、本该保留的 case 结果）一起
     打断，`execute_scenario` 变成一次性失败，Agent 只能原样转述一坨 Python 堆栈。
     正确做法是转换成普通的"步骤失败"字符串，交给调用方按"这一条 case 失败"正常收尾，
     批量执行继续跑下一条——下一条 case 开始前的 `probe_session_alive()` 探活会自动
     发现会话已死并重新连接，这一层不需要也不应该自己做重连尝试。
   - `execute_scenario` 批量执行（`ScenarioExecutionRequest.case_ids/max_cases/exclude_types`、
     `ExecutionWorkflow.execute_batch()`、`_select_cases()`）：`exclude_types` 必须先过滤
     再按 `max_cases` 截取（"前 N 条非排除类型的用例"，不是"JSON 前 N 个位置里再筛"）；
     `execute_scenario` 自身是 `FINAL_ANSWER_PASSTHROUGH`（一次调用即结束当前轮次），
     所以批量参数必须在同一次工具调用里一起表达，不能指望 Agent 多次调用来跑多条——
     `react_loop.py` 的硬路由检测到"前N条/回归/跳过"等限定词（`_has_batch_qualifier`）
     时会让步给 LLM 走正常工具调用，不要重新让硬路由无条件透传路径。

10. **元素定位/断言的几个健壮性修复不要改坏**：
    - `ParsedScreen.find_by_text()`：即使 `exact=False`（默认模糊模式），也要**先扫描一遍
      找完全相等的元素，找不到才退化为子串包含匹配**。星期切换栏这类单字目标（"一""二"等）
      极易被页面上任何包含该字符的无关文案（如"换一批"按钮）误命中——如果命中顺序在
      真正目标之前，会导致点击了完全不相关的元素却没有任何报错（断言依然能通过，因为
      断言本身也只是弱校验）。
    - `ParsedScreen.find_by_resource_id()` 必须支持 `index` 参数（同一 resource-id 常见于
      RecyclerView 中重复出现的卡片组件，如每张卡片自己的封面控件）；`ActionTool._find_target_element()`
      的 `id` 分支要把 `index` 透传下去。不要因为"id 通常唯一"就退化回只取第一个匹配——
      列表页的封面/卡片类元素基本都不唯一。
    - **禁止用笼统的 `target_type: class + 泛化 class 名 + index` 去定位列表中的封面图**：
      同一页面上有大量同类控件（含无关的推荐位图片），`index` 是按整页文档顺序数的
      第 N 个，几乎必然点到无关元素且没有任何报错（tap 本身"成功"，只是页面毫无变化）。
      能拿到具体 resource-id 时必须优先用 `target_type: id` + `index` 精确定位。
    - `ActionTool._resolve_coords()` 查找目标失败时会短暂重试（`_LOOKUP_RETRY_ATTEMPTS` /
      `_LOOKUP_RETRY_INTERVAL_S`）：应对元素需要额外接口返回后才完成渲染、短暂晚一两帧
      出现在无障碍树里的场景，不重试会造成偶发性的误报失败。
    - **"加追/关注/收藏"这类图标+文字合并渲染的按钮，实测是纯 `ImageView`**（图标和
      文案一起画进同一张位图）：点击前后 `text`/`content-desc`/`checked`/`selected`
      等无障碍属性完全不变、也不会弹 toast，即使肉眼截图能看到文案从"加追"变成
      "已加追"，UIAutomator 也拿不到任何可区分的信号——这是 App 自身的无障碍设计
      缺陷，不是代码 bug，靠增加重试次数解决不了。**禁止用 `target_type: "text"` 定位
      这类按钮**（无障碍树里根本没有这个文字），必须用 `target_type: "id"` + 具体
      resource-id（+ index）；**禁止用 `assert_text` 断言点击后的文案变化**（同样原因
      必然失败），应改为验证点击后未发生异常跳转（`assert_page`）或按钮本身仍然存在
      （`assert_element`），并在用例 notes 里注明该按钮的视觉状态变化需要人工核实
      截图，当前自动化框架无法通过无障碍树验证。
    - `AssertionTool` 新增 `assert_not_page`（`page` 语义为"要离开的原页面"）：`assert_page`
      的 `page` 字段必须是能匹配英文 Activity 类名/包名的关键字，**中文业务页面名称永远
      不可能作为子串出现在英文 Activity 类名里**，这类断言必然失败。点击封面等场景如果
      不确定目标页面的具体 Activity（例如可能落地到同一 Activity 上的半屏浮层），改用
      `assert_not_page` 只验证"已经离开当前页面"，不要瞎猜目标 Activity 名称。
    - 用 `assertion_tool` 校验"点击后是否发生跳转"时，断言必须**在 `back` 返回步骤之前**
      执行——`automation_steps` 是全部跑完才轮到外层 `assertions`，如果把断言放在外层
      `assertions` 而把 `back` 放进 `automation_steps` 尾部，断言时页面早已经被返回操作
      改变，断言必然基于错误的页面状态。需要在返回前断言时，应把 `assertion_tool` 作为
      普通 step 内嵌到 `automation_steps` 中间（执行器支持把 `assertion_tool` 当作一个
      普通工具调用），确保"先断言、再返回"的顺序正确。
    - **禁止在 `ExecutionWorkflow.execute_batch()` 的 case 之间插入任何"无条件复位滚动"
      的动作**（曾经加过又撤销的坑，别再加回来）：`ActionTool._scroll_until_condition()`
      本身在每次尝试前就会先检查目标是否已经在当前屏幕上可见，已经展示正常时不做任何
      多余滑动（"共滑动 0 次"）——这就是"先验证当前页面模块是否正常，正常就不要再滚动"
      的机制，且已经在正常工作。曾经错误地在每条 case 前加过一段盲目的"先向上滑 N 次
      再说"，意图是"抹平上一条 case 遗留的滚动偏移"，但实测这个"复位"比不做还糟：
      (1) 会把原本已经展示正常、不需要任何滚动的模块滑走，制造出本不存在的问题；
      (2) 反复顶到页面顶部会触发 App 自身的下拉刷新，导致数据/状态被意外重置（用户
      直接观察到"总下拉刷新和向上滑动"）；(3) 单纯增加的滑动调用次数会放大撞上
      Appium/UiAutomator2 偶发 `INJECT_EVENTS` 权限异常（一次性事件注入失败，
      设备/环境层面的瞬时抖动，非代码逻辑问题）的概率。真正需要"重新定位模块"的
      只有个别 case 自身的特殊步骤（例如点击进入详情页后 `back` 返回需要重新滚动），
      应该在**那条 case 自己的 automation_steps 里**加合理的 scroll 步骤解决，而不是
      对所有 case 一刀切地在批量层面强加一段无条件的复位动作。

## 编码约定

- Python 3.13，行宽 100 字符
- 类型注解：`str | None` 不用 `Optional[str]`，`list[dict]` 不用 `List[Dict]`
- import 分三段：`__future__` → 标准库 → 第三方 → `src.*`
- Appium 阻塞操作用 `asyncio.to_thread` 包装
