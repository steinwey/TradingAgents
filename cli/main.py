# 中文阅读导航：本文件是命令行入口，负责收集设置、启动分析、展示进度和保存报告。
# 建议按以下顺序跳读；行号对应当前文件，后续增删代码时需同步更新，可用 Ctrl+G 跳转：
#   ① app()（第 1526 行）→ analyze()（第 1421 行）：看命令如何进入程序。
#   ② run_analysis()（第 1089 行）：看一次交互分析的完整主线（内部按 A～H 分块）。
#   ③ get_user_selections()（第 530 行）→ _prompt_selections()（第 537 行）→ _build_run_config()（第 1047 行）：看配置来源。
#   ④ graph.graph.stream(...)（第 1233 行）：看图的输出如何进入命令行界面。
#   ⑤ MessageBuffer（第 100 行）→ update_analyst_statuses()（第 911 行）→ update_display()（第 317 行）：看进度和报告展示。
#   ⑥ backtest()（第 1476 行）：需要理解批量历史评估时再读。
# 核心业务流程在 tradingagents/graph/setup.py，Agent 提示词在 tradingagents/agents/。
# 注意三类数据：selections 是用户选择，config 是引擎配置，chunk 是图流出的状态。
# message_buffer 则保存界面所需的数据；其中的状态标记不会调度 Agent 执行。

# ==================== 1. 依赖与 CLI 初始化 ====================
# Typer 注册命令和解析参数；Rich 负责终端布局、表格及实时刷新。
# 项目内导入则连接交互工具、模型调用统计、分析引擎和报告写入器。
import datetime
import os
import sys
import time
from collections import deque
from functools import wraps
from pathlib import Path

import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from cli.announcements import display_announcements, fetch_announcements
from cli.prefs import load_last_run, sanitize, save_last_run
from cli.stats_handler import StatsCallbackHandler
from cli.utils import (
    ask_anthropic_effort,
    ask_gemini_thinking_config,
    ask_glm_region,
    ask_minimax_region,
    ask_openai_reasoning_effort,
    ask_output_language,
    ask_qwen_region,
    confirm_ollama_endpoint,
    detect_asset_type,
    ensure_api_key,
    get_ticker,
    prompt_openai_compatible_url,
    resolve_backend_url,
    select_analysts,
    select_deep_thinking_agent,
    select_llm_provider,
    select_research_depth,
    select_shallow_thinking_agent,
)
from tradingagents.agents.utils.rating import is_review
from tradingagents.backtest import iter_grid, run_backtest, summarize
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.portfolio import load_portfolio
from tradingagents.reporting import write_report_tree

console = Console()

# prompt_toolkit 的 Windows 输出模块在导入时检查平台，因此只在 Windows 上导入。
# 先判断平台，可让 Windows 上真实的依赖故障正常暴露，
# 避免捕获所有导入异常后悄悄禁用下方的异常处理。
# 其他平台使用空元组作为异常类型集合，
# except 接受空元组，但不会匹配任何异常（问题 #1138）。
if sys.platform == "win32":  # pragma: no cover — 依赖运行平台
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError

    _NO_CONSOLE_ERRORS: tuple[type[BaseException], ...] = (NoConsoleScreenBufferError,)
else:
    _NO_CONSOLE_ERRORS = ()

app = typer.Typer(
    name="TradingAgents",
    help="TradingAgents CLI: Multi-Agents LLM Financial Trading Framework",
    add_completion=True,  # 启用命令行自动补全
)


# ==================== 2. 界面数据缓冲区：MessageBuffer ====================
# 阅读重点：init_for_analysis() 初始化 → add/update 方法接收变化 → 报告拼接方法。
# messages/tool_calls 只保留近期记录供屏幕展示，完整过程由 run_analysis() 写入日志。
# report_sections 保存各阶段报告；agent_status 记录 pending/in_progress/completed 等展示状态。
# 使用有长度上限的双端队列保存近期消息。
class MessageBuffer:
    # 每次分析都会运行的固定团队，用户不能单独选择。
    FIXED_AGENTS = {
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # 分析师配置键与展示名称的对应关系。
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "social": "Sentiment Analyst",
        "news": "News Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # 报告章节映射：章节键 →（控制该章节的分析师键，负责定稿的智能体）。
    # 分析师键用于按用户选择筛选章节；None 表示始终包含。
    # 只有负责定稿的智能体状态为 completed，该报告才计为完成。
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "sentiment_report": ("social", "Sentiment Analyst"),
        "news_report": ("news", "News Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "investment_plan": (None, "Research Manager"),
        "trader_investment_plan": (None, "Trader"),
        "final_trade_decision": (None, "Portfolio Manager"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # 保存完整的最终报告
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._processed_message_ids = set()

    def init_for_analysis(self, selected_analysts):
        """根据所选分析师初始化智能体状态与报告章节。

        参数：
            selected_analysts：分析师类型字符串列表，例如 ["market", "news"]。"""
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # 根据本次选择动态构建智能体状态表。
        self.agent_status = {}

        # 加入用户选择的分析师。
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # 加入固定团队。
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # 根据选择动态构建报告章节。
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # 重置其余展示状态。
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._processed_message_ids.clear()

    def get_completed_reports_count(self):
        """统计已经定稿的报告数量。

        报告计为完成须同时满足：
        1. 对应章节已有内容，不为 None。
        2. 负责定稿的智能体状态为 completed。

        这样可避免把辩论过程中的中间更新误计为完成。"""
        # 有中间文本不代表报告已经定稿，还要检查负责该报告的 Agent 是否完成。
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # 报告有内容，且负责定稿的智能体已完成，才计入完成数。
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # 按 report_sections 的阶段排列取最后一个非 None 的内容，供当前报告面板展示。
        # 这里没有更新时间记录；下面还会重新拼接完整报告。
        # 选择一个阶段的报告内容供面板展示。
        latest_section = None
        latest_content = None

        # 按章节顺序查找最后一个已有内容的章节。
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            # 为当前章节添加展示标题。
            section_titles = {
                "market_report": "Market Analysis",
                "sentiment_report": "Social Sentiment",
                "news_report": "News Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "investment_plan": "Research Team Decision",
                "trader_investment_plan": "Trading Team Plan",
                "final_trade_decision": "Portfolio Management Decision",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n{latest_content}"
            )

        # 同步更新完整报告。
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # 汇总分析师报告；使用 .get() 兼容未选中或缺失的章节。
        analyst_sections = ["market_report", "sentiment_report", "news_report", "fundamentals_report"]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n{self.report_sections['market_report']}"
                )
            if self.report_sections.get("sentiment_report"):
                report_parts.append(
                    f"### Social Sentiment\n{self.report_sections['sentiment_report']}"
                )
            if self.report_sections.get("news_report"):
                report_parts.append(
                    f"### News Analysis\n{self.report_sections['news_report']}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n{self.report_sections['fundamentals_report']}"
                )

        # 汇总研究团队报告。
        if self.report_sections.get("investment_plan"):
            report_parts.append("## Research Team Decision")
            report_parts.append(f"{self.report_sections['investment_plan']}")

        # 汇总交易团队报告。
        if self.report_sections.get("trader_investment_plan"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(f"{self.report_sections['trader_investment_plan']}")

        # 汇总投资组合经理的最终决策。
        if self.report_sections.get("final_trade_decision"):
            report_parts.append("## Portfolio Management Decision")
            report_parts.append(f"{self.report_sections['final_trade_decision']}")

        self.final_report = "\n\n".join(report_parts) if report_parts else None


# CLI 共用一个展示缓冲区，每次 run_analysis() 会调用 init_for_analysis() 重置内容。
message_buffer = MessageBuffer()


# ==================== 3. 终端界面：布局与刷新 ====================
# create_layout() 划分区域；update_display() 从缓冲区读取内容并重绘。
# 第一遍只需理解区域和数据来源，表格宽度、颜色、边框等细节可以跳过。
def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """将词元数量格式化为便于展示的文本。"""
    if n >= 1000:
        return f"{n/1000:.1f}k"
    return str(n)


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # 展示分为五块：标题、团队进度、近期消息/工具、当前报告、底部统计。
    # stats_handler 提供调用次数和 token 用量，message_buffer 提供 Agent 与报告进度。
    # 顶部区域：显示欢迎信息。
    layout["header"].update(
        Panel(
            "[bold green]Welcome to TradingAgents CLI[/bold green]\n"
            "[dim]© [Tauric Research](https://github.com/TauricResearch)[/dim]",
            title="Welcome to TradingAgents",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )

    # 进度区域：显示各智能体状态。
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,  # 使用带横线的简洁表头
        title=None,  # 不重复显示进度标题
        padding=(0, 2),  # 增加左右内边距
        expand=True,  # 让表格填满可用空间
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    # 按团队组织智能体，仅保留本次状态表中存在的成员。
    all_teams = {
        "Analyst Team": [
            "Market Analyst",
            "Sentiment Analyst",
            "News Analyst",
            "Fundamentals Analyst",
        ],
        "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
        "Trading Team": ["Trader"],
        "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
        "Portfolio Management": ["Portfolio Manager"],
    }

    # 筛选出本次实际参与的团队成员。
    teams = {}
    for team, agents in all_teams.items():
        active_agents = [a for a in agents if a in message_buffer.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        # 添加团队首个成员，同时显示团队名称。
        first_agent = agents[0]
        status = message_buffer.agent_status.get(first_agent, "pending")
        if status == "in_progress":
            spinner = Spinner(
                "dots", text="[blue]in_progress[/blue]", style="bold cyan"
            )
            status_cell = spinner
        else:
            status_color = {
                "pending": "yellow",
                "completed": "green",
                "error": "red",
            }.get(status, "white")
            status_cell = f"[{status_color}]{status}[/{status_color}]"
        progress_table.add_row(team, first_agent, status_cell)

        # 添加该团队的其余成员。
        for agent in agents[1:]:
            status = message_buffer.agent_status.get(agent, "pending")
            if status == "in_progress":
                spinner = Spinner(
                    "dots", text="[blue]in_progress[/blue]", style="bold cyan"
                )
                status_cell = spinner
            else:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "error": "red",
                }.get(status, "white")
                status_cell = f"[{status_color}]{status}[/{status_color}]"
            progress_table.add_row("", agent, status_cell)

        # 在每个团队后添加分隔线。
        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    layout["progress"].update(
        Panel(progress_table, title="Progress", border_style="cyan", padding=(1, 2))
    )

    # 消息区域：显示近期消息和工具调用。
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,  # 让表格填满可用空间
        box=box.MINIMAL,  # 使用简洁边框样式
        show_lines=True,  # 保留行间分隔线
        padding=(0, 1),  # 增加列间留白
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column(
        "Content", style="white", no_wrap=False, ratio=1
    )  # 让内容列自动扩展

    # 合并工具调用记录和普通消息。
    all_messages = []

    # 加入工具调用记录。
    for timestamp, tool_name, args in message_buffer.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    # 加入普通消息。
    for timestamp, msg_type, content in message_buffer.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    # 按时间倒序排列，最新消息排在最前。
    all_messages.sort(key=lambda x: x[0], reverse=True)

    # 限制面板最多展示的消息条数。
    max_messages = 12

    # 取最新的若干条消息。
    recent_messages = all_messages[:max_messages]

    # 将已按时间倒序排列的消息加入表格。
    for timestamp, msg_type, content in recent_messages:
        # 允许消息内容自动换行。
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    layout["messages"].update(
        Panel(
            messages_table,
            title="Messages & Tools",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # 分析区域：显示当前报告。
    if message_buffer.current_report:
        layout["analysis"].update(
            Panel(
                Markdown(message_buffer.current_report),
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        layout["analysis"].update(
            Panel(
                "[italic]Waiting for analysis report...[/italic]",
                title="Current Report",
                border_style="green",
                padding=(1, 2),
            )
        )

    # 底部区域：显示运行统计。
    # 根据智能体状态表统计完成进度。
    agents_completed = sum(
        1 for status in message_buffer.agent_status.values() if status == "completed"
    )
    agents_total = len(message_buffer.agent_status)

    # 报告进度同时检查内容和负责定稿的智能体是否完成。
    reports_completed = message_buffer.get_completed_reports_count()
    reports_total = len(message_buffer.report_sections)

    # 组织底部统计字段。
    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    # 从回调处理器读取模型与工具调用统计。
    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")

        # 优先展示输入和输出词元用量，数据不足时退回总量展示。
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    # 计算已运行时长。
    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))

    layout["footer"].update(Panel(stats_table, border_style="grey50"))


# ==================== 4. 交互配置：收集一次运行的 selections ====================
# get_user_selections() 读取并保存上次选择；_prompt_selections() 执行具体问答。
# prefs 用作菜单默认值；受支持的环境变量可以跳过对应问题，具体规则看各分支。
# 最终返回的 selections 还要交给 _build_run_config() 转换成引擎配置。
def get_user_selections():
    """收集本次运行设置，并以上次运行的选择作为默认值。"""
    selections = _prompt_selections(load_last_run())
    save_last_run(selections)
    return selections


def _prompt_selections(prefs):
    """逐步执行配置问答；prefs 提供默认值，环境变量可跳过对应问题。"""
    # 显示字符画欢迎信息。
    with open(Path(__file__).parent / "static" / "welcome.txt", encoding="utf-8") as f:  #__file__当前python文件路径 .parent 取得所在目录 / 是路径拼接
        welcome_ascii = f.read()

    # 构建欢迎面板内容。
    welcome_content = f"{welcome_ascii}\n"
    welcome_content += "[bold green]TradingAgents: Multi-Agents LLM Financial Trading Framework - CLI[/bold green]\n\n"
    welcome_content += "[bold]Workflow Steps:[/bold]\n"
    welcome_content += "I. Analyst Team → II. Research Team → III. Trader → IV. Risk Management → V. Portfolio Management\n\n"
    welcome_content += (
        "[dim]Built by [Tauric Research](https://github.com/TauricResearch)[/dim]"
    )

    # 创建欢迎面板并居中展示。
    welcome_box = Panel(
        welcome_content,
        border_style="green",
        padding=(1, 2),
        title="Welcome to TradingAgents",
        subtitle="Multi-Agents LLM Financial Trading Framework",
    )
    console.print(Align.center(welcome_box))
    console.print()
    console.print()  # 在公告前增加空行

    # 获取并显示公告；获取失败时不打断流程。
    announcements = fetch_announcements()
    display_announcements(console, announcements)

    # 为每一步问答生成带边框的提示面板。
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    def thinking_value_or_prompt(env_var, config_key, label, box_title, box_body, prompt_fn):
        """优先返回环境变量配置的推理参数，否则向用户询问。

        设置 env_var 时跳过交互，使用 DEFAULT_CONFIG 中已经应用环境变量的值，
        与其他配置步骤的环境变量优先规则保持一致。"""
        if os.environ.get(env_var):
            value = DEFAULT_CONFIG[config_key]
            console.print(f"[green]✓ {label} from environment:[/green] {value}")
            return value
        console.print(create_question_box(box_title, box_body))
        return prompt_fn()

    # 4A. 分析对象：标的、自动识别的资产类型、分析日期。
    # 步骤一：输入标的代码。
    console.print(
        create_question_box(
            "Step 1: Ticker Symbol",
            "Enter the ticker, with exchange suffix when needed (e.g. SPY, 0700.HK, BTC-USD)",
            "SPY",
        )
    )
    selected_ticker = get_ticker()
    asset_type = detect_asset_type(selected_ticker) #识别是否为加密资产
    # 仅在识别为非股票资产时显示提示，
    # 避免每次运行都重复提示默认的股票类型。
    if asset_type.value != "stock":
        console.print(
            f"[green]Detected asset type:[/green] {asset_type.value}"
        )

    # 步骤二：输入分析日期。
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # 4B. 分析范围：输出语言、参与的分析师、多空与风险辩论深度。
    # 步骤三：选择输出语言；设置对应环境变量时跳过问答。
    if os.environ.get("TRADINGAGENTS_OUTPUT_LANGUAGE"):
        output_language = DEFAULT_CONFIG["output_language"]
        console.print(
            f"[green]✓ Output language from environment:[/green] {output_language}"
        )
    else:
        console.print(
            create_question_box(
                "Step 3: Output Language",
                "Select the language for analyst reports and final decision"
            )
        )
        output_language = ask_output_language(prefs.get("output_language"))

    # 步骤四：选择参与分析的分析师。
    console.print(
        create_question_box(
            "Step 4: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    prefs = sanitize(prefs, asset_type.value)
    selected_analysts = select_analysts(asset_type, prefs.get("analysts"))
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # 步骤五：选择研究深度；两个辩论轮数都由环境变量指定时跳过问答。
    # 研究深度对应多空辩论与风险讨论的轮数。
    # 当 TRADINGAGENTS_MAX_DEBATE_ROUNDS 和 TRADINGAGENTS_MAX_RISK_ROUNDS
    # 均已设置时，直接采用环境变量中的值（问题 #977）。
    depth_from_env = bool(os.environ.get("TRADINGAGENTS_MAX_DEBATE_ROUNDS")) and bool(
        os.environ.get("TRADINGAGENTS_MAX_RISK_ROUNDS")
    )
    if depth_from_env:
        selected_research_depth = DEFAULT_CONFIG["max_debate_rounds"]
        console.print(
            f"[green]✓ Research depth from environment:[/green] "
            f"{DEFAULT_CONFIG['max_debate_rounds']} debate / "
            f"{DEFAULT_CONFIG['max_risk_discuss_rounds']} risk rounds"
        )
    else:
        console.print(
            create_question_box(
                "Step 5: Research Depth", "Select your research depth level"
            )
        )
        selected_research_depth = select_research_depth(prefs.get("research_depth"))

    # 4C. 模型连接：服务商、接口地址和 API key；不同服务商可能需要选择区域。
    # 步骤六：选择模型服务商；设置对应环境变量时跳过问答。
    # 优先采用 TRADINGAGENTS_LLM_BACKEND_URL 指定的接口地址；
    # 未指定时使用该服务商的默认地址，
    # 与通过菜单选择时的默认值保持一致。
    provider_from_env = bool(os.environ.get("TRADINGAGENTS_LLM_PROVIDER"))
    if provider_from_env:
        selected_llm_provider = DEFAULT_CONFIG["llm_provider"].lower()
        backend_url = resolve_backend_url(
            selected_llm_provider, env_url=DEFAULT_CONFIG["backend_url"]
        )
        console.print(f"[green]✓ LLM provider from environment:[/green] {selected_llm_provider}")
        console.print(f"[green]✓ Backend URL:[/green] {backend_url}")
        # 仍需检查并保存接口密钥，避免后续调用失败。
        ensure_api_key(selected_llm_provider)
    else:
        console.print(
            create_question_box(
                "Step 6: LLM Provider", "Select your LLM provider"
            )
        )
        selected_llm_provider, backend_url = select_llm_provider(prefs.get("llm_provider"))

        # 对提供区域接口的服务商，额外询问使用区域，
        # 以保持主菜单简洁；中国大陆与国际区域的账户
        # 不能共用接口密钥。
        if selected_llm_provider == "qwen":
            selected_llm_provider, backend_url = ask_qwen_region()
        elif selected_llm_provider == "minimax":
            selected_llm_provider, backend_url = ask_minimax_region()
        elif selected_llm_provider == "glm":
            selected_llm_provider, backend_url = ask_glm_region()

        # 即使服务商来自交互选择，也优先使用环境变量明确指定的接口地址，
        # 避免被菜单默认值覆盖（问题 #978）。
        backend_url = resolve_backend_url(
            selected_llm_provider, backend_url, env_url=DEFAULT_CONFIG["backend_url"]
        )

        # 通用 OpenAI 兼容接口没有默认地址；
        # 菜单和环境变量都未提供地址时，再向用户询问。
        if selected_llm_provider == "openai_compatible" and not backend_url:
            remembered_url = (prefs.get("backend_url")
                              if prefs.get("llm_provider") == selected_llm_provider else None)
            backend_url = prompt_openai_compatible_url(remembered_url)

        # 使用 Ollama 时，在选择模型前展示解析后的接口地址，
        # 让用户明确当前连接的是环境变量指定地址还是默认地址。
        if selected_llm_provider == "ollama":
            confirm_ollama_endpoint(backend_url)

        # 检查模型服务商的接口密钥是否存在；缺失时提示输入，
        # 并将其保存到 .env，
        # 避免首次调用模型接口时才因缺少密钥失败。
        ensure_api_key(selected_llm_provider)

    # 4D. 模型分工：quick/deep 两套模型，以及服务商特有的推理强度参数。
    # 步骤七：选择模型；任一模型已由环境变量指定时跳过此处问答。
    if os.environ.get("TRADINGAGENTS_QUICK_THINK_LLM") or os.environ.get("TRADINGAGENTS_DEEP_THINK_LLM"):
        selected_shallow_thinker = DEFAULT_CONFIG["quick_think_llm"]
        selected_deep_thinker = DEFAULT_CONFIG["deep_think_llm"]
        console.print(
            f"[green]✓ Thinking agents from environment:[/green] "
            f"quick={selected_shallow_thinker}, deep={selected_deep_thinker}"
        )
    else:
        console.print(
            create_question_box(
                "Step 7: Thinking Agents", "Select your thinking agents for analysis"
            )
        )
        remembered = prefs if prefs.get("llm_provider") == selected_llm_provider else {}
        selected_shallow_thinker = select_shallow_thinking_agent(
            selected_llm_provider, remembered.get("quick_think_llm")
        )
        selected_deep_thinker = select_deep_thinking_agent(
            selected_llm_provider, remembered.get("deep_think_llm")
        )

    # 步骤八：设置模型服务商特有的推理参数。
    # 各参数均可通过对应的 TRADINGAGENTS_* 环境变量指定。
    # 参数已由环境变量指定，或服务商本身来自环境变量时，跳过相关问答，
    # 直接采用配置值，与前面步骤的环境变量优先规则一致。
    # None 表示使用服务商自身的默认值。
    thinking_level = None
    reasoning_effort = None
    anthropic_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_from_env:
        thinking_level = DEFAULT_CONFIG["google_thinking_level"]
        reasoning_effort = DEFAULT_CONFIG["openai_reasoning_effort"]
        anthropic_effort = DEFAULT_CONFIG["anthropic_effort"]
    elif provider_lower == "google":
        thinking_level = thinking_value_or_prompt(
            "TRADINGAGENTS_GOOGLE_THINKING_LEVEL", "google_thinking_level",
            "Gemini thinking mode", "Step 8: Thinking Mode",
            "Configure Gemini thinking mode", ask_gemini_thinking_config,
        )
    elif provider_lower == "openai":
        reasoning_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_OPENAI_REASONING_EFFORT", "openai_reasoning_effort",
            "Reasoning effort", "Step 8: Reasoning Effort",
            "Configure OpenAI reasoning effort level", ask_openai_reasoning_effort,
        )
    elif provider_lower == "anthropic":
        anthropic_effort = thinking_value_or_prompt(
            "TRADINGAGENTS_ANTHROPIC_EFFORT", "anthropic_effort",
            "Claude effort", "Step 8: Effort Level",
            "Configure Claude effort level", ask_anthropic_effort,
        )

    # 4E. 将菜单答案统一打包；后续代码通过这些键读取，不再直接访问菜单控件。
    return {
        "ticker": selected_ticker,
        "asset_type": asset_type.value,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "quick_think_llm": selected_shallow_thinker,
        "deep_think_llm": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "anthropic_effort": anthropic_effort,
        "output_language": output_language,
    }


def get_analysis_date():
    """读取用户输入的分析日期。"""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # 验证日期格式，并确保分析日期不晚于今天。
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


# ==================== 5. 完整报告：写入文件与终端展示 ====================
# 这组函数消费 final_state；文件组织复用 reporting.write_report_tree()。
# display_complete_report() 则按团队展示报告及辩论内容。
def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """将完整分析报告写入磁盘，复用命令行与程序接口共用的报告写入器。"""
    return write_report_tree(final_state, ticker, save_path)


def display_complete_report(final_state):
    """按顺序展示完整分析报告，避免内容被截断。"""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # 一、分析师团队报告。
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan"))
        for title, content in analysts:
            console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # 二、研究团队报告。
    if final_state.get("investment_debate_state"):
        debate = final_state["investment_debate_state"]
        research = []
        if debate.get("bull_history"):
            research.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research.append(("Research Manager", debate["judge_decision"]))
        if research:
            console.print(Panel("[bold]II. Research Team Decision[/bold]", border_style="magenta"))
            for title, content in research:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

    # 三、交易团队方案。
    if final_state.get("trader_investment_plan"):
        console.print(Panel("[bold]III. Trading Team Plan[/bold]", border_style="yellow"))
        console.print(Panel(Markdown(final_state["trader_investment_plan"]), title="Trader", border_style="blue", padding=(1, 2)))

    # 四、风险管理团队讨论。
    if final_state.get("risk_debate_state"):
        risk = final_state["risk_debate_state"]
        risk_reports = []
        if risk.get("aggressive_history"):
            risk_reports.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_reports.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_reports.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_reports:
            console.print(Panel("[bold]IV. Risk Management Team Decision[/bold]", border_style="red"))
            for title, content in risk_reports:
                console.print(Panel(Markdown(content), title=title, border_style="blue", padding=(1, 2)))

        # 五、投资组合经理决策。
        if risk.get("judge_decision"):
            console.print(Panel("[bold]V. Portfolio Manager Decision[/bold]", border_style="green"))
            console.print(Panel(Markdown(risk["judge_decision"]), title="Portfolio Manager", border_style="blue", padding=(1, 2)))


# ==================== 6. 将图状态转换为界面信息 ====================
# update_*_statuses() 根据已有报告推断显示进度；实际执行顺序由 LangGraph 决定。
# extract_content_string()/classify_message_type() 将不同消息格式转为可展示文本。
def update_research_team_status(status):
    """统一更新研究团队成员的状态，不包含交易员。"""
    research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
    for agent in research_team:
        message_buffer.update_agent_status(agent, status)


# 用于界面状态切换的分析师固定顺序。
ANALYST_ORDER = ["market", "social", "news", "fundamentals"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Sentiment Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}


def update_analyst_statuses(message_buffer, chunk, wall_time_tracker=None):
    """根据累计报告内容更新分析师的界面状态。

    处理规则：
    - 保存当前状态中已有的新报告内容。
    - 根据缓冲区累计的报告章节判断进度。
    - 已有报告的分析师标记为完成。
    - 首个尚无报告的分析师标记为进行中。
    - 其余尚无报告的分析师标记为等待。
    - 全部分析师完成后，将看多研究员标记为进行中。"""
    selected = message_buffer.selected_analysts
    found_active = False

    if wall_time_tracker is not None:
        sync_analyst_tracker_from_chunk(wall_time_tracker, chunk)

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]

        # 从当前状态中读取新报告内容。
        if chunk.get(report_key):
            message_buffer.update_report_section(report_key, chunk[report_key])

        # 根据缓冲区累计的报告判断进度，而非仅依赖当前状态。
        has_report = bool(message_buffer.report_sections.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
        elif not found_active:
            message_buffer.update_agent_status(agent_name, "in_progress")
            found_active = True
        else:
            message_buffer.update_agent_status(agent_name, "pending")

    # 全部分析师完成后，将看多研究员标记为进行中。
    if (
        not found_active
        and selected
        and message_buffer.agent_status.get("Bull Researcher") == "pending"
    ):
        message_buffer.update_agent_status("Bull Researcher", "in_progress")

def extract_content_string(content):
    """从不同消息格式中提取文本。
    没有有效文本内容时返回 None。"""
    def is_empty(val):
        """判断内容是否为空、是否没有可展示信息。

        对文本检查是否实际包含字符，不把它解释成 Python 字面量。
        报告中的字符串 "0" 或 "None" 也是有效内容，不应被当成空值丢弃。"""
        if isinstance(val, str):
            return not val.strip()
        return val is None or not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """将消息归类为展示类型，并提取文本内容。

    返回：
        (type, content)：类型标记为 User（用户）、Agent（智能体）、
        Data（工具数据）、Control（控制）或 System（兜底系统消息）；
        内容为提取出的字符串，没有有效文本时为 None。"""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # 未知消息类型统一按系统消息展示。
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """将工具参数格式化为适合终端展示的文本。"""
    result = str(args)
    if len(result) > max_length:
        return result[:max_length - 3] + "..."
    return result

# ==================== 7. 运行准备：输出路径、恢复提示、配置合并 ====================
# 第一遍重点读 _build_run_config()：它连接“用户选择”和 TradingAgentsGraph。
def _run_directory(config: dict, ticker: str, trade_date: str) -> Path:
    """生成本次运行的输出目录，并校验标的代码是否可用作路径组成部分。

    与其他使用标的代码拼接路径的位置一样，先进行校验，
    避免 ".." 之类的值使输出落到结果目录之外。"""
    return Path(config["results_dir"]) / safe_ticker_component(ticker) / trade_date


def _announce_checkpoint_state(graph, ticker: str, trade_date: str) -> None:
    """在用户可见的消息区域提示本次是恢复运行还是重新开始。

    图内部虽有日志，但命令行未在此配置日志展示，实时界面也会占据屏幕，
    因此需要单独显示恢复状态。"""
    if getattr(graph, "_resuming", False):
        message_buffer.add_message(
            "System", f"Resuming the saved run for {ticker} on {trade_date}"
        )
    else:
        message_buffer.add_message("System", f"Starting fresh for {ticker} on {trade_date}")


def _build_run_config(selections: dict, checkpoint: bool | None) -> dict:
    """根据交互选择组装运行配置，并遵循环境变量优先规则。

    辩论轮数保留环境变量明确指定的值；检查点设置仅在显式传入
    命令行开关时覆盖 DEFAULT_CONFIG 中已经应用环境变量的配置。"""
    # DEFAULT_CONFIG 已应用环境变量；复制后再填入本次选择，保留明确指定的轮数。
    # checkpoint 的 None 表示未传 CLI 开关，只有显式 True/False 才覆盖配置。
    config = DEFAULT_CONFIG.copy()
    # 研究深度通常同时设置两类辩论轮数，但明确的环境变量配置优先。
    # 分别检查 TRADINGAGENTS_MAX_DEBATE_ROUNDS 与 TRADINGAGENTS_MAX_RISK_ROUNDS，
    # 保留已由环境变量覆盖的默认配置值（问题 #977）。
    for env_var, key in (("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "max_debate_rounds"),
                         ("TRADINGAGENTS_MAX_RISK_ROUNDS", "max_risk_discuss_rounds")):
        if os.environ.get(env_var):
            # 仅设置一个轮数时仍会询问研究深度，
            # 因此需要说明哪一项实际采用环境变量值。
            console.print(
                f"[green]✓ {key} from environment:[/green] {config[key]} "
                f"(set by {env_var}, so the research depth you chose does not apply to it)"
            )
        else:
            config[key] = selections["research_depth"]
    config["quick_think_llm"] = selections["quick_think_llm"]
    config["deep_think_llm"] = selections["deep_think_llm"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # 填入模型服务商特有的推理参数。
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["anthropic_effort"] = selections.get("anthropic_effort")
    config["output_language"] = selections.get("output_language", "English")
    # 只有显式传入 --checkpoint/--no-checkpoint 才覆盖检查点设置；
    # 省略开关时保留环境变量或默认配置中的值（问题 #976）。
    if checkpoint is not None:
        config["checkpoint_enabled"] = checkpoint
    return config


# ==================== 8. 单次分析主线：优先精读这个函数 ====================
# 调用链：analyze() → run_analysis() → graph.graph.stream(...)。
# CLI 直接消费图的流式状态，以便每步更新屏幕；这里没有调用 graph.propagate()。
# 因此本函数也负责初始状态、检查点生命周期和结束后的决策记录。
def run_analysis(checkpoint: bool | None = None, portfolio=None):
    # A. 收集设置并创建执行组件：配置、调用统计、分析师顺序与耗时追踪器。
    # 首先收集本次运行的全部用户选择。
    selections = get_user_selections()

    config = _build_run_config(selections, checkpoint)

    # 创建统计回调，追踪模型与工具调用。
    stats_handler = StatsCallbackHandler()

    # 将用户选择视为集合，再按预定义顺序排列分析师。
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]
    analyst_execution_plan = build_analyst_execution_plan(selected_analyst_keys)
    analyst_wall_time_tracker = AnalystWallTimeTracker(analyst_execution_plan)

    # B. 创建分析引擎，并重置界面缓冲区；真正的 Agent 节点与边在 graph 模块构建。
    # 初始化图，并将统计回调绑定到模型。
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # 按所选分析师初始化界面缓冲区。
    message_buffer.init_for_analysis(selected_analyst_keys)

    # 记录开始时间，用于显示运行时长。
    start_time = time.time()

    # C. 准备过程文件：消息/工具日志，以及随分析更新的各章节 Markdown。
    # 下方三个包装函数在原来的缓冲区更新后追加写盘行为，@wraps 保留函数元信息。
    # 创建本次运行的结果目录。
    results_dir = _run_directory(config, selections["ticker"], selections["analysis_date"])
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # 将换行替换为空格
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")
        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")
        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)
        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if section_name in obj.report_sections and obj.report_sections[section_name] is not None:
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    text = "\n".join(str(item) for item in content) if isinstance(content, list) else content
                    with open(report_dir / file_name, "w", encoding="utf-8") as f:
                        f.write(text)
        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(message_buffer, "add_tool_call")
    message_buffer.update_report_section = save_report_section_decorator(message_buffer, "update_report_section")

    # D. 开启 Rich 实时界面，显示本次设置，并将第一个分析师标为进行中。
    # 创建实时展示布局。
    layout = create_layout()

    # 使用终端备用屏幕，避免布局高于窗口时反复滚动重绘；
    # 退出实时展示后，再在普通屏幕输出完整报告。
    with Live(layout, refresh_per_second=4, screen=True):
        # 绘制初始界面。
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # 记录本次分析的初始信息。
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        if selections["asset_type"] != "stock":
            message_buffer.add_message("System", f"Detected asset type: {selections['asset_type']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # 将第一个分析师标记为进行中。
        first_analyst = get_initial_analyst_node(analyst_execution_plan)
        message_buffer.update_agent_status(first_analyst, "in_progress")
        analyst_wall_time_tracker.mark_started(selected_analyst_keys[0])
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # 生成加载动画旁的提示文字。
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(layout, spinner_text, stats_handler=stats_handler, start_time=start_time)

        # E. 构造图的初始状态与执行参数：标的信息、历史经验、持仓及统计回调。
        # 如果启用检查点，则带上 thread_id；恢复时 checkpoint_input() 返回 None。
        # 与 propagate() 使用相同的初始状态构造逻辑：
        # 结算历史决策，并注入历史背景与解析后的标的身份。
        init_agent_state = graph.create_run_state(
            selections["ticker"], selections["analysis_date"], selections["asset_type"], portfolio
        )
        # 通过图执行配置传递回调，以统计工具调用；
        # 模型调用统计已通过模型构造器单独绑定。
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # 使用检查点存储重新编译图并注入 thread_id，
        # 让命令行路径支持保存与恢复；关闭检查点时不执行这些操作（问题 #1249）。
        # 检查点资源在下方 finally 中释放。
        checkpoint_tid = graph.begin_checkpoint(
            selections["ticker"], selections["analysis_date"], selections["asset_type"], portfolio
        )
        if checkpoint_tid is not None:
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = checkpoint_tid

        # 以流式方式执行分析；恢复时传入 None，
        # 让图继续中断的运行，而非再次追加初始状态（问题 #1249）。
        # 使用 try/finally 确保流式执行抛出异常时仍能释放检查点资源。
        # F. 核心循环：每收到一次图状态，就处理消息、同步阶段报告并刷新屏幕。
        # get_graph_args() 当前使用 stream_mode="values"，chunk 是该步的状态快照；
        # 其中可能带着此前的消息和报告，所以消息需要去重，字段存在也不代表刚刚更新。
        trace = []
        try:
            for chunk in graph.graph.stream(graph.checkpoint_input(init_agent_state), **args):
                # F1. 提取模型/用户/工具消息，按消息 ID 去重，并记录模型提出的工具调用。
                # 处理状态中的消息，并通过消息标识去重。
                for message in chunk.get("messages", []):
                    msg_id = getattr(message, "id", None)
                    if msg_id is not None:
                        if msg_id in message_buffer._processed_message_ids:
                            continue
                        message_buffer._processed_message_ids.add(msg_id)

                    msg_type, content = classify_message_type(message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(tool_call["name"], tool_call["args"])
                            else:
                                message_buffer.add_tool_call(tool_call.name, tool_call.args)

                # F2. 分析师阶段：将 *_report 写入展示缓冲区，更新完成状态与耗时。
                # 每次收到状态时，根据报告内容同步分析师进度。
                update_analyst_statuses(
                    message_buffer,
                    chunk,
                    wall_time_tracker=analyst_wall_time_tracker,
                )

                # F3. 研究阶段：展示 Bull/Bear 辩论；judge_decision 出现后展示投资计划。
                # 读取并展示多空辩论状态。
                if chunk.get("investment_debate_state"):
                    debate_state = chunk["investment_debate_state"]
                    bull_hist = debate_state.get("bull_history", "").strip()
                    bear_hist = debate_state.get("bear_history", "").strip()
                    judge = debate_state.get("judge_decision", "").strip()

                    # 仅在已有实际发言内容时更新团队状态。
                    if bull_hist or bear_hist:
                        update_research_team_status("in_progress")
                    if bull_hist:
                        message_buffer.update_report_section(
                            "investment_plan", f"### Bull Researcher Analysis\n{bull_hist}"
                        )
                    if bear_hist:
                        message_buffer.update_report_section(
                            "investment_plan", f"### Bear Researcher Analysis\n{bear_hist}"
                        )
                    if judge:
                        message_buffer.update_report_section(
                            "investment_plan", f"### Research Manager Decision\n{judge}"
                        )
                        update_research_team_status("completed")
                        message_buffer.update_agent_status("Trader", "in_progress")

                # F4. 交易方案阶段：展示 Trader 的方案，界面转入风险评估阶段。
                # 展示交易团队方案。
                if chunk.get("trader_investment_plan"):
                    message_buffer.update_report_section(
                        "trader_investment_plan", chunk["trader_investment_plan"]
                    )
                    if message_buffer.agent_status.get("Trader") != "completed":
                        message_buffer.update_agent_status("Trader", "completed")
                        message_buffer.update_agent_status("Aggressive Analyst", "in_progress")

                # F5. 风险阶段：展示三方辩论，再由 Portfolio Manager 的裁决完成该阶段。
                # 读取并展示风险辩论状态。
                if chunk.get("risk_debate_state"):
                    risk_state = chunk["risk_debate_state"]
                    agg_hist = risk_state.get("aggressive_history", "").strip()
                    con_hist = risk_state.get("conservative_history", "").strip()
                    neu_hist = risk_state.get("neutral_history", "").strip()
                    judge = risk_state.get("judge_decision", "").strip()

                    if agg_hist:
                        if message_buffer.agent_status.get("Aggressive Analyst") != "completed":
                            message_buffer.update_agent_status("Aggressive Analyst", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}"
                        )
                    if con_hist:
                        if message_buffer.agent_status.get("Conservative Analyst") != "completed":
                            message_buffer.update_agent_status("Conservative Analyst", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}"
                        )
                    if neu_hist:
                        if message_buffer.agent_status.get("Neutral Analyst") != "completed":
                            message_buffer.update_agent_status("Neutral Analyst", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}"
                        )
                    if judge and message_buffer.agent_status.get("Portfolio Manager") != "completed":
                        message_buffer.update_agent_status("Portfolio Manager", "in_progress")
                        message_buffer.update_report_section(
                            "final_trade_decision", f"### Portfolio Manager Decision\n{judge}"
                        )
                        message_buffer.update_agent_status("Aggressive Analyst", "completed")
                        message_buffer.update_agent_status("Conservative Analyst", "completed")
                        message_buffer.update_agent_status("Neutral Analyst", "completed")
                        message_buffer.update_agent_status("Portfolio Manager", "completed")

                # 刷新界面。
                update_display(layout, stats_handler=stats_handler, start_time=start_time)

                trace.append(chunk)

            # G. 汇总流出的状态，记录最终决策；仅成功完成时清除本次检查点。
            # 当前 values 模式给出状态快照；这里依次 update，让后续字段覆盖此前值。
            final_state = {}
            for chunk in trace:
                final_state.update(chunk)

            # 正常完成时先记录决策，再删除本次检查点，
            # 使后续运行可以重新开始；中途失败会跳过这两步，
            # 保留检查点以便恢复。
            graph.record_decision(selections["ticker"], selections["analysis_date"], final_state)
            graph.clear_checkpoint_on_success(
                selections["ticker"], selections["analysis_date"], selections["asset_type"], portfolio
            )
        finally:
            # 释放检查点资源并恢复普通图；发生异常时，已保存的检查点仍供后续恢复。
            # 无论是否发生异常，都恢复为不带检查点的普通图。
            graph.end_checkpoint()

        # 将所有智能体的界面状态标记为完成。
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )
        message_buffer.add_message("System", analyst_wall_time_tracker.format_summary())

        # 用最终状态更新各报告章节。
        for section in message_buffer.report_sections:
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # H. 退出实时界面后进行收尾交互：提示评级解析问题、询问完整报告保存位置和展示方式。
    # 这里保存的是用户选择的完整报告副本；C 阶段的过程日志和章节文件已随运行写入。
    # 退出实时展示后再进行问答，避免界面刷新干扰输入。
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    # 无法解析评级的结果需要人工复核，
    # 在此明确提示，避免被当成正常持仓决策。
    if is_review(graph.process_signal(final_state.get("final_trade_decision", ""))):
        console.print(
            "[yellow]No rating could be read from the final decision, so this run "
            "is recorded for review rather than as a position. Re-run, or read the "
            "decision text below and judge it yourself.[/yellow]\n"
        )
    console.print(f"[dim]{analyst_wall_time_tracker.format_summary()}[/dim]")

    # 询问是否保存完整报告。
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 默认保存到 results_dir 下，而非当前工作目录。
        # 在容器中，工作目录中的报告可能随容器删除而丢失；
        # results_dir 则可对应挂载的数据卷，并与其他运行输出保持一致。
        default_path = (Path(config["results_dir"]) / "reports"
                        / f"{safe_ticker_component(selections['ticker'])}_{timestamp}")
        save_path_str = typer.prompt(
            "Save path (press Enter for default)",
            default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(final_state, selections["ticker"], save_path)
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # 询问是否在屏幕上展示完整报告。
    display_choice = typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)


# ==================== 9. 默认命令入口：从这里开始追调用 ====================
# Typer 将命令行选项注入参数；不带子命令时运行 analyze() 的分析逻辑。
# 若指定 backtest 子命令，回调通过 invoked_subcommand 判断并直接返回。
# 本层处理检查点清理、持仓文件和终端错误，然后把分析工作交给 run_analysis()。
@app.callback(invoke_without_command=True) #把下面的 analyze 函数注册为命令行应用的主回调，并允许不带子命令时执行它
def analyze(
    ctx: typer.Context,  # 命令上下文：Typer 根据类型注解自动传入，用户无需在终端提供。
    checkpoint: bool | None = typer.Option(  # 定义命令行选项规则；解析后参数可为 True、False 或 None。
        None,  # 未传该选项时为 None，表示沿用配置；不等于明确关闭。
        "--checkpoint/--no-checkpoint",  # 两个开关分别传入 True（开启恢复）和 False（关闭恢复）。
        help="Enable/disable checkpoint-resume (save state after each node so a "  # help 是 --help 中显示的说明。
        "crashed run can resume). Omit to honor TRADINGAGENTS_CHECKPOINT_ENABLED.",  # 相邻字符串会自动拼接。
    ),
    clear_checkpoints: bool = typer.Option(  # 是否删除已有检查点；与本次是否启用恢复是两个独立选项。
        False,  # 默认不删除已保存的执行状态。
        "--clear-checkpoints",  # 命令中出现此开关时传入 True，无需在后面再写 True。
        help="Delete all saved checkpoints before running (force fresh start).",  # 帮助说明：清除检查点后重新开始。
    ),
    portfolio: str = typer.Option(  # 持仓文件路径；实际也允许默认的 None，类型写成 str | None 更准确。
        None,  # 未提供路径时为 None，表示没有传入持仓文件。
        "--portfolio",  # 此选项需要跟路径，例如：tradingagents --portfolio holdings.json。
        help="JSON file with current holdings and cash, so the trader, risk and "  # 文件包含持仓和现金信息。
        "portfolio agents size against your actual position.",  # 此处只声明选项；文件由后面的 load_portfolio() 读取。
    ),
):
    """运行单次分析；直接执行 tradingagents 且不带子命令时进入此流程。"""
    if ctx.invoked_subcommand is not None:
        return
    if clear_checkpoints:
        from tradingagents.graph.checkpointer import clear_all_checkpoints
        n = clear_all_checkpoints(DEFAULT_CONFIG["data_cache_dir"])
        console.print(f"[yellow]Cleared {n} checkpoint(s).[/yellow]")
    portfolio_context = None
    if portfolio:
        from tradingagents.portfolio import load_portfolio
        try:
            portfolio_context = load_portfolio(portfolio)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

    try:
        run_analysis(checkpoint=checkpoint, portfolio=portfolio_context)
    except _NO_CONSOLE_ERRORS:
        # 没有控制台缓冲区的终端无法承载交互问答。
        # 向标准错误输出一条可操作的提示，避免展示依赖库的完整异常堆栈。
        # 此处使用纯文本，因为富文本渲染也可能不可用（问题 #1138）。
        typer.echo(
            "Error: no Windows console available. The interactive CLI needs a real "
            "console buffer — run it from Windows Terminal, PowerShell, or cmd.exe "
            "rather than a piped or embedded terminal.",
            err=True,
        )
        raise typer.Exit(code=1) from None


# ==================== 10. 回测子命令：批量运行并评估历史决策 ====================
# 输入多个标的和日期范围 → 构造日期网格 → run_backtest() → 展示统计与失败项。
# 第一遍理解单次分析时可跳过；具体评估逻辑在 tradingagents/backtest.py。
@app.command()
def backtest(
    tickers: str = typer.Argument(..., help="Comma-separated tickers, e.g. NVDA,AAPL"),
    start: str = typer.Option(..., "--start", help="First analysis date, YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="Last analysis date, YYYY-MM-DD"),
    every: int = typer.Option(7, "--every", help="Days between analysis dates"),
    analysts: str = typer.Option(
        None, "--analysts", help="Comma-separated analysts to run; omit for all four"
    ),
    asset_type: str = typer.Option("stock", "--asset-type", help="stock or crypto"),
    portfolio: str = typer.Option(
        None, "--portfolio", help="JSON file with holdings and cash, held constant across the grid"
    ),
    run_id: str = typer.Option(
        None, "--run-id", help="Continue an earlier sweep: its cells are skipped and its log reused"
    ),
):
    """在标的与日期组成的网格上评估历史决策。"""
    from tradingagents.agents.utils.memory import TradingMemoryLog

    try:
        dates = iter_grid(start, end, every)
        book = load_portfolio(portfolio) if portfolio else None
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    names = [t.strip() for t in tickers.split(",") if t.strip()]
    if not names:
        console.print("[red]No ticker to analyze; pass them comma-separated, e.g. NVDA,AAPL[/red]")
        raise typer.Exit(code=1)

    kwargs = {"asset_type": asset_type, "portfolio": book, "run_id": run_id}
    if analysts:
        kwargs["selected_analysts"] = [a.strip().lower() for a in analysts.split(",") if a.strip()]

    try:
        result = run_backtest(names, dates, DEFAULT_CONFIG, **kwargs)
    except Exception as exc:  # 缺少密钥或分析师名称未知属于配置错误
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    console.print(summarize(TradingMemoryLog({"memory_log_path": str(result.log_path)})).render())
    console.print(f"\nRan {result.cells_run} cells, skipped {result.skipped}. Log: {result.log_path}")
    for ticker, date, reason in result.failures:
        console.print(f"[yellow]failed:[/yellow] {ticker} {date}: {reason}")
    for ticker, reason in result.settlement_failures:
        console.print(f"[yellow]unsettled:[/yellow] {ticker}: {reason}")


# 直接运行本模块时启动 Typer，由它解析参数并分发到上述回调或子命令。
if __name__ == "__main__":
    app()
