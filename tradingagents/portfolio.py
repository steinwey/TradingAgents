"""将调用方提供的持仓信息整理成决策智能体可读取的背景。

这是一次分析的可选输入，描述持有什么、平均买入价格以及可用现金。
必须区分三种情况：已有持仓、明确空仓、完全没有提供持仓信息。
不能将“未提供信息”解释成“空仓”，否则会凭空推断用户账户的情况。

本模块不连接券商或执行交易。数量采用通用单位，货币名称由调用方提供，
只负责数据校验、持仓查找、背景文本生成和持仓指纹计算。
"""

# 中文阅读导航：先看数据结构，再沿“读取文件 → 校验 → 查找 → 生成文本”理解流程。
#   ① Position（第 35 行）：单个标的的持仓字段。
#   ② PortfolioContext（第 44 行）：现金、货币和多条持仓组成的整体背景。
#   ③ load_portfolio()（第 91 行）：命令行传入文件路径后，从这里读取和校验。
#   ④ position_in()（第 50 行）→ render()（第 56 行）：找到当前标的，并生成智能体能阅读的说明。
#   ⑤ fingerprint()（第 81 行）：计算持仓指纹，供检查点恢复时区分不同的持仓输入。
# 调用关系：cli/main.py 读取文件 → 图初始化状态时调用 render() → 写入 portfolio_context。
# 注意：Python 对象 PortfolioContext 和图状态中的 portfolio_context 文本是不同形式的数据。

# ==================== 1. 依赖与类型注解 ====================
# 延迟求值类型注解；函数签名中的 str、Path、Position 等用来说明数据类型。
from __future__ import annotations

import hashlib  # 计算持仓内容的哈希摘要。
import json  # 将 JSON 文本解析为 Python 字典、列表等对象。
from pathlib import Path  # 统一处理文件路径和文件读取。

from pydantic import BaseModel, Field, ValidationError  # 数据模型、字段规则、数据校验异常。


# ==================== 2. 单条持仓：Position ====================
# 继承 BaseModel，让 Pydantic 按字段类型校验数据，并在允许的情况下转换类型。
# Field() 声明默认值和字段说明；description 是模型元数据，不会自动生成持仓内容。
# 没有指定默认值的 ticker、quantity 为必填字段；average_price 可以省略或为 None。
class Position(BaseModel):
    ticker: str = Field(description="Instrument symbol, e.g. AAPL")  # 标的代码，例如 AAPL。
    quantity: float = Field(description="Signed units held; negative is short")  # 持有数量；负数表示空头持仓。
    average_price: float | None = Field(default=None, description="Average entry price per unit")  # 单位平均入场价格；None 表示未提供。


# ==================== 3. 整体持仓背景：PortfolioContext ====================
# 一个对象包含多条 Position。现金和货币可省略；positions 未提供时默认为空列表。
# 持仓文件示例：{"cash": 10000, "currency": "USD", "positions": [{"ticker": "AAPL", "quantity": 10}]}
class PortfolioContext(BaseModel):
    cash: float | None = Field(default=None, description="Free cash available")  # 可用现金；None 表示未知，不是零。
    currency: str | None = Field(default=None, description="Currency label for cash and prices")  # 货币标签；本模块不做汇率换算。
    positions: list[Position] = Field(default_factory=list)  # 每次创建对象时调用 list()，生成各自独立的空列表。

    # 3A. 查找：self 是当前持仓对象；返回首条匹配持仓，找不到则返回 None。
    def position_in(self, ticker: str) -> Position | None:
        # 生成器逐条筛选：查询代码去除首尾空白，双方转大写后比较，因此不区分大小写。
        # next(生成器, None) 取首个匹配项；第二个参数指定没有匹配时的返回值。
        return next((p for p in self.positions if p.ticker.upper() == ticker.strip().upper()), None)

    # 3B. 展示：将结构化持仓对象转换为一段文本，作为提示词中的账户背景。
    def render(self, ticker: str) -> str:
        """生成供决策智能体阅读的持仓说明，优先描述当前分析的标的。"""
        symbol = ticker.strip().upper()  # 规范化本次查询的标的代码。
        held = self.position_in(symbol)  # 调用当前对象的方法查找持仓。
        # 先构造当前标的的说明。这里已有持仓背景对象，未找到标的才描述为没有该持仓。
        # 没有提供持仓背景时，调用方不会调用本方法，而是向图状态写入空文本。
        if held is None:
            lines = [f"- No current position in {symbol}"]
        else:
            # “值 if 条件 else 另一值”是条件表达式；未提供平均价格就省略价格说明。
            # f 字符串中 :,.2f 表示千位分隔并保留两位小数，:,.4g 表示千位分隔并使用四位有效数字。
            price = f", average price {held.average_price:,.2f}" if held.average_price is not None else ""
            lines = [f"- Current position in {symbol}: {held.quantity:,.4g} units{price}"]
        # 显式判断是否为 None，使现金为 0 时仍会展示；货币标签仅在提供时追加。
        if self.cash is not None:
            lines.append(f"- Cash available: {self.cash:,.2f}{' ' + self.currency if self.currency else ''}")
        # 列表推导式收集其余持仓；is not 比较对象身份，排除刚才找到的那个对象。
        others = [p for p in self.positions if p is not held]
        if others:
            # 用逗号连接其他持仓的代码和数量，形成一行补充信息。
            lines.append("- Other positions: " + ", ".join(f"{p.ticker.upper()} {p.quantity:,.4g}" for p in others))
        # 用换行符连接各行并添加标题；这里返回字符串，不写文件，也不直接调用模型。
        return "Portfolio at the analysis date:\n" + "\n".join(lines)

    # 3C. 指纹：把本次持仓输入纳入运行标识，避免持仓变化后继续使用旧运行的检查点。
    def fingerprint(self) -> str:
        """计算持仓内容的稳定摘要，用于区分检查点对应的持仓输入。"""
        # 模型转 JSON 字符串 → 编码为字节 → SHA-256 摘要 → 十六进制文本 → 截取前 12 位。
        # 它是内容标识，不是加密；持仓列表顺序也参与计算，这里没有对持仓排序。
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:12]


# ==================== 4. 文件入口：读取并校验持仓 ====================
# path: str | Path 表示接受路径字符串或 Path 对象；-> PortfolioContext 表示返回持仓对象。
# CLI 中的 load_portfolio(portfolio) 会进入这里；读取或校验失败时，分析尚未开始。
def load_portfolio(path: str | Path) -> PortfolioContext:
    """读取并校验持仓 JSON 文件，将错误提前到分析开始之前报告。"""
    try:
        # Path(path) 统一路径类型；read_text() 按 UTF-8 读取；json.loads() 将文本解析为 Python 数据。
        data = json.loads(Path(path).read_text(encoding="utf-8"))    #把传入的路径转换为一个Path路径对象
        # 按上面的字段规则校验数据，嵌套的持仓字典也会转换成 Position 对象。
        return PortfolioContext.model_validate(data)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        # 分别处理文件访问错误、JSON 语法错误、模型字段校验错误，并绑定到变量 exc。
        # 对外统一抛出 ValueError，便于 CLI 捕获；from exc 保留原始异常作为原因。
        raise ValueError(f"portfolio file {path} is not usable: {exc}") from exc
