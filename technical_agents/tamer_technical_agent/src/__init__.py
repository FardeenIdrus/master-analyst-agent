"""
Tamer Technical Agent - Technical Analysis Pipeline

LLM Priority:
- Tries Anthropic first (if ANTHROPIC_API_KEY is set)
- Falls back to OpenAI (if OPENAI_API_KEY is set)
"""

from .llm_agent import (
    generate_trade_note,
    TradeNote,
    CLAUDE_MODEL,
    OPENAI_MODEL,
    VERSION,
)
from .data_collector import DataPipeline, PipelineOutput
from .report_generator import generate_all_reports, generate_json_report
from .trade_note_reports import TradeNoteReportGenerator, PremiumHTMLFormatter

__all__ = [
    'generate_trade_note',
    'TradeNote',
    'CLAUDE_MODEL',
    'OPENAI_MODEL',
    'VERSION',
    'DataPipeline',
    'PipelineOutput',
    'generate_all_reports',
    'generate_json_report',
    'TradeNoteReportGenerator',
    'PremiumHTMLFormatter',
]
