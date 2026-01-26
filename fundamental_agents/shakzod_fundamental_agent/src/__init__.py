"""
Fundamental Analyst Agent - Source Package

MSc Coursework: AI Agents in Asset Management
Track A: Fundamental Analyst Agent

This package contains all modules for the fundamental analysis pipeline:
    Step 0: config             - Configuration and constants
    Step 1: data_collector     - Data ingestion from Yahoo Finance
    Step 2: data_processor     - Data cleaning and standardization
    Step 3: profitability_analyzer - Profitability bridge analysis
    Step 4: cash_flow_analyzer - Cash flow bridge analysis
    Step 5: earnings_quality_analyzer - Earnings quality assessment
    Step 6: working_capital_analyzer - Working capital analysis
    Step 7: ratio_calculator   - Financial ratio calculations
    Step 8: valuation          - Valuation multiples analysis
    Step 9: memo_generator     - LLM-powered investment memo generation
    Step 10: validation        - Comprehensive accuracy and logic verification
    Step 11: agent             - Main orchestration class

Usage:
    from src import FundamentalAnalystAgent
    
    agent = FundamentalAnalystAgent("AAPL")
    results = agent.run_analysis()
    agent.export_results("outputs/AAPL")
    agent.print_summary()
"""

# =============================================================================
# VERSION INFORMATION
# =============================================================================

__version__ = "1.1.0"
__author__ = "MSc AI Agents in Asset Management"

# =============================================================================
# STEP 0: CONFIGURATION
# =============================================================================

from .config import (
    # Threshold Classes
    VALIDATION,
    EARNINGS_QUALITY,
    VALUATION_PARAMS,
    OUTPUT,
    
    # Enumerations
    ValidationStatus,
    AnalysisStep,
    EarningsQualityRating,
    ValuationAssessment,
    
    # Helper Functions
    get_field_alternatives,
    get_all_required_fields
)

# =============================================================================
# STEP 1: DATA COLLECTION
# =============================================================================

from .data_collector import (
    DataCollector,
    CollectedData,
    CompanyInfo,
    FinancialStatements,
    DataValidationResult,
    collect_financial_data,
    validate_ticker_data
)

# =============================================================================
# STEP 2: DATA PROCESSING
# =============================================================================

from .data_processor import (
    DataProcessor,
    ProcessedData,
    ProcessedStatements,
    DataQualityMetrics,
    TransformationRecord,
    StandardField,
    process_financial_data,
    get_standard_field_value
)

# =============================================================================
# STEP 3: PROFITABILITY ANALYSIS
# =============================================================================

from .profitability_analyzer import (
    ProfitabilityAnalyzer,
    ProfitabilityAnalysisResult,
    ProfitabilityBridge,
    ProfitabilityMetrics,
    BridgeComponent,
    MarginAnalysis,
    ProfitDriver,
    MarginType,
    analyze_profitability,
    get_ebit_bridge_summary
)

# =============================================================================
# STEP 4: CASH FLOW ANALYSIS
# =============================================================================

from .cash_flow_analyzer import (
    CashFlowAnalyzer,
    CashFlowAnalysisResult,
    CashFlowBridge,
    CashFlowMetrics,
    CashFlowComponent,
    CashConversionQuality,
    analyze_cash_flow,
    get_cash_flow_summary
)

# =============================================================================
# STEP 5: EARNINGS QUALITY ANALYSIS
# =============================================================================

from .earnings_quality_analyzer import (
    EarningsQualityAnalyzer,
    EarningsQualityResult,
    AccrualsAnalysis,
    GrowthDivergence,
    RedFlag,
    RedFlagSeverity,
    RedFlagCategory,
    QualityScore,
    analyze_earnings_quality,
    get_earnings_quality_summary
)

# =============================================================================
# STEP 6: WORKING CAPITAL ANALYSIS
# =============================================================================

from .working_capital_analyzer import (
    WorkingCapitalAnalyzer,
    WorkingCapitalAnalysisResult,
    EfficiencyMetrics,
    CashConversionCycle,
    WorkingCapitalPosition,
    WorkingCapitalAlert,
    AlertType,
    AlertSeverity,
    TrendDirection,
    analyze_working_capital,
    get_working_capital_summary
)

# =============================================================================
# STEP 7: FINANCIAL RATIOS
# =============================================================================

from .ratio_calculator import (
    RatioCalculator,
    FinancialRatiosResult,
    RatioValue,
    RatioCategory,
    RatioInterpretation,
    DuPontAnalysis,
    calculate_all_ratios,
    get_ratio_summary
)

# =============================================================================
# STEP 8: VALUATION ANALYSIS
# =============================================================================

from .valuation import (
    ValuationAnalyzer,
    ValuationAnalysisResult,
    ValuationMultiple,
    MultipleQuality,
    ImpliedGrowthAnalysis,
    analyze_valuation,
    get_valuation_summary
)

# =============================================================================
# STEP 9: MEMO GENERATION
# =============================================================================

from .memo_generator import (
    MemoGenerator,
    InvestmentMemo,
    InvestmentRecommendation,
    MemoSection,
    MemoGenerationMode,
    MemoSectionContent,
    DataCitation,
    AnalysisSummary,
    generate_investment_memo,
    get_recommendation_summary
)

# =============================================================================
# STEP 10: VALIDATION
# =============================================================================

from .validation import (
    # Main Classes
    ValidationEngine,
    ValidationReport,
    ValidationFinding,
    ValidationTolerances,
    
    # Enumerations
    ValidationSeverity,
    ValidationCategory,
    
    # Standalone Validation Functions
    validate_ebit_bridge,
    validate_fcf_calculation,
    validate_cash_conversion_cycle,
    validate_scoring_calculation,
    validate_margin_display,
    validate_valuation_unit_consistency,
    validate_report_accuracy,
    
    # Default Tolerances
    TOLERANCES
)

# =============================================================================
# STEP 11: AGENT ORCHESTRATION
# =============================================================================

from .agent import (
    FundamentalAnalystAgent,
    AnalysisResults,
    StepResult,
    AnalysisStatus,
    StepStatus,
    run_fundamental_analysis
)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Version
    '__version__',
    '__author__',
    
    # Configuration (Step 0)
    'VALIDATION',
    'EARNINGS_QUALITY',
    'VALUATION_PARAMS',
    'OUTPUT',
    'ValidationStatus',
    'AnalysisStep',
    'EarningsQualityRating',
    'ValuationAssessment',
    'get_field_alternatives',
    'get_all_required_fields',
    
    # Data Collection (Step 1)
    'DataCollector',
    'CollectedData',
    'CompanyInfo',
    'FinancialStatements',
    'DataValidationResult',
    'collect_financial_data',
    'validate_ticker_data',
    
    # Data Processing (Step 2)
    'DataProcessor',
    'ProcessedData',
    'ProcessedStatements',
    'DataQualityMetrics',
    'TransformationRecord',
    'StandardField',
    'process_financial_data',
    'get_standard_field_value',
    
    # Profitability Analysis (Step 3)
    'ProfitabilityAnalyzer',
    'ProfitabilityAnalysisResult',
    'ProfitabilityBridge',
    'ProfitabilityMetrics',
    'BridgeComponent',
    'MarginAnalysis',
    'ProfitDriver',
    'MarginType',
    'analyze_profitability',
    'get_ebit_bridge_summary',
    
    # Cash Flow Analysis (Step 4)
    'CashFlowAnalyzer',
    'CashFlowAnalysisResult',
    'CashFlowBridge',
    'CashFlowMetrics',
    'CashFlowComponent',
    'CashConversionQuality',
    'analyze_cash_flow',
    'get_cash_flow_summary',
    
    # Earnings Quality Analysis (Step 5)
    'EarningsQualityAnalyzer',
    'EarningsQualityResult',
    'AccrualsAnalysis',
    'GrowthDivergence',
    'RedFlag',
    'RedFlagSeverity',
    'RedFlagCategory',
    'QualityScore',
    'analyze_earnings_quality',
    'get_earnings_quality_summary',
    
    # Working Capital Analysis (Step 6)
    'WorkingCapitalAnalyzer',
    'WorkingCapitalAnalysisResult',
    'EfficiencyMetrics',
    'CashConversionCycle',
    'WorkingCapitalPosition',
    'WorkingCapitalAlert',
    'AlertType',
    'AlertSeverity',
    'TrendDirection',
    'analyze_working_capital',
    'get_working_capital_summary',
    
    # Financial Ratios (Step 7)
    'RatioCalculator',
    'FinancialRatiosResult',
    'RatioValue',
    'RatioCategory',
    'RatioInterpretation',
    'DuPontAnalysis',
    'calculate_all_ratios',
    'get_ratio_summary',
    
    # Valuation Analysis (Step 8)
    'ValuationAnalyzer',
    'ValuationAnalysisResult',
    'ValuationMultiple',
    'MultipleQuality',
    'ImpliedGrowthAnalysis',
    'analyze_valuation',
    'get_valuation_summary',
    
    # Memo Generation (Step 9)
    'MemoGenerator',
    'InvestmentMemo',
    'InvestmentRecommendation',
    'MemoSection',
    'MemoGenerationMode',
    'MemoSectionContent',
    'DataCitation',
    'AnalysisSummary',
    'generate_investment_memo',
    'get_recommendation_summary',
    
    # Validation (Step 10)
    'ValidationEngine',
    'ValidationReport',
    'ValidationFinding',
    'ValidationTolerances',
    'ValidationSeverity',
    'ValidationCategory',
    'validate_ebit_bridge',
    'validate_fcf_calculation',
    'validate_cash_conversion_cycle',
    'validate_scoring_calculation',
    'validate_margin_display',
    'validate_valuation_unit_consistency',
    'validate_report_accuracy',
    'TOLERANCES',
    
    # Agent Orchestration (Step 11)
    'FundamentalAnalystAgent',
    'AnalysisResults',
    'StepResult',
    'AnalysisStatus',
    'StepStatus',
    'run_fundamental_analysis',
]