"""
Fundamental Analyst Agent - Phases 1-4 Data Acquisition, Validation, Ratio & DuPont Analysis
=============================================================================================

IFTE0001: AI Agents in Asset Management - Track A

Phase 1: Data Acquisition
- 5-year financial statement collection (Income, Balance, Cash Flow)
- Company overview and market data
- Intelligent 5-hour caching with metadata
- Data quality scoring and tier classification
- Accounting equation validation
- Derived metrics calculation (FCF, EBITDA, Working Capital, Net Debt)
- Dividend history extraction for DDM

Phase 2: Data Validation & Standardization
- Cross-statement reconciliation
- Statistical outlier detection (IQR, Z-score)
- Sign convention validation
- Growth rate analysis (YoY, CAGR)
- Trend classification
- Data standardization for ratio analysis

Phase 3: Financial Ratio Analysis
- 39 ratios across 5 categories (Profitability, Leverage, Liquidity, Efficiency, Growth)
- Time-series ratio calculation with multi-year trends
- Benchmark-based assessment and scoring
- Category-level and overall financial health scoring
- Interest expense estimation for consolidated reporting

Phase 4: DuPont Analysis
- Traditional 3-Factor DuPont decomposition (NPM x AT x EM)
- Extended 5-Factor DuPont decomposition (Tax x Interest x Operating x AT x EM)
- ROA decomposition
- Year-over-year variance attribution with exact reconciliation
- Multi-year trend analysis for all components
- ROE quality and sustainability assessment

Version: 3.3.0
"""

from .config import (
    # Phase 1 Configuration
    ALPHA_VANTAGE_CONFIG,
    DATA_CONFIG,
    VALIDATION_CONFIG,
    LOGGER,
    CACHE_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    ValidationStatus,
    DataQualityTier,
    StatementType,
    
    # Phase 2 Configuration
    RECONCILIATION_CONFIG,
    OUTLIER_CONFIG,
    GROWTH_CONFIG,
    PHASE2_VALIDATION_THRESHOLDS,
    ReconciliationStatus,
    TrendClassification,
    OutlierSeverity,
    SignConvention,
    KEY_GROWTH_METRICS,
    get_sign_convention,
    classify_trend,
    
    # Phase 3 Configuration
    RATIO_ANALYSIS_CONFIG,
    RatioCategory,
    RatioAssessment,
    RatioTrend,
    RatioDefinition,
    PROFITABILITY_RATIOS,
    LEVERAGE_RATIOS,
    LIQUIDITY_RATIOS,
    EFFICIENCY_RATIOS,
    GROWTH_RATIOS,
    ALL_RATIO_DEFINITIONS,
    get_ratio_definition,
    assess_ratio_value,
)

from .data_collector import (
    DataCollector,
    DataPostProcessor,
    CompanyProfile,
    FinancialStatements,
    QualityMetrics,
    DerivedMetrics,
    DividendHistory,
    CollectionResult,
    collect_financial_data,
    __version__ as phase1_version,
)

from .data_validator import (
    Phase2Validator,
    CrossStatementReconciler,
    OutlierDetector,
    GrowthAnalyzer,
    SignValidator,
    DataStandardizer,
    ReconciliationResult,
    ReconciliationReport,
    OutlierResult,
    OutlierReport,
    GrowthMetric,
    GrowthReport,
    SignValidationResult,
    SignValidationReport,
    StandardizedStatements,
    ValidationSummary,
    ValidatedData,
    validate_financial_data,
    __version__ as phase2_version,
)

from .ratio_analyzer import (
    Phase3Analyzer,
    ProfitabilityCalculator,
    LeverageCalculator,
    LiquidityCalculator,
    EfficiencyCalculator,
    GrowthRatioCalculator,
    CategoryAnalyzer,
    RatioValue,
    RatioTimeSeries,
    CategoryAnalysis,
    OverallAssessment,
    RatioAnalysisResult,
    analyze_financial_ratios,
    __version__ as phase3_version,
)

from .dupont_analyzer import (
    Phase4Analyzer,
    DuPontCalculator,
    VarianceAnalyzer,
    TrendAnalyzer,
    QualityAssessor,
    DuPontDriver,
    ROEQuality,
    TrendDirection,
    DuPontComponent,
    ThreeFactorDuPont,
    FiveFactorDuPont,
    ROADecomposition,
    VarianceComponent,
    ROEVarianceAnalysis,
    ComponentTrend,
    QualityAssessment,
    DuPontAnalysisResult,
    analyze_dupont,
    __version__ as phase4_version,
)

# Try to import Yahoo Finance supplementer (optional)
try:
    from .yahoo_supplementer import (
        YahooFinanceSupplementer,
        SupplementationReport,
        SupplementedValue,
        DataSource as SupplementDataSource,
    )
    YAHOO_SUPPLEMENTER_AVAILABLE = True
except ImportError:
    YAHOO_SUPPLEMENTER_AVAILABLE = False
    YahooFinanceSupplementer = None
    SupplementationReport = None
    SupplementedValue = None
    SupplementDataSource = None

__version__ = "3.3.0"

__all__ = [
    # Phase 1 Configuration
    "ALPHA_VANTAGE_CONFIG",
    "DATA_CONFIG",
    "VALIDATION_CONFIG",
    "LOGGER",
    "CACHE_DIR",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    
    # Phase 1 Enums
    "ValidationStatus",
    "DataQualityTier",
    "StatementType",
    
    # Phase 1 Data Containers
    "CompanyProfile",
    "FinancialStatements",
    "QualityMetrics",
    "DerivedMetrics",
    "DividendHistory",
    "CollectionResult",
    
    # Phase 1 Classes
    "DataCollector",
    "DataPostProcessor",
    
    # Phase 1 Functions
    "collect_financial_data",
    
    # Phase 2 Configuration
    "RECONCILIATION_CONFIG",
    "OUTLIER_CONFIG",
    "GROWTH_CONFIG",
    "PHASE2_VALIDATION_THRESHOLDS",
    
    # Phase 2 Enums
    "ReconciliationStatus",
    "TrendClassification",
    "OutlierSeverity",
    "SignConvention",
    "KEY_GROWTH_METRICS",
    
    # Phase 2 Data Containers
    "ReconciliationResult",
    "ReconciliationReport",
    "OutlierResult",
    "OutlierReport",
    "GrowthMetric",
    "GrowthReport",
    "SignValidationResult",
    "SignValidationReport",
    "StandardizedStatements",
    "ValidationSummary",
    "ValidatedData",
    
    # Phase 2 Classes
    "Phase2Validator",
    "CrossStatementReconciler",
    "OutlierDetector",
    "GrowthAnalyzer",
    "SignValidator",
    "DataStandardizer",
    
    # Phase 2 Functions
    "validate_financial_data",
    "get_sign_convention",
    "classify_trend",
    
    # Phase 3 Configuration
    "RATIO_ANALYSIS_CONFIG",
    "PROFITABILITY_RATIOS",
    "LEVERAGE_RATIOS",
    "LIQUIDITY_RATIOS",
    "EFFICIENCY_RATIOS",
    "GROWTH_RATIOS",
    "ALL_RATIO_DEFINITIONS",
    
    # Phase 3 Enums
    "RatioCategory",
    "RatioAssessment",
    "RatioTrend",
    "RatioDefinition",
    
    # Phase 3 Data Containers
    "RatioValue",
    "RatioTimeSeries",
    "CategoryAnalysis",
    "OverallAssessment",
    "RatioAnalysisResult",
    
    # Phase 3 Classes
    "Phase3Analyzer",
    "ProfitabilityCalculator",
    "LeverageCalculator",
    "LiquidityCalculator",
    "EfficiencyCalculator",
    "GrowthRatioCalculator",
    "CategoryAnalyzer",
    
    # Phase 3 Functions
    "analyze_financial_ratios",
    "get_ratio_definition",
    "assess_ratio_value",
    
    # Phase 4 Enums
    "DuPontDriver",
    "ROEQuality",
    "TrendDirection",
    
    # Phase 4 Data Containers
    "DuPontComponent",
    "ThreeFactorDuPont",
    "FiveFactorDuPont",
    "ROADecomposition",
    "VarianceComponent",
    "ROEVarianceAnalysis",
    "ComponentTrend",
    "QualityAssessment",
    "DuPontAnalysisResult",
    
    # Phase 4 Classes
    "Phase4Analyzer",
    "DuPontCalculator",
    "VarianceAnalyzer",
    "TrendAnalyzer",
    "QualityAssessor",
    
    # Phase 4 Functions
    "analyze_dupont",
    
    # Yahoo Finance Supplementer (optional)
    "YAHOO_SUPPLEMENTER_AVAILABLE",
    "YahooFinanceSupplementer",
    "SupplementationReport",
    "SupplementedValue",
    "SupplementDataSource",
    
    # Version
    "__version__",
    "phase1_version",
    "phase2_version",
    "phase3_version",
    "phase4_version",
]