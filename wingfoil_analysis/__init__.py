"""Wingfoil activity analysis package."""

from wingfoil_analysis.pipeline import (
    ANALYSIS_VERSION,
    AnalysisError,
    AnalysisConfig,
    AnalysisResult,
    Fall,
    Run,
    Sample,
    WindContext,
    analysis_result_to_dict,
    analyze_session_file,
    analyze_activity,
    build_wind_context,
    default_analysis_config,
    write_analysis_outputs,
)

__all__ = [
    "ANALYSIS_VERSION",
    "AnalysisError",
    "AnalysisConfig",
    "AnalysisResult",
    "Fall",
    "Run",
    "Sample",
    "WindContext",
    "analysis_result_to_dict",
    "analyze_session_file",
    "analyze_activity",
    "build_wind_context",
    "default_analysis_config",
    "write_analysis_outputs",
]
