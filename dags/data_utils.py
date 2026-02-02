"""
Consolidated Data Utilities for Customer Churn Prediction Pipeline

This module provides a unified interface for:
- Configuration management with environment variable substitution
- Comprehensive data validation with quality scoring
- Model registry and version management  
- Logging utilities and performance tracking
"""

import pandas as pd
import numpy as np
import yaml
import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationResult:
    """Container for data validation results with quality metrics"""
    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def summary(self) -> str:
        """Get validation summary string"""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"{status} | Quality: {self.quality_score:.2f} | Issues: {len(self.issues)} | Warnings: {len(self.warnings)}"


# =============================================================================
# UNIFIED DATA UTILITIES CLASS
# =============================================================================

class DataUtils:
    """Unified data utilities with configuration, validation, and model registry"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        self._logger = self._setup_logger()
        self._model_registry = {}
        self._registry_path = Path("models/registry.json")
        self._load_model_registry()
    
    # =========================================================================
    # CONFIGURATION MANAGEMENT
    # =========================================================================
    
    @property
    def config(self) -> Dict[str, Any]:
        """Lazy load and cache configuration"""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> Dict[str, Any]:
        """Load and process configuration with environment variable substitution"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Recursively substitute environment variables
        config = self._substitute_env_vars(config)
        self._logger.info(f"Configuration loaded from {self.config_path}")
        return config
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Recursively substitute ${VAR:default} environment variables"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            var_spec = obj[2:-1]
            var_name, default = var_spec.split(':', 1) if ':' in var_spec else (var_spec, obj)
            return os.getenv(var_name, default)
        return obj
    
    # =========================================================================
    # DATA VALIDATION 
    # =========================================================================
    
    def validate_data(self, df: pd.DataFrame, stage: str = 'training') -> ValidationResult:
        """
        Comprehensive data validation with quality scoring
        
        Args:
            df: DataFrame to validate
            stage: Validation stage ('training', 'prediction', 'production')
            
        Returns:
            ValidationResult with detailed findings and quality score
        """
        self._logger.info(f"Validating dataset at {stage} stage: {df.shape}")
        
        issues, warnings, metrics = [], [], {}
        
        # Run all validation checks in one pass
        structure_issues = self._validate_structure(df, stage)
        quality_issues, quality_metrics = self._validate_quality(df)
        stat_warnings = self._validate_statistics(df)
        
        issues.extend(structure_issues + quality_issues)
        warnings.extend(stat_warnings)
        metrics.update(quality_metrics)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(df, issues, warnings)
        
        # Create comprehensive report
        report = self._create_report(df, issues, warnings, metrics)
        
        result = ValidationResult(
            is_valid=not any(issue.startswith('CRITICAL:') for issue in issues),
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
            report=report
        )
        
        self._logger.info(f"Validation complete: {result.summary}")
        return result
    
    def _validate_structure(self, df: pd.DataFrame, stage: str) -> List[str]:
        """Validate dataset structure and schema"""
        issues = []
        validation_rules = self.config.get('data', {}).get('validation_rules', {})
        
        # Basic structure checks
        if df.empty:
            issues.append("CRITICAL: Dataset is empty")
            return issues
        
        # Minimum sample size
        min_samples = validation_rules.get('min_samples', 100)
        if len(df) < min_samples:
            issues.append(f"WARNING: Only {len(df)} samples (minimum: {min_samples})")
        
        # Schema validation
        expected_cols = set(self.config.get('data', {}).get('expected_columns', []))
        if expected_cols:
            missing_cols = expected_cols - set(df.columns)
            if missing_cols:
                issues.append(f"Missing expected columns: {list(missing_cols)}")
        
        # Target validation for training
        if stage == 'training':
            target_issues = self._validate_target(df)
            issues.extend(target_issues)
        
        return issues
    
    def _validate_quality(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """Validate data quality metrics"""
        issues = []
        metrics = {}
        validation_rules = self.config.get('data', {}).get('validation_rules', {})
        
        # Null analysis
        null_percentages = df.isnull().sum() / len(df)
        null_threshold = validation_rules.get('null_threshold', 0.3)
        critical_nulls = null_percentages[null_percentages > null_threshold]
        
        if not critical_nulls.empty:
            issues.append(f"High null percentages: {dict(critical_nulls.round(3))}")
        
        # Duplicate analysis
        duplicate_count = df.duplicated().sum()
        duplicate_percentage = duplicate_count / len(df)
        duplicate_threshold = validation_rules.get('duplicate_threshold', 0.05)
        
        if duplicate_percentage > duplicate_threshold:
            issues.append(f"High duplicate percentage: {duplicate_percentage:.2%}")
        
        # Data type validation
        type_issues = self._validate_types(df)
        issues.extend(type_issues)
        
        metrics.update({
            'null_percentages': null_percentages.to_dict(),
            'duplicate_count': duplicate_count,
            'duplicate_percentage': duplicate_percentage
        })
        
        return issues, metrics
    
    def _validate_statistics(self, df: pd.DataFrame) -> List[str]:
        """Validate statistical properties"""
        warnings = []
        validation_rules = self.config.get('data', {}).get('validation_rules', {})
        outlier_threshold = validation_rules.get('outlier_threshold', 3.0)
        
        # Get numerical columns
        numerical_cols = self._get_numerical_columns(df)
        
        for col in numerical_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                # Outlier detection
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outlier_percentage = (z_scores > outlier_threshold).sum() / len(z_scores)
                
                if outlier_percentage > 0.05:
                    warnings.append(f"Column '{col}' has {outlier_percentage:.1%} potential outliers")
                
                # Constant values
                if df[col].nunique() <= 1:
                    warnings.append(f"Column '{col}' has constant values")
        
        return warnings
    
    def _validate_target(self, df: pd.DataFrame) -> List[str]:
        """Validate target variable for training datasets"""
        issues = []
        target_col = 'Churn'
        
        if target_col not in df.columns:
            issues.append(f"Target column '{target_col}' not found")
            return issues
        
        # Class balance check
        target_balance = df[target_col].value_counts(normalize=True)
        min_class_percentage = target_balance.min()
        balance_threshold = self.config.get('data', {}).get('validation_rules', {}).get('target_balance_threshold', 0.05)
        
        if min_class_percentage < balance_threshold:
            issues.append(f"Severe class imbalance: minority class is {min_class_percentage:.1%}")
        
        # Binary validation
        unique_values = df[target_col].dropna().unique()
        if len(unique_values) != 2 or not all(v in [0, 1] for v in unique_values):
            issues.append(f"Target should be binary [0,1], found: {unique_values}")
        
        return issues
    
    def _validate_types(self, df: pd.DataFrame) -> List[str]:
        """Validate column data types"""
        issues = []
        
        # Check numerical columns
        numerical_cols = self._get_numerical_columns(df)
        for col in numerical_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' should be numerical, found {df[col].dtype}")
        
        # Check categorical columns
        categorical_cols = self._get_categorical_columns()
        for col in categorical_cols:
            if col in df.columns and df[col].nunique() > 50:
                issues.append(f"Column '{col}' has {df[col].nunique()} unique values (too high for categorical)")
        
        return issues
    
    def _get_numerical_columns(self, df: pd.DataFrame = None) -> List[str]:
        """Get numerical columns from config or infer from dataframe"""
        config_numerical = (self.config.get('data', {}).get('numerical_columns', []) or 
                           self.config.get('features', {}).get('numerical', []))
        
        if config_numerical:
            return config_numerical
        
        if df is not None:
            return df.select_dtypes(include=[np.number]).columns.tolist()
        
        return []
    
    def _get_categorical_columns(self) -> List[str]:
        """Get categorical columns from config"""
        return (self.config.get('data', {}).get('categorical_columns', []) or 
                self.config.get('features', {}).get('categorical', []))
    
    def _calculate_quality_score(self, df: pd.DataFrame, issues: List[str], warnings: List[str]) -> float:
        """Calculate overall data quality score (0-1)"""
        base_score = 1.0
        
        # Apply penalties
        critical_issues = [i for i in issues if i.startswith('CRITICAL:')]
        regular_issues = [i for i in issues if not i.startswith(('CRITICAL:', 'WARNING:'))]
        warning_issues = [i for i in issues if i.startswith('WARNING:')]
        
        base_score -= len(critical_issues) * 0.4    # Heavy penalty for critical issues
        base_score -= len(regular_issues) * 0.15    # Moderate penalty for issues  
        base_score -= len(warning_issues) * 0.05    # Light penalty for warnings
        base_score -= len(warnings) * 0.02          # Very light penalty for warnings
        
        # Quality bonuses
        completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
        base_score += completeness * 0.1
        
        return max(0.0, min(1.0, base_score))
    
    def _create_report(self, df: pd.DataFrame, issues: List[str], warnings: List[str], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive validation report"""
        return {
            'dataset_info': {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'columns': df.columns.tolist()
            },
            'quality_metrics': metrics,
            'data_types': df.dtypes.astype(str).to_dict(),
            'missing_data': df.isnull().sum().to_dict(),
            'summary_stats': df.describe().to_dict() if not df.empty else {},
            'issues': issues,
            'warnings': warnings,
            'validation_timestamp': datetime.now().isoformat()
        }
    
    # =========================================================================
    # MODEL REGISTRY
    # =========================================================================
    
    def _load_model_registry(self):
        """Load existing model registry"""
        if self._registry_path.exists():
            try:
                with open(self._registry_path, 'r') as f:
                    self._model_registry = json.load(f)
            except Exception as e:
                self._logger.warning(f"Error loading model registry: {e}")
                self._model_registry = {'models': {}, 'version_counter': 0}
        else:
            self._model_registry = {'models': {}, 'version_counter': 0}
            self._registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        try:
            with open(self._registry_path, 'w') as f:
                json.dump(self._model_registry, f, indent=2)
        except Exception as e:
            self._logger.error(f"Error saving model registry: {e}")
    
    def register_model(self, model_path: str, model_name: str, model_type: str,
                      metrics: Dict[str, float], metadata: Optional[Dict] = None) -> int:
        """Register new model version"""
        self._model_registry['version_counter'] += 1
        version = self._model_registry['version_counter']
        
        if model_name not in self._model_registry['models']:
            self._model_registry['models'][model_name] = {'versions': {}, 'stages': {}}
        
        model_info = {
            'version': version,
            'model_type': model_type,
            'model_path': str(model_path),
            'metrics': metrics,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat(),
            'stage': 'staging'
        }
        
        self._model_registry['models'][model_name]['versions'][str(version)] = model_info
        self._model_registry['models'][model_name]['stages']['staging'] = version
        
        self._save_model_registry()
        self._logger.info(f"Registered {model_name} v{version} in staging")
        return version
    
    def promote_model(self, model_name: str, version: int, stage: str = 'production') -> bool:
        """Promote model to specified stage"""
        if (model_name not in self._model_registry['models'] or 
            str(version) not in self._model_registry['models'][model_name]['versions']):
            self._logger.error(f"Model {model_name} v{version} not found")
            return False
        
        self._model_registry['models'][model_name]['stages'][stage] = version
        self._model_registry['models'][model_name]['versions'][str(version)]['stage'] = stage
        self._model_registry['models'][model_name]['versions'][str(version)]['promoted_at'] = datetime.now().isoformat()
        
        self._save_model_registry()
        self._logger.info(f"Promoted {model_name} v{version} to {stage}")
        return True
    
    def get_model_info(self, model_name: str, stage: str = 'production') -> Optional[Dict[str, Any]]:
        """Get model information for specific stage"""
        if (model_name not in self._model_registry['models'] or 
            stage not in self._model_registry['models'][model_name]['stages']):
            return None
        
        version = self._model_registry['models'][model_name]['stages'][stage]
        return self._model_registry['models'][model_name]['versions'][str(version)]
    
    def list_models(self) -> Dict[str, Any]:
        """List all registered models"""
        return self._model_registry['models']
    
    # =========================================================================
    # LOGGING UTILITIES
    # =========================================================================
    
    def _setup_logger(self) -> logging.Logger:
        """Setup configured logger instance"""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float], stage: str = "evaluation"):
        """Log model performance metrics"""
        self._logger.info(f"=== {model_name.upper()} {stage.upper()} METRICS ===")
        for metric, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                self._logger.info(f"{metric}: {value:.4f}")
            else:
                self._logger.info(f"{metric}: {value}")
        self._logger.info("=" * 50)
    
    def log_data_summary(self, df: pd.DataFrame, stage: str = "processing"):
        """Log comprehensive dataset summary"""
        self._logger.info(f"=== DATA SUMMARY - {stage.upper()} ===")
        self._logger.info(f"Shape: {df.shape}")
        self._logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        if 'Churn' in df.columns:
            churn_rate = df['Churn'].mean()
            self._logger.info(f"Churn rate: {churn_rate:.2%}")
        
        self._logger.info("=" * 50)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self.config.get('features', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model training configuration"""
        return self.config.get('model', {})
    
    def get_mlflow_config(self) -> Dict[str, Any]:
        """Get MLflow tracking configuration"""
        return self.config.get('mlflow', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration"""
        return self.config.get('data', {})


# =============================================================================
# GLOBAL INSTANCE AND CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance for easy access
_data_utils = DataUtils()

def get_config() -> Dict[str, Any]:
    """Get application configuration"""
    return _data_utils.config

def get_logger(name: str = __name__) -> logging.Logger:
    """Get configured logger instance"""
    return logging.getLogger(name)

def validate_data(df: pd.DataFrame, stage: str = 'training') -> ValidationResult:
    """Validate dataset with comprehensive checks"""
    return _data_utils.validate_data(df, stage)

def log_model_metrics(model_name: str, metrics: Dict[str, float], stage: str = "evaluation"):
    """Log model performance metrics"""
    _data_utils.log_model_performance(model_name, metrics, stage)

def log_data_summary(df: pd.DataFrame, stage: str = "processing"):
    """Log dataset summary information"""
    _data_utils.log_data_summary(df, stage)

def register_model(model_path: str, model_name: str, model_type: str,
                  metrics: Dict[str, float], metadata: Optional[Dict] = None) -> int:
    """Register new model version"""
    return _data_utils.register_model(model_path, model_name, model_type, metrics, metadata)

def promote_model(model_name: str, version: int, stage: str = 'production') -> bool:
    """Promote model to specified stage"""
    return _data_utils.promote_model(model_name, version, stage)

def get_model_info(model_name: str, stage: str = 'production') -> Optional[Dict[str, Any]]:
    """Get model information for specific stage"""
    return _data_utils.get_model_info(model_name, stage)