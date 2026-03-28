"""
MPS Performance Optimization Config Loader
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .smart_optimization_controller import OptimizationThresholds

logger = logging.getLogger(__name__)


class OptimizationConfigLoader:
    """MPS Performance Optimization Config Loader"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # default config file path
            current_dir = Path(__file__).parent
            config_path = current_dir / "optimization_config.json"
        
        self.config_path = Path(config_path)
        self.config_data = None
        self._load_config()
    
    def _load_config(self):
        """Load optimization config file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self.config_data = json.load(f)
                logger.info(f"Loaded optimization config from {self.config_path}")
            else:
                logger.warning(f"Config file not found: {self.config_path}, using default settings")
                self.config_data = self._get_default_config()
        except Exception as e:
            logger.error(f"Failed to load config file: {e}, using default settings")
            self.config_data = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default optimization config"""
        return {
            "optimization_thresholds": {
                "default_thresholds": {
                    "min_frame_pixels": 2073600,  # 1080p
                    "min_batch_size": 4,
                    "min_processing_complexity": 0.6,
                    "min_concurrent_tasks": 2,
                    "min_memory_usage_mb": 500,
                    "min_compute_time_sec": 0.5
                }
            },
            "load_level_configs": {
                "light_load": {"enable_optimization": False},
                "medium_load": {"enable_optimization": False},
                "heavy_load": {"enable_optimization": True},
                "stress_load": {"enable_optimization": True}
            }
        }
    
    def get_thresholds_for_scenario(self, scenario: str = "default") -> OptimizationThresholds:
        """Get optimization thresholds for specific scenario"""
        try:
            thresholds_config = self.config_data.get("optimization_thresholds", {})
            
            if scenario != "default" and scenario in thresholds_config.get("scenario_specific_thresholds", {}):
                # Use scenario-specific thresholds
                scenario_config = thresholds_config["scenario_specific_thresholds"][scenario]
                logger.info(f"Using scenario-specific thresholds for: {scenario}")
            else:
                # use default thresholds
                scenario_config = thresholds_config.get("default_thresholds", {})
                logger.info("Using default optimization thresholds")
            
            return OptimizationThresholds(
                min_frame_pixels=scenario_config.get("min_frame_pixels", 2073600),
                min_batch_size=scenario_config.get("min_batch_size", 4),
                min_processing_complexity=scenario_config.get("min_processing_complexity", 0.6),
                min_concurrent_tasks=scenario_config.get("min_concurrent_tasks", 2),
                min_memory_usage_mb=scenario_config.get("min_memory_usage_mb", 500),
                min_compute_time_sec=scenario_config.get("min_compute_time_sec", 0.5)
            )
        except Exception as e:
            logger.error(f"Failed to load thresholds for scenario {scenario}: {e}")
            return OptimizationThresholds()  # Use default values
    
    def should_enable_optimization_for_load_level(self, load_level: str) -> bool:
        """Check if optimization should be enabled for specific load level"""
        try:
            load_configs = self.config_data.get("load_level_configs", {})
            load_config = load_configs.get(load_level, {})
            return load_config.get("enable_optimization", False)
        except Exception as e:
            logger.error(f"Failed to check optimization setting for load level {load_level}: {e}")
            # default policy: only enable optimization for heavy_load and stress_load
            return load_level in ["heavy_load", "stress_load"]
    
    def get_optimization_config_for_load_level(self, load_level: str) -> Dict[str, Any]:
        """Get optimization config for specific load level"""
        try:
            load_configs = self.config_data.get("load_level_configs", {})
            load_config = load_configs.get(load_level, {})
            return load_config.get("optimization_config", {})
        except Exception as e:
            logger.error(f"Failed to get optimization config for load level {load_level}: {e}")
            return {}
    
    def get_expected_performance_gain(self, load_level: str) -> float:
        """Get expected performance gain for specific load level"""
        try:
            load_configs = self.config_data.get("load_level_configs", {})
            load_config = load_configs.get(load_level, {})
            return load_config.get("expected_performance_change", 0.0)
        except Exception as e:
            logger.error(f"Failed to get expected performance gain for {load_level}: {e}")
            return 0.0
    
    def get_integration_settings(self) -> Dict[str, Any]:
        """Get integration settings"""
        return self.config_data.get("integration_settings", {})
    
    def is_scenario_force_disabled(self, scenario: str) -> bool:
        """Check if scenario is force disabled for optimization"""
        try:
            thresholds_config = self.config_data.get("optimization_thresholds", {})
            scenario_config = thresholds_config.get("scenario_specific_thresholds", {}).get(scenario, {})
            return scenario_config.get("force_disable", False)
        except Exception as e:
            logger.error(f"Failed to check force disable for scenario {scenario}: {e}")
            return False
    
    def is_scenario_force_enabled(self, scenario: str, frame_pixels: int = 0, batch_size: int = 0) -> bool:
        """Check if scenario is force enabled for optimization"""
        try:
            thresholds_config = self.config_data.get("optimization_thresholds", {})
            scenario_config = thresholds_config.get("scenario_specific_thresholds", {}).get(scenario, {})
            
            # Check if force enable for 4K
            if scenario_config.get("force_enable_for_4k", False) and frame_pixels >= 3840 * 2160:
                return True
            
            # Check if force enable for batch size greater than 4
            if scenario_config.get("force_enable_for_batch_gt_4", False) and batch_size > 4:
                return True
            
            return False
        except Exception as e:
            logger.error(f"Failed to check force enable for scenario {scenario}: {e}")
            return False
    
    def get_benchmark_results(self) -> Dict[str, Any]:
        """Get benchmark results"""
        return self.config_data.get("benchmark_results", {})
    
    def reload_config(self):
        """Reload configuration file"""
        self._load_config()
        logger.info("Configuration reloaded")
    
    def save_config(self, config_data: Dict[str, Any], backup: bool = True):
        """Save configuration file"""
        try:
            if backup and self.config_path.exists():
                backup_path = self.config_path.with_suffix('.json.backup')
                import shutil
                shutil.copy2(self.config_path, backup_path)
                logger.info(f"Created backup at {backup_path}")
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.config_data = config_data
            logger.info(f"Configuration saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise


# Global configuration loader instance
_global_config_loader = None


def get_config_loader(config_path: Optional[str] = None) -> OptimizationConfigLoader:
    """Get global configuration loader instance"""
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = OptimizationConfigLoader(config_path)
    return _global_config_loader


def reload_global_config():
    """Reload global configuration"""
    global _global_config_loader
    if _global_config_loader:
        _global_config_loader.reload_config()


# Convenience functions
def get_thresholds_for_scenario(scenario: str = "default") -> OptimizationThresholds:
    """Get thresholds for specific scenario"""
    return get_config_loader().get_thresholds_for_scenario(scenario)


def should_enable_optimization_for_load_level(load_level: str) -> bool:
    """Check if optimization should be enabled for specific load level"""
    return get_config_loader().should_enable_optimization_for_load_level(load_level)


def get_expected_performance_gain(load_level: str) -> float:
    """Get expected performance gain for specific load level"""
    return get_config_loader().get_expected_performance_gain(load_level)