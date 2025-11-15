"""
Experiment tracking utility for organizing model iterations and results.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class ExperimentTracker:
    """
    Tracks experiments and model iterations for organized development.
    
    This helps keep track of different model versions, their configurations,
    and results across multiple training runs.
    """
    
    def __init__(self, experiment_name: str, base_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for storing experiment data
        """
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.checkpoints_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.configs_dir = self.experiment_dir / "configs"
        self.results_dir = self.experiment_dir / "results"
        
        for directory in [self.checkpoints_dir, self.logs_dir, 
                         self.configs_dir, self.results_dir]:
            directory.mkdir(exist_ok=True)
        
        # Load or initialize experiment metadata
        self.metadata_file = self.experiment_dir / "metadata.json"
        self.metadata = self._load_metadata()
        
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load experiment metadata from file.
        
        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'experiment_name': self.experiment_name,
                'created_at': datetime.now().isoformat(),
                'runs': []
            }
    
    def _save_metadata(self):
        """Save experiment metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def start_run(self, run_name: str, config: Dict[str, Any]) -> str:
        """
        Start a new training run.
        
        Args:
            run_name: Name for this run (e.g., 'cnn_v1_run1')
            config: Configuration dictionary for this run
            
        Returns:
            Run ID
        """
        run_id = f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        run_info = {
            'run_id': run_id,
            'run_name': run_name,
            'started_at': datetime.now().isoformat(),
            'config': config,
            'status': 'running'
        }
        
        self.metadata['runs'].append(run_info)
        self._save_metadata()
        
        # Save run configuration
        config_path = self.configs_dir / f"{run_id}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return run_id
    
    def end_run(self, run_id: str, results: Dict[str, Any], status: str = 'completed'):
        """
        End a training run and save results.
        
        Args:
            run_id: ID of the run to end
            results: Dictionary with training results/metrics
            status: Final status ('completed', 'failed', 'stopped')
        """
        # Update metadata
        for run in self.metadata['runs']:
            if run['run_id'] == run_id:
                run['ended_at'] = datetime.now().isoformat()
                run['status'] = status
                run['results'] = results
                break
        
        self._save_metadata()
        
        # Save detailed results
        results_path = self.results_dir / f"{run_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: int):
        """
        Log metrics for a training run.
        
        Args:
            run_id: ID of the run
            metrics: Dictionary of metric name to value
            step: Training step/epoch number
        """
        log_file = self.logs_dir / f"{run_id}_metrics.jsonl"
        
        log_entry = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_best_run(self, metric: str = 'val_loss', minimize: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a specific metric.
        
        Args:
            metric: Metric name to compare
            minimize: Whether to minimize (True) or maximize (False) the metric
            
        Returns:
            Best run information or None
        """
        completed_runs = [run for run in self.metadata['runs'] 
                         if run.get('status') == 'completed' 
                         and 'results' in run 
                         and metric in run['results']]
        
        if not completed_runs:
            return None
        
        if minimize:
            return min(completed_runs, key=lambda r: r['results'][metric])
        else:
            return max(completed_runs, key=lambda r: r['results'][metric])
    
    def list_runs(self, status: Optional[str] = None) -> list:
        """
        List all runs, optionally filtered by status.
        
        Args:
            status: Optional status filter ('running', 'completed', 'failed', 'stopped')
            
        Returns:
            List of run information dictionaries
        """
        if status:
            return [run for run in self.metadata['runs'] if run.get('status') == status]
        return self.metadata['runs']
    
    def get_checkpoint_path(self, run_id: str, epoch: Optional[int] = None) -> Path:
        """
        Get path for saving/loading checkpoint.
        
        Args:
            run_id: ID of the run
            epoch: Optional epoch number (if None, returns 'best' checkpoint path)
            
        Returns:
            Path object for checkpoint
        """
        if epoch is not None:
            filename = f"{run_id}_epoch_{epoch}.pth"
        else:
            filename = f"{run_id}_best.pth"
        
        return self.checkpoints_dir / filename
