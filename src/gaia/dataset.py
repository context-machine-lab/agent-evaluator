"""GAIA benchmark utilities for dataset loading and evaluation"""

import json
import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Any
from ..core.logger import ResearchLogger


class GAIADataset:
    """Load and manage GAIA benchmark data"""

    def __init__(self, data_path: str, split: str = "validation"):
        """Initialize GAIA dataset

        Args:
            data_path: Path to GAIA dataset directory
            split: Dataset split (validation or test)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.logger = ResearchLogger()
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load GAIA data from JSON file

        Returns:
            DataFrame with GAIA tasks
        """
        # GAIA dataset file naming convention
        if self.split == "validation":
            json_file = self.data_path / "2023_validation.json"
        elif self.split == "test":
            json_file = self.data_path / "2023_test.json"
        else:
            raise ValueError(f"Invalid split: {self.split}")

        if not json_file.exists():
            raise FileNotFoundError(f"GAIA dataset not found at {json_file}")

        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Ensure required columns exist
        required_cols = ['task_id', 'question']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Add missing columns with defaults
        if 'file_name' not in df.columns:
            df['file_name'] = None
        if 'final_answer' not in df.columns:
            df['final_answer'] = "?"  # Unknown for test set

        self.logger.info(f"Loaded {len(df)} tasks from {self.split} split")
        return df

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Get a specific task by ID

        Args:
            task_id: Task identifier

        Returns:
            Task dictionary
        """
        task = self.data[self.data['task_id'] == task_id]
        if task.empty:
            raise ValueError(f"Task {task_id} not found")
        return task.iloc[0].to_dict()

    def get_tasks(self, max_tasks: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all tasks or a subset

        Args:
            max_tasks: Maximum number of tasks to return

        Returns:
            List of task dictionaries
        """
        tasks = self.data
        if max_tasks:
            tasks = tasks.head(max_tasks)
        return tasks.to_dict(orient="records")

    def __len__(self) -> int:
        """Number of tasks in dataset"""
        return len(self.data)

    def __repr__(self) -> str:
        """String representation"""
        return f"GAIADataset(split={self.split}, tasks={len(self)})"


class GAIAScorer:
    """Score GAIA predictions against ground truth"""

    def __init__(self):
        """Initialize GAIA scorer"""
        self.logger = ResearchLogger()

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison

        Args:
            answer: Raw answer string

        Returns:
            Normalized answer
        """
        if answer is None:
            return ""

        # Convert to string and lowercase
        answer = str(answer).lower().strip()

        # Remove common punctuation
        answer = re.sub(r'[.,;:!?]', '', answer)

        # Normalize whitespace
        answer = ' '.join(answer.split())

        return answer

    def score_answer(self, prediction: str, truth: str) -> bool:
        """Score a single prediction

        Args:
            prediction: Model prediction
            truth: Ground truth answer

        Returns:
            True if correct, False otherwise
        """
        # Skip if no ground truth (test set)
        if truth == "?" or truth is None:
            return None

        # Handle "Unable to determine" predictions
        if prediction is None or "unable to determine" in str(prediction).lower():
            return False

        # Normalize both answers
        pred_norm = self.normalize_answer(prediction)
        truth_norm = self.normalize_answer(truth)

        # Check exact match
        if pred_norm == truth_norm:
            return True

        # Check if truth is contained in prediction
        if truth_norm in pred_norm:
            return True

        # Check numeric equivalence
        try:
            pred_num = float(pred_norm.replace(',', '').replace(' ', ''))
            truth_num = float(truth_norm.replace(',', '').replace(' ', ''))
            if abs(pred_num - truth_num) < 0.001:
                return True
        except (ValueError, AttributeError):
            pass

        return False

    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute evaluation metrics

        Args:
            results: List of result dictionaries with predictions and truths

        Returns:
            Dictionary of metrics
        """
        total = len(results)
        if total == 0:
            return {"accuracy": 0.0, "total": 0, "correct": 0, "invalid": 0}

        correct = 0
        invalid = 0
        scored = 0

        for result in results:
            prediction = result.get('prediction')
            truth = result.get('true_answer', result.get('final_answer'))

            # Skip test examples without ground truth
            if truth == "?" or truth is None:
                continue

            scored += 1

            # Check if prediction is invalid
            if prediction is None or "unable to determine" in str(prediction).lower():
                invalid += 1
                continue

            # Score the prediction
            if self.score_answer(prediction, truth):
                correct += 1

        accuracy = (correct / scored * 100) if scored > 0 else 0

        metrics = {
            "accuracy": accuracy,
            "total": total,
            "scored": scored,
            "correct": correct,
            "invalid": invalid,
            "skipped": total - scored
        }

        return metrics

    def print_summary(self, metrics: Dict[str, float]) -> None:
        """Print evaluation summary

        Args:
            metrics: Computed metrics dictionary
        """
        self.logger.section("GAIA Evaluation Summary")
        self.logger.metrics({
            "Total Tasks": metrics['total'],
            "Scored Tasks": metrics['scored'],
            "Correct": metrics['correct'],
            "Invalid": metrics['invalid'],
            "Skipped (no truth)": metrics['skipped'],
            "Accuracy": f"{metrics['accuracy']:.2f}%"
        })


class TaskFilter:
    """Filter completed tasks from previous runs"""

    def __init__(self, results_path: str):
        """Initialize task filter

        Args:
            results_path: Path to results JSONL file
        """
        self.results_path = Path(results_path)
        self.logger = ResearchLogger()
        self.completed = self._load_completed()

    def _load_completed(self) -> set:
        """Load completed task IDs from results file

        Returns:
            Set of completed task IDs
        """
        completed = set()

        if not self.results_path.exists():
            return completed

        try:
            with open(self.results_path, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        task_id = result.get('task_id')
                        prediction = result.get('prediction')

                        # Only skip if we have a valid prediction
                        if task_id and prediction and str(prediction) != "Unable to determine":
                            completed.add(task_id)
                    except json.JSONDecodeError:
                        continue

            if completed:
                self.logger.info(f"Found {len(completed)} completed tasks")

        except Exception as e:
            self.logger.warning(f"Error loading previous results: {e}")

        return completed

    def filter_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out completed tasks

        Args:
            tasks: List of all tasks

        Returns:
            List of tasks to run
        """
        if not self.completed:
            return tasks

        filtered = [
            task for task in tasks
            if task['task_id'] not in self.completed
        ]

        if len(filtered) < len(tasks):
            self.logger.info(f"Filtered {len(tasks) - len(filtered)} completed tasks")

        return filtered

    def is_completed(self, task_id: str) -> bool:
        """Check if a task is already completed

        Args:
            task_id: Task identifier

        Returns:
            True if completed, False otherwise
        """
        return task_id in self.completed