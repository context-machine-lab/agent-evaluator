"""
Planning Tool for hierarchical agent orchestration.
Manages structured plans with step tracking and progress monitoring.
"""

import uuid
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from ..core.registry import register_tool
from ..core.logger import logger


class StepStatus(str, Enum):
    """Status of a plan step."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in a plan."""
    description: str
    status: StepStatus = StepStatus.NOT_STARTED
    agent: Optional[str] = None  # Which agent should handle this
    dependencies: List[int] = field(default_factory=list)  # Step indices this depends on
    result: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data


@dataclass
class Plan:
    """A structured plan for task execution."""
    plan_id: str
    title: str
    steps: List[PlanStep]
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> Dict[str, Any]:
        """Calculate plan progress."""
        total = len(self.steps)
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        in_progress = sum(1 for s in self.steps if s.status == StepStatus.IN_PROGRESS)
        blocked = sum(1 for s in self.steps if s.status == StepStatus.BLOCKED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)

        return {
            "total_steps": total,
            "completed": completed,
            "in_progress": in_progress,
            "blocked": blocked,
            "failed": failed,
            "not_started": total - completed - in_progress - blocked - failed,
            "percentage": (completed / total * 100) if total > 0 else 0
        }

    def get_next_steps(self) -> List[int]:
        """Get indices of steps that can be executed next."""
        next_steps = []
        for i, step in enumerate(self.steps):
            if step.status != StepStatus.NOT_STARTED:
                continue

            # Check if dependencies are satisfied
            deps_satisfied = all(
                self.steps[dep].status == StepStatus.COMPLETED
                for dep in step.dependencies
            )

            if deps_satisfied:
                next_steps.append(i)

        return next_steps

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "steps": [s.to_dict() for s in self.steps],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "progress": self.progress
        }

    def format_display(self) -> str:
        """Format plan for display."""
        lines = [
            f"\nPlan: {self.title} (ID: {self.plan_id})",
            "=" * 80,
            f"\nProgress: {self.progress['completed']}/{self.progress['total_steps']} steps completed ({self.progress['percentage']:.1f}%)",
            f"Status: {self.progress['completed']} completed, {self.progress['in_progress']} in progress, "
            f"{self.progress['blocked']} blocked, {self.progress['not_started']} not started\n",
            "Steps:"
        ]

        for i, step in enumerate(self.steps):
            status_icon = {
                StepStatus.NOT_STARTED: "[ ]",
                StepStatus.IN_PROGRESS: "[→]",
                StepStatus.COMPLETED: "[✓]",
                StepStatus.BLOCKED: "[⊗]",
                StepStatus.FAILED: "[✗]",
                StepStatus.SKIPPED: "[—]"
            }.get(step.status, "[ ]")

            agent_info = f" ({step.agent})" if step.agent else ""
            deps_info = f" [deps: {step.dependencies}]" if step.dependencies else ""

            lines.append(f"{i}. {status_icon} {step.description}{agent_info}{deps_info}")

            if step.result:
                lines.append(f"   └─ Result: {step.result[:100]}...")
            if step.error:
                lines.append(f"   └─ Error: {step.error}")

        return "\n".join(lines)


class PlanManager:
    """Manages multiple plans."""

    def __init__(self):
        self.plans: Dict[str, Plan] = {}

    def create_plan(self, title: str, steps: List[Dict[str, Any]], plan_id: Optional[str] = None) -> Plan:
        """Create a new plan."""
        if not plan_id:
            plan_id = str(uuid.uuid4())[:8]

        plan_steps = []
        for step_data in steps:
            step = PlanStep(
                description=step_data["description"],
                agent=step_data.get("agent"),
                dependencies=step_data.get("dependencies", [])
            )
            plan_steps.append(step)

        plan = Plan(
            plan_id=plan_id,
            title=title,
            steps=plan_steps
        )

        self.plans[plan_id] = plan
        logger.info(f"Created plan '{title}' with ID: {plan_id}")
        return plan

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self.plans.get(plan_id)

    def update_step(self, plan_id: str, step_index: int,
                    status: Optional[StepStatus] = None,
                    result: Optional[str] = None,
                    error: Optional[str] = None) -> bool:
        """Update a step in a plan."""
        plan = self.get_plan(plan_id)
        if not plan or step_index >= len(plan.steps):
            return False

        step = plan.steps[step_index]

        if status:
            # Track timing
            if status == StepStatus.IN_PROGRESS and not step.started_at:
                step.started_at = datetime.now()
            elif status in [StepStatus.COMPLETED, StepStatus.FAILED, StepStatus.SKIPPED]:
                step.completed_at = datetime.now()

            step.status = status

        if result is not None:
            step.result = result

        if error is not None:
            step.error = error

        plan.updated_at = datetime.now()

        # Check for blocked steps
        self._check_blocked_steps(plan)

        return True

    def _check_blocked_steps(self, plan: Plan):
        """Check if any steps should be marked as blocked."""
        for i, step in enumerate(plan.steps):
            if step.status != StepStatus.NOT_STARTED:
                continue

            # Check if any dependency has failed
            for dep_idx in step.dependencies:
                if plan.steps[dep_idx].status == StepStatus.FAILED:
                    step.status = StepStatus.BLOCKED
                    step.error = f"Blocked due to failed dependency: Step {dep_idx}"
                    break

    def list_plans(self) -> List[Dict[str, Any]]:
        """List all plans."""
        return [
            {
                "plan_id": plan.plan_id,
                "title": plan.title,
                "created_at": plan.created_at.isoformat(),
                "progress": plan.progress
            }
            for plan in self.plans.values()
        ]

    def delete_plan(self, plan_id: str) -> bool:
        """Delete a plan."""
        if plan_id in self.plans:
            del self.plans[plan_id]
            logger.info(f"Deleted plan: {plan_id}")
            return True
        return False


# Global plan manager instance
_plan_manager = PlanManager()


@register_tool()
async def planning_tool(
    action: Literal["create", "update", "list", "get", "mark_step", "delete"],
    title: Optional[str] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    plan_id: Optional[str] = None,
    step_index: Optional[int] = None,
    step_status: Optional[str] = None,
    step_result: Optional[str] = None,
    step_error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Planning tool for managing structured task execution plans.

    Actions:
    - create: Create a new plan with steps
    - update: Update plan metadata
    - list: List all plans
    - get: Get a specific plan
    - mark_step: Update step status/result
    - delete: Delete a plan

    Args:
        action: The action to perform
        title: Plan title (for create)
        steps: List of step definitions (for create)
            Each step should have:
            - description: What to do
            - agent: (optional) Which agent should handle it
            - dependencies: (optional) List of step indices this depends on
        plan_id: ID of the plan to operate on
        step_index: Index of the step to update
        step_status: New status for the step (not_started, in_progress, completed, failed, blocked, skipped)
        step_result: Result of the step execution
        step_error: Error message if step failed

    Returns:
        Result of the action
    """

    if action == "create":
        if not title or not steps:
            return {"error": "Title and steps are required for creating a plan"}

        plan = _plan_manager.create_plan(title, steps, plan_id)
        return {
            "success": True,
            "plan_id": plan.plan_id,
            "message": f"Plan created successfully with ID: {plan.plan_id}",
            "plan": plan.format_display()
        }

    elif action == "list":
        plans = _plan_manager.list_plans()
        return {
            "success": True,
            "plans": plans,
            "count": len(plans)
        }

    elif action == "get":
        if not plan_id:
            return {"error": "plan_id is required"}

        plan = _plan_manager.get_plan(plan_id)
        if not plan:
            return {"error": f"Plan not found: {plan_id}"}

        return {
            "success": True,
            "plan": plan.to_dict(),
            "display": plan.format_display(),
            "next_steps": plan.get_next_steps()
        }

    elif action == "mark_step":
        if not plan_id or step_index is None:
            return {"error": "plan_id and step_index are required"}

        status = None
        if step_status:
            try:
                status = StepStatus(step_status)
            except ValueError:
                return {"error": f"Invalid status: {step_status}"}

        success = _plan_manager.update_step(
            plan_id, step_index,
            status=status,
            result=step_result,
            error=step_error
        )

        if not success:
            return {"error": "Failed to update step"}

        plan = _plan_manager.get_plan(plan_id)
        return {
            "success": True,
            "message": f"Step {step_index} updated successfully",
            "plan": plan.format_display(),
            "next_steps": plan.get_next_steps()
        }

    elif action == "delete":
        if not plan_id:
            return {"error": "plan_id is required"}

        success = _plan_manager.delete_plan(plan_id)
        if not success:
            return {"error": f"Plan not found: {plan_id}"}

        return {
            "success": True,
            "message": f"Plan {plan_id} deleted successfully"
        }

    else:
        return {"error": f"Unknown action: {action}"}


# Additional helper function for creating plans from task analysis
@register_tool()
async def analyze_and_plan(
    task_description: str,
    available_agents: List[str],
    max_steps: int = 10
) -> Dict[str, Any]:
    """
    Analyze a task and create an execution plan.

    This is a helper tool that analyzes the task and suggests a plan structure.
    In a real implementation, this would use an LLM to decompose the task.

    Args:
        task_description: Description of the task to plan
        available_agents: List of available agents
        max_steps: Maximum number of steps to generate

    Returns:
        Suggested plan structure
    """

    # This is a simplified version - in practice, would use LLM for decomposition
    # For now, create a basic template based on task keywords

    steps = []

    # Analyze task for keywords
    task_lower = task_description.lower()

    if "research" in task_lower or "search" in task_lower or "find" in task_lower:
        steps.append({
            "description": "Research and gather information about the topic",
            "agent": "deep_researcher_agent" if "deep_researcher_agent" in available_agents else None
        })

    if "analyze" in task_lower or "calculate" in task_lower or "compute" in task_lower:
        steps.append({
            "description": "Analyze the gathered information",
            "agent": "deep_analyzer_agent" if "deep_analyzer_agent" in available_agents else None,
            "dependencies": [0] if steps else []
        })

    if "web" in task_lower or "website" in task_lower or "browse" in task_lower:
        steps.append({
            "description": "Interact with web pages to extract information",
            "agent": "browser_use_agent" if "browser_use_agent" in available_agents else None
        })

    if "compare" in task_lower or "evaluate" in task_lower:
        steps.append({
            "description": "Compare and evaluate the findings",
            "agent": "deep_analyzer_agent" if "deep_analyzer_agent" in available_agents else None,
            "dependencies": list(range(len(steps)))
        })

    # Always add synthesis and final answer steps
    steps.append({
        "description": "Synthesize all findings into a coherent answer",
        "agent": None,  # Planning agent handles this
        "dependencies": list(range(len(steps)))
    })

    steps.append({
        "description": "Provide the final answer",
        "agent": None,  # Planning agent handles this
        "dependencies": [len(steps)]
    })

    return {
        "success": True,
        "suggested_title": f"Plan for: {task_description[:50]}...",
        "suggested_steps": steps[:max_steps],
        "analysis": {
            "task_type": "research" if "research" in task_lower else "general",
            "complexity": "high" if len(steps) > 5 else "medium" if len(steps) > 3 else "low",
            "recommended_agents": list(set(s.get("agent") for s in steps if s.get("agent")))
        }
    }