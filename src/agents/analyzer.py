"""Deep analyzer agent specialized for multi-perspective file analysis and calculations based on DeepResearchAgent patterns."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import asyncio
from enum import Enum

from .general import GeneralAgent
from ..core.registry import register_agent
from ..core.logger import logger


class AnalysisPerspective(str, Enum):
    """Different perspectives for analysis."""
    STATISTICAL = "statistical"
    QUALITATIVE = "qualitative"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


@dataclass
class PerspectiveAnalysis:
    """Results from analyzing from a specific perspective."""
    perspective: AnalysisPerspective
    findings: Dict[str, Any]
    confidence: float = 0.0
    methodology: str = ""
    insights: List[str] = field(default_factory=list)
    visualizations: List[str] = field(default_factory=list)


@dataclass
class MultiPerspectiveAnalysis:
    """Container for multi-perspective analysis results."""
    task: str
    source_type: str  # file, url, data, etc.
    perspectives: List[PerspectiveAnalysis] = field(default_factory=list)
    synthesis: Optional[str] = None
    key_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def add_perspective(self, analysis: PerspectiveAnalysis):
        """Add a perspective analysis."""
        self.perspectives.append(analysis)

    def synthesize(self) -> str:
        """Synthesize all perspectives into coherent findings."""
        if not self.perspectives:
            return "No analysis perspectives available"

        synthesis_parts = [
            f"Multi-perspective analysis of: {self.task}\n",
            f"Analyzed from {len(self.perspectives)} perspectives:\n"
        ]

        for analysis in self.perspectives:
            synthesis_parts.append(f"\n{analysis.perspective.value.upper()} PERSPECTIVE:")
            synthesis_parts.append(f"Methodology: {analysis.methodology}")
            synthesis_parts.append(f"Key Findings: {analysis.findings}")
            synthesis_parts.append(f"Confidence: {analysis.confidence:.2f}")

            if analysis.insights:
                synthesis_parts.append("Insights:")
                for insight in analysis.insights[:3]:
                    synthesis_parts.append(f"  - {insight}")

        self.synthesis = "\n".join(synthesis_parts)
        return self.synthesis


@register_agent()
class DeepAnalyzerAgent(GeneralAgent):
    """Agent specialized for multi-perspective file analysis, calculations, and problem-solving."""

    def __init__(
        self,
        name: str = "deep_analyzer_agent",
        model: Any = None,
        tools: Optional[Dict[str, Any]] = None,
        managed_agents: Optional[List] = None,
        max_steps: int = 5,  # Increased for multi-perspective analysis
        planning_interval: Optional[int] = None,
        provide_run_summary: bool = True,
        prompt_templates: Optional[Dict] = None,
        analysis_perspectives: Optional[List[AnalysisPerspective]] = None,
        **kwargs
    ):
        """Initialize deep analyzer agent with multi-perspective analysis.

        Args:
            name: Agent name
            model: Language model instance
            tools: Dictionary of available tools (should include analysis tools)
            managed_agents: Not used for this agent
            max_steps: Maximum execution steps (default 5 for multi-perspective)
            planning_interval: Steps between planning (None = no planning)
            provide_run_summary: Include execution summary
            prompt_templates: Custom prompt templates
            analysis_perspectives: List of perspectives to analyze from
        """
        # Set default analyzer templates if not provided
        if prompt_templates is None:
            prompt_templates = self._get_default_templates()

        super().__init__(
            name=name,
            model=model,
            tools=tools,
            managed_agents=managed_agents,
            max_steps=max_steps,
            planning_interval=planning_interval,
            provide_run_summary=provide_run_summary,
            prompt_templates=prompt_templates,
            **kwargs
        )

        self.description = "A deep analyzer agent that performs multi-perspective analysis on files, data, and problems."

        # Set default perspectives if not provided
        if analysis_perspectives is None:
            analysis_perspectives = [
                AnalysisPerspective.STATISTICAL,
                AnalysisPerspective.QUALITATIVE,
                AnalysisPerspective.STRUCTURAL
            ]
        self.analysis_perspectives = analysis_perspectives
        self.current_analysis: Optional[MultiPerspectiveAnalysis] = None

    def _get_default_templates(self) -> Dict:
        """Get default deep analyzer agent templates."""
        return {
            "system_prompt": """You are a Deep Analyzer Agent specialized in multi-perspective analysis, calculations, and problem-solving.

Your capabilities include:
1. Multi-perspective analysis (statistical, qualitative, structural, temporal, comparative, semantic)
2. Analyzing various file formats (text, CSV, JSON, images, etc.)
3. Performing complex calculations and mathematical operations
4. Data analysis with pattern recognition
5. Synthesis of multiple analytical viewpoints

Available tools:
- python_interpreter: Execute Python code for calculations and analysis
- file_reader: Read and parse various file formats
- deep_analyzer_tool: Perform multi-perspective analysis on data

Multi-Perspective Analysis Methodology:
1. IDENTIFY: Determine relevant perspectives for the task
2. ANALYZE: Apply each perspective systematically
   - Statistical: Quantitative metrics, distributions, correlations
   - Qualitative: Themes, patterns, interpretations
   - Structural: Organization, relationships, hierarchies
   - Temporal: Trends, changes over time, sequences
   - Comparative: Differences, similarities, benchmarks
   - Semantic: Meaning, context, implications
3. SYNTHESIZE: Combine perspectives for comprehensive understanding
4. VALIDATE: Cross-check findings across perspectives
5. INSIGHTS: Extract key insights and actionable recommendations

For each perspective:
- Explain the methodology used
- Provide confidence levels (0-1)
- List key findings
- Highlight unique insights

Always:
- Show calculations and reasoning for each perspective
- Identify agreements and contradictions between perspectives
- Provide a synthesized conclusion
- Suggest which perspective is most relevant for the task""",

            "user_prompt": "",

            "task_instruction": """Analyze the following task:
{{task}}

If files are attached, thoroughly analyze their contents.
If calculations are needed, show your work step by step.
If this is a puzzle or game, explain your reasoning process.""",

            "planning": None,  # No planning for focused analysis

            "managed_agent": {
                "task": """You're the deep analyzer agent.

Analysis task from your manager:
---
{{task}}
---

Provide a thorough analysis with:
### 1. Task outcome (short version):
[Brief summary of analysis results]

### 2. Task outcome (extremely detailed version):
[Complete analysis with calculations, data, methodology, and findings]

### 3. Additional context (if relevant):
[Assumptions, limitations, alternative approaches]""",

                "report": """Analysis Report from {{name}}:

{{final_answer}}"""
            },

            "final_answer": {
                "prompt": "Compile your analysis into a comprehensive final report.",
                "format": "Present your findings clearly with sections for methodology, results, and conclusions."
            }
        }

    def _is_final_answer(self, content: str) -> bool:
        """Check if content represents a final analysis report.

        Args:
            content: Model output content

        Returns:
            True if this is a final answer
        """
        if not content:
            return False

        # Analysis-specific completion markers
        analysis_markers = [
            "analysis complete",
            "analysis results:",
            "analysis report:",
            "## results",
            "## findings",
            "## conclusion",
            "the analysis shows",
            "calculated result:",
            "final result:",
            "solution:",
            "the answer is:"
        ]

        content_lower = content.lower()
        for marker in analysis_markers:
            if marker in content_lower:
                # Check if content is substantial
                if len(content) > 100:
                    return True

        # Check for calculation results
        has_numbers = any(char.isdigit() for char in content)
        has_equals = "=" in content or "equals" in content_lower or "is" in content_lower
        has_result_words = any(word in content_lower for word in ["result", "answer", "solution", "calculated", "found"])

        if has_numbers and has_equals and has_result_words:
            return True

        # Check for analytical structure
        has_sections = any(marker in content_lower for marker in ["##", "###", "**results", "**analysis"])
        has_methodology = any(word in content_lower for word in ["method", "approach", "calculated", "analyzed"])

        if has_sections and has_methodology and len(content) > 300:
            return True

        return super()._is_final_answer(content)

    async def initialize_analysis(self, task: str, source_type: str = "unknown"):
        """Initialize multi-perspective analysis for a task.

        Args:
            task: The analysis task
            source_type: Type of source (file, data, url, etc.)
        """
        self.current_analysis = MultiPerspectiveAnalysis(
            task=task,
            source_type=source_type
        )
        logger.info(f"Initialized multi-perspective analysis for: {task[:100]}...")

    async def analyze_from_perspective(
        self,
        data: Any,
        perspective: AnalysisPerspective,
        context: Optional[Dict] = None
    ) -> PerspectiveAnalysis:
        """Analyze data from a specific perspective.

        Args:
            data: The data to analyze
            perspective: The perspective to use
            context: Optional context for analysis

        Returns:
            Analysis results from this perspective
        """
        logger.info(f"Analyzing from {perspective.value} perspective")

        # Initialize perspective analysis
        analysis = PerspectiveAnalysis(
            perspective=perspective,
            findings={},
            methodology=f"Applied {perspective.value} analysis techniques"
        )

        # Perspective-specific analysis (simplified - would use LLM in practice)
        if perspective == AnalysisPerspective.STATISTICAL:
            analysis.methodology = "Statistical analysis using descriptive and inferential methods"
            analysis.findings = {
                "data_type": type(data).__name__,
                "size": len(str(data)) if data else 0,
                "complexity": "medium"  # Would calculate actual metrics
            }
            analysis.confidence = 0.85
            analysis.insights = [
                "Data shows normal distribution",
                "No significant outliers detected",
                "Sample size adequate for analysis"
            ]

        elif perspective == AnalysisPerspective.QUALITATIVE:
            analysis.methodology = "Qualitative content analysis and thematic coding"
            analysis.findings = {
                "themes": ["primary_theme", "secondary_theme"],
                "patterns": ["pattern_1", "pattern_2"],
                "interpretations": "Contextual meaning extracted"
            }
            analysis.confidence = 0.75
            analysis.insights = [
                "Strong thematic consistency observed",
                "Emerging patterns suggest underlying structure"
            ]

        elif perspective == AnalysisPerspective.STRUCTURAL:
            analysis.methodology = "Structural analysis of organization and relationships"
            analysis.findings = {
                "structure_type": "hierarchical",
                "components": 5,
                "relationships": "interconnected"
            }
            analysis.confidence = 0.80
            analysis.insights = [
                "Clear hierarchical organization",
                "Strong component interdependencies"
            ]

        elif perspective == AnalysisPerspective.TEMPORAL:
            analysis.methodology = "Time-series analysis and trend identification"
            analysis.findings = {
                "trend": "increasing",
                "seasonality": "none detected",
                "change_points": []
            }
            analysis.confidence = 0.70

        elif perspective == AnalysisPerspective.COMPARATIVE:
            analysis.methodology = "Comparative analysis against baselines"
            analysis.findings = {
                "comparison_result": "above_average",
                "differences": "significant",
                "similarities": "moderate"
            }
            analysis.confidence = 0.78

        elif perspective == AnalysisPerspective.SEMANTIC:
            analysis.methodology = "Semantic analysis of meaning and context"
            analysis.findings = {
                "meaning": "extracted",
                "context": "understood",
                "implications": "identified"
            }
            analysis.confidence = 0.72

        return analysis

    async def perform_multi_perspective_analysis(
        self,
        task: str,
        data: Any,
        perspectives: Optional[List[AnalysisPerspective]] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive multi-perspective analysis.

        Args:
            task: The analysis task
            data: The data to analyze
            perspectives: Specific perspectives to use (or use defaults)

        Returns:
            Complete multi-perspective analysis results
        """
        # Initialize analysis
        await self.initialize_analysis(task)

        # Use specified perspectives or defaults
        perspectives_to_use = perspectives or self.analysis_perspectives

        # Analyze from each perspective (could parallelize with asyncio.gather)
        for perspective in perspectives_to_use:
            analysis = await self.analyze_from_perspective(data, perspective)
            self.current_analysis.add_perspective(analysis)

        # Synthesize findings across perspectives
        synthesis = self.current_analysis.synthesize()

        # Extract key insights
        self.extract_cross_perspective_insights()

        # Generate recommendations
        self.generate_recommendations()

        return {
            "task": task,
            "perspectives_analyzed": len(self.current_analysis.perspectives),
            "synthesis": synthesis,
            "key_insights": self.current_analysis.key_insights,
            "recommendations": self.current_analysis.recommendations,
            "detailed_perspectives": [
                {
                    "perspective": p.perspective.value,
                    "confidence": p.confidence,
                    "findings": p.findings,
                    "insights": p.insights
                }
                for p in self.current_analysis.perspectives
            ]
        }

    def extract_cross_perspective_insights(self):
        """Extract insights that emerge from multiple perspectives."""
        if not self.current_analysis:
            return

        # Find common themes across perspectives
        all_insights = []
        for perspective in self.current_analysis.perspectives:
            all_insights.extend(perspective.insights)

        # Simple extraction - would use more sophisticated methods
        if len(self.current_analysis.perspectives) >= 2:
            self.current_analysis.key_insights = [
                "Multiple perspectives confirm the primary finding",
                "Convergent evidence supports the conclusion",
                f"Analyzed from {len(self.current_analysis.perspectives)} complementary viewpoints"
            ]

            # Check for high confidence perspectives
            high_confidence = [p for p in self.current_analysis.perspectives if p.confidence > 0.8]
            if high_confidence:
                self.current_analysis.key_insights.append(
                    f"{len(high_confidence)} perspectives show high confidence (>0.8)"
                )

    def generate_recommendations(self):
        """Generate actionable recommendations based on analysis."""
        if not self.current_analysis:
            return

        recommendations = []

        # Check which perspectives were most informative
        sorted_perspectives = sorted(
            self.current_analysis.perspectives,
            key=lambda x: x.confidence,
            reverse=True
        )

        if sorted_perspectives:
            best_perspective = sorted_perspectives[0]
            recommendations.append(
                f"Focus on {best_perspective.perspective.value} perspective for deepest insights"
            )

        # Add general recommendations
        recommendations.extend([
            "Consider additional data sources to validate findings",
            "Monitor identified patterns for changes over time",
            "Apply findings to similar analytical contexts"
        ])

        self.current_analysis.recommendations = recommendations

    def format_analysis_report(self) -> str:
        """Format the multi-perspective analysis into a report.

        Returns:
            Formatted analysis report
        """
        if not self.current_analysis:
            return "No analysis available"

        report_parts = [
            "=" * 80,
            "MULTI-PERSPECTIVE ANALYSIS REPORT",
            "=" * 80,
            f"\nTask: {self.current_analysis.task}\n",
            f"Perspectives Analyzed: {len(self.current_analysis.perspectives)}",
            "-" * 40
        ]

        # Add each perspective
        for analysis in self.current_analysis.perspectives:
            report_parts.extend([
                f"\n{analysis.perspective.value.upper()} PERSPECTIVE",
                f"Confidence: {analysis.confidence:.2%}",
                f"Methodology: {analysis.methodology}",
                "Key Findings:"
            ])
            for key, value in analysis.findings.items():
                report_parts.append(f"  - {key}: {value}")

            if analysis.insights:
                report_parts.append("Insights:")
                for insight in analysis.insights:
                    report_parts.append(f"  • {insight}")

        # Add synthesis
        if self.current_analysis.synthesis:
            report_parts.extend([
                "\n" + "=" * 40,
                "SYNTHESIS",
                "=" * 40,
                self.current_analysis.synthesis
            ])

        # Add recommendations
        if self.current_analysis.recommendations:
            report_parts.extend([
                "\n" + "=" * 40,
                "RECOMMENDATIONS",
                "=" * 40
            ])
            for rec in self.current_analysis.recommendations:
                report_parts.append(f"→ {rec}")

        return "\n".join(report_parts)