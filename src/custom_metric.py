from mlflow.metrics import make_metric
from mlflow.metrics.genai.genai_metric import _get_aggregate_results
from mlflow.metrics.base import MetricValue
from mlflow.models import EvaluationMetric
from prometheus_eval import PrometheusEval
from typing import List, Dict, Optional, Any, Literal
from prometheus_eval.litellm import LiteLLM  
import pandas as pd  


def make_prometheus_metric(
    name: str,
    client: LiteLLM,
    grade_template: str,
    grade_rubric: str,
    grading_reference: Optional[Literal["ground_truth", "context", "no_reference"]] = "ground_truth",
    parameters: Optional[Dict[str, Any]] = None,
    aggregations: Optional[List[str]] = None,
    greater_is_better: bool = True,
    metric_metadata: Optional[Dict[str, Any]] = None,
) -> EvaluationMetric:
    
    genai_metric_args = {
        "name": name,
        "grade_template": grade_template,
        "grade_rubric": grade_rubric,
        "model": getattr(client, "name"),
        "parameters": parameters,
        "aggregations": aggregations,
        "greater_is_better": greater_is_better,
        "metric_metadata": metric_metadata,
        # Record the mlflow version for serialization in case the function signature changes later
        "mlflow_version": "2.15.1",
        "fn_name": make_prometheus_metric.__name__,
    }

    aggregations = aggregations or ["mean", "variance", "p90"]
    judge = PrometheusEval(model=client, absolute_grade_template=grade_template)

    def prometheus_eval_fn(
        instructions: pd.Series,
        responses: pd.Series,
        reference_answers: pd.Series,
        retrieved_contexts: pd.Series,
    ) -> MetricValue:
        if grading_reference == "context":
            reference = retrieved_contexts
        elif grading_reference == "ground_truth":
            reference = reference_answers
        else:
            reference = pd.Series([])
        feedbacks, scores = judge.absolute_grade(
            instructions=instructions.values.tolist(),       # user's input
            responses=responses.values.tolist(),             # RAG output
            rubric=grade_rubric,                             # grading rubric
            reference_answers=reference.values.tolist(),     # reference for evaluation 
            params=parameters,                               # evaluator LLM parameters
        )
        aggregate_scores = _get_aggregate_results(scores, aggregations)

        return MetricValue(scores, feedbacks, aggregate_scores)
    
    return make_metric(
        eval_fn=prometheus_eval_fn,
        greater_is_better=greater_is_better,
        name=name,
        metric_metadata=metric_metadata,
        genai_metric_args=genai_metric_args,
    )