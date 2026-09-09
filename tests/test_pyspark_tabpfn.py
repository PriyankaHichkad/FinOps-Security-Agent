import os
import pytest
from src.pyspark_batch import PySparkBatchEngine

def test_pyspark_tabpfn_batch_execution():
    """Verifies PySpark TabPFN batch engine executes cleanly and generates artifact."""
    engine = PySparkBatchEngine()
    result = engine.run_tabpfn_pyspark_batch()
    
    assert result is not None
    assert result["status"] == "SUCCESS"
    assert "model_name" in result
    assert result["recall_at_5_fpr"] >= 0.0
    assert result["pr_auc"] >= 0.0
    assert result["roc_auc"] >= 0.0
    assert result["throughput_items_per_sec"] > 0
    assert os.path.exists("artifacts/pyspark_tabpfn_summary.json")

def test_pyspark_batch_standard_pipeline():
    """Verifies PySpark batch pipeline executes standard batch decisions."""
    engine = PySparkBatchEngine()
    result = engine.run_batch_pipeline()
    assert result["status"] == "SUCCESS"
    assert "verdict_distribution" in result
