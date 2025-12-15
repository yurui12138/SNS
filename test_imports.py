#!/usr/bin/env python3
"""Test all SNS imports to catch errors early."""

import sys

def test_imports():
    errors = []
    
    # Test 1: Main SNS imports
    try:
        from knowledge_storm.sns import SNSRunner, SNSArguments
        print("✓ Main SNS classes imported successfully")
    except ImportError as e:
        errors.append(f"✗ Main SNS classes: {e}")
    
    # Test 2: Data structures
    try:
        from knowledge_storm.sns import (
            MultiViewBaseline,
            TaxonomyView,
            TaxonomyTree,
            TaxonomyTreeNode,
            FitVector,
            StressCluster,
            EvolutionProposal,
            DeltaAwareGuidance,
        )
        print("✓ Core data structures imported successfully")
    except ImportError as e:
        errors.append(f"✗ Core data structures: {e}")
    
    # Test 3: Phase modules
    try:
        from knowledge_storm.sns.modules import (
            Phase1Pipeline,
            Phase2Pipeline,
            Phase3Pipeline,
            Phase4Pipeline,
        )
        print("✓ Phase modules imported successfully")
    except ImportError as e:
        errors.append(f"✗ Phase modules: {e}")
    
    # Test 4: Evaluation
    try:
        from knowledge_storm.sns.evaluation import (
            TimeSliceEvaluator,
            compute_all_metrics,
            print_metrics_report,
        )
        print("✓ Evaluation modules imported successfully")
    except ImportError as e:
        errors.append(f"✗ Evaluation modules: {e}")
    
    # Test 5: Infrastructure
    try:
        from knowledge_storm.sns import (
            create_embedding_model,
            create_nli_model,
        )
        print("✓ Infrastructure modules imported successfully")
    except ImportError as e:
        errors.append(f"✗ Infrastructure modules: {e}")
    
    if errors:
        print("\n❌ Import test FAILED:")
        for error in errors:
            print(f"  {error}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
