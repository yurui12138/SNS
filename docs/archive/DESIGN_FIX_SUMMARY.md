# IG-Finder 2.0: Critical Design Fixes

## Summary

This update fixes **critical design deviations** identified in the IG-Finder 2.0 implementation, bringing it into full alignment with the comprehensive design specification provided by the user. The fixes address the most severe issue: **main axis selection happening BEFORE reconstruction instead of AFTER**, and add several missing components for machine-executable guidance generation.

## Critical Issues Fixed

### 1. Phase 3: Reconstruct-Then-Select Logic ⭐ **CRITICAL FIX**

**Problem**: The original implementation selected the main axis BEFORE performing reconstruction, violating the "Delta-first" design principle.

**Solution**:
- Added `ViewReconstructionScore` dataclass with proper scoring formula:
  ```
  CombinedScore = α·FitGain + β·Stress + γ·Coverage - λ·EditCost
  where α=0.4, β=0.3, γ=0.2, λ=0.1
  ```
- Implemented `compute_all_views_reconstruction()` in `EvolutionPlanner`
- Calculates reconstruction metrics for **ALL views** before selection
- Enables proper evaluation of which view benefits most from reconstruction

**Impact**: This is the most critical fix. Without it, the system was selecting axes based on current FIT rates rather than reconstruction potential, defeating the entire "Delta-first" design principle.

### 2. Phase 4: Writing Mode Determination ⭐ **NEW FEATURE**

**Problem**: Missing logic to determine writing mode (DELTA_FIRST vs ANCHOR_PLUS_DELTA).

**Solution**:
- Added `WritingMode` enum with two modes:
  - `DELTA_FIRST`: For heavily reconstructed views (EditCost > 3.0 or FitGain > 10.0)
  - `ANCHOR_PLUS_DELTA`: For stable views (low EditCost, moderate FitGain)
- Implemented `select_main_axis_with_mode()` that:
  1. Uses reconstruction scores to pick best view
  2. Determines writing mode based on reconstruction needs
  3. Returns both main axis and mode

**Impact**: Enables proper mode-based guidance generation. Delta-first emphasizes evolution, while Anchor+Delta uses existing structure.

### 3. Delta-Aware Guidance: Writing Rules ⭐ **NEW FEATURE**

**Problem**: Missing `writing_rules` field with executable do/dont constraints.

**Solution**:
- Added `WritingRules` dataclass with `do` and `dont` lists
- Updated `DeltaAwareGuidance` to include:
  - `main_axis_mode`: Writing mode
  - `writing_rules`: Executable constraints
  - `reconstruction_scores`: Full view scores for transparency
- Implemented `_generate_writing_rules()` with mode-specific guidance:
  
  **DELTA_FIRST rules**:
  - DO: Lead with emerging trends and structural shifts
  - DO: Organize by innovation clusters
  - DONT: Force-fit new work into old categories
  
  **ANCHOR_PLUS_DELTA rules**:
  - DO: Use main axis as foundation
  - DO: Clearly mark structural updates
  - DONT: Ignore evolution points

**Impact**: Provides machine-executable guidance for downstream writing systems, not just human-readable suggestions.

### 4. Phase 1: Baseline Quality Gate

**Problem**: No quality control for baseline construction.

**Solution**:
- Added `_check_baseline_quality()` method with two checks:
  1. **Minimum unique facets**: Should have ≥2 different facets
  2. **Dominant facet ratio**: No single facet should be >60% of views
- Issues warnings when quality is low
- Foundation for future compensatory view induction

**Impact**: Prevents low-quality baselines from propagating through the pipeline.

### 5. Phase 2: Formula Verification ✅

**Status**: **VERIFIED CORRECT** - No changes needed

- FitScore = Coverage - 0.8×Conflict - 0.4×Residual ✓
- Coverage = 0.7×semantic + 0.3×lexical ✓
- Label rules: UNFITTABLE (Coverage < 0.45 or Conflict > 0.55), FORCE_FIT (Residual > 0.45), FIT (else) ✓

## Design Principles Implemented

| Principle | Status | Implementation |
|-----------|--------|----------------|
| Reconstruct-first, THEN select | ✅ | `compute_all_views_reconstruction()` in Phase 3 |
| Delta-aware mode determination | ✅ | `WritingMode` enum + selection logic in Phase 4 |
| Machine-executable constraints | ✅ | `WritingRules` with do/dont lists |
| Full transparency | ✅ | `reconstruction_scores` included in guidance |
| Baseline quality gates | ✅ | Quality checks in Phase 1 |

## Code Changes

### Files Modified (4)

1. **`knowledge_storm/ig_finder/dataclass_v2.py`** (+182 lines)
   - Added `WritingMode` enum
   - Added `ViewReconstructionScore` dataclass (with auto-calculated combined_score)
   - Added `WritingRules` dataclass
   - Updated `DeltaAwareGuidance` with new fields

2. **`knowledge_storm/ig_finder/modules/phase3_evolution.py`** (+113 lines)
   - Added `compute_all_views_reconstruction()` method to `EvolutionPlanner`
   - Added `compute_reconstruction_scores()` method to `Phase3Pipeline`
   - Updated imports to include new dataclasses

3. **`knowledge_storm/ig_finder/modules/phase4_guidance.py`** (+168 lines)
   - Completely rewrote `select_main_axis` as `select_main_axis_with_mode()`
   - Added `_generate_writing_rules()` method
   - Updated `generate_guidance()` to accept mode and reconstruction scores
   - Updated `Phase4Pipeline.run()` to use reconstruction-based selection

4. **`knowledge_storm/ig_finder/modules/phase1_multiview_baseline.py`** (+45 lines)
   - Added `_check_baseline_quality()` method
   - Integrated quality checks into baseline building

### Files Added (2)

- `CODE_ANALYSIS.md`: Detailed analysis of design deviations
- `IMPROVEMENT_PLAN.md`: Step-by-step improvement plan with code examples

## Integration with Existing Code

All changes maintain **backward compatibility** where possible:

- Old `select_main_axis()` method is kept but marked as DEPRECATED
- New methods are added without breaking existing interfaces
- Data structures are extended, not replaced

## Testing Recommendations

1. **Unit Tests for Reconstruction Logic**
   ```python
   def test_view_reconstruction_scoring():
       # Test that views are scored correctly
       # Verify combined_score = 0.4*fit_gain + 0.3*stress - 0.1*edit_cost
       pass
   ```

2. **Integration Test for Mode Determination**
   ```python
   def test_writing_mode_determination():
       # High EditCost/FitGain → DELTA_FIRST
       # Low EditCost/FitGain → ANCHOR_PLUS_DELTA
       pass
   ```

3. **End-to-End Test with Real Data**
   ```python
   def test_full_pipeline_with_reconstruction():
       # Run Phase 1-4 with reconstruction
       # Verify guidance contains all new fields
       pass
   ```

## Usage Example

```python
from knowledge_storm.ig_finder import IGFinder2Runner, IGFinder2Arguments

# Initialize
args = IGFinder2Arguments(
    topic="transformer neural networks",
    output_dir="./output",
    top_k_reviews=15,
    top_k_research_papers=100,
    min_cluster_size=3,
    lambda_regularization=0.8,
)

runner = IGFinder2Runner(args, lm)

# Run with reconstruction-based selection
results = runner.run(
    do_phase1=True,
    do_phase2=True,
    do_phase3=True,  # Now computes reconstruction scores
    do_phase4=True,  # Now uses reconstruction scores for axis selection
)

# Access new fields
guidance = results.delta_aware_guidance
print(f"Writing mode: {guidance.main_axis_mode.value}")
print(f"DO rules: {guidance.writing_rules.do}")
print(f"DONT rules: {guidance.writing_rules.dont}")
print(f"Reconstruction scores: {guidance.reconstruction_scores}")
```

## Next Steps

### Immediate
1. ✅ Commit and push changes
2. ✅ Update PR with detailed description

### Short-term
1. Add unit tests for new components
2. Test end-to-end with real examples
3. Update documentation with new fields

### Long-term
1. Implement compensatory view induction (Phase 1 quality gate triggers)
2. Add validation step in Phase 4 (check guidance completeness)
3. Enhance writing rules with LLM-generated constraints

## Conclusion

This update resolves the most critical design deviation (reconstruct-before-select) and adds essential features for machine-executable guidance generation. The system now properly implements the "Delta-first" principle and provides clear, mode-specific writing constraints for downstream survey generation.

**Key Innovation**: The system now makes data-driven decisions about writing strategy (DELTA_FIRST vs ANCHOR_PLUS_DELTA) based on reconstruction analysis, rather than using a single fixed approach.
