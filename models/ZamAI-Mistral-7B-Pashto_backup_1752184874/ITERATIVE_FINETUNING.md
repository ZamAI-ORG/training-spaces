# Iterative Fine-tuning Process for ZamAI-Mistral-7B-Pashto

This document outlines a systematic approach to iteratively improve your Pashto language model through multiple fine-tuning cycles. Each iteration builds on insights from the previous one to create a progressively better model.

## The Iterative Workflow

```
┌─────────────────┐
│                 │
│  Initial Setup  │
│                 │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│                 │
│ Prepare Dataset ├────┐
│                 │    │
└────────┬────────┘    │
         │             │
         ▼             │
┌─────────────────┐    │
│                 │    │
│   Fine-tune     │    │
│                 │    │
└────────┬────────┘    │
         │             │
         ▼             │
┌─────────────────┐    │
│                 │    │
│    Evaluate     │    │
│                 │    │
└────────┬────────┘    │
         │             │
         ▼             │
┌─────────────────┐    │
│  Analyze and    │    │
│ Identify Issues ├────┘
└─────────────────┘
```

## Iteration 1: Initial Fine-tuning

**Goal**: Establish a baseline model and identify key performance issues.

1. **Dataset Preparation**:
   - Run `prepare_for_autotrain.py` with default parameters
   - Use the instruction-response format for first iteration

2. **Fine-tuning**:
   - Use `autotrain_finetune.py` with:
     - Learning rate: 2e-4
     - Epochs: 3
     - LoRA r: 16
     - LoRA alpha: 32

3. **Evaluation**:
   - Run `evaluate_and_iterate.py` on 20+ diverse samples
   - Document performance in key areas (completion quality, language accuracy)

4. **Analysis**:
   - Identify the most obvious issues (language mixing, short responses, etc.)
   - Determine which dataset aspects need improvement

## Iteration 2: Targeted Improvements

**Goal**: Address the major issues identified in the first iteration.

1. **Dataset Updates**:
   - Add more examples in underrepresented categories
   - Clean existing examples with issues
   - Consider experimenting with different formats (text vs instruction-response)

2. **Fine-tuning Adjustments**:
   - Adjust learning rate based on first run (increase if underfitting, decrease if overfitting)
   - Potentially increase epochs to 4-5 if needed

3. **Evaluation**:
   - Compare new model against the baseline
   - Identify if the targeted issues show improvement
   - Document new issues that emerge

## Iteration 3: Parameter Optimization

**Goal**: Fine-tune the training parameters for optimal performance.

1. **Parameter Experiments**:
   - Try different LoRA configurations:
     - Higher rank (24 or 32) for more capacity
     - Different target modules if certain outputs show weakness
   - Experiment with batch size and optimization parameters

2. **Focused Evaluation**:
   - Test specifically on challenging examples
   - Evaluate language consistency in longer responses
   - Measure performance improvements against previous iterations

## Iteration 4: Dataset Expansion

**Goal**: Expand the model's capabilities to handle a wider range of content.

1. **Dataset Expansion**:
   - Add examples in different domains based on evaluation gaps
   - Create specialized test sets for different use cases
   - Consider augmenting data with variations of successful examples

2. **Advanced Fine-tuning**:
   - Consider mixed-precision training for efficiency
   - Experiment with different weights for different example types

## Measuring Progress

For each iteration, track these key metrics:

1. **Output Quality**: 
   - Coherence and relevance of generated text
   - Grammar and spelling accuracy

2. **Language Consistency**: 
   - Consistent use of Pashto throughout responses
   - Handling of code-switching if relevant to your use case

3. **Specific Task Performance**:
   - Performance on targeted tasks (translation, question answering, etc.)
   - Ability to follow complex instructions

4. **Technical Metrics**:
   - Training loss curves
   - Generation speed
   - Memory usage

## When to Stop Iterating

Consider your fine-tuning process complete when:

1. The model meets your quality thresholds for your target use cases
2. Successive iterations show diminishing returns in improvement
3. The most important failure cases have been addressed

Remember that perfection is not the goal - a useful, reliable model for your specific needs is!

## Additional Resources

- Use our `check_dataset_format.py` to validate your dataset before each iteration
- Maintain a versioning system for your models to track progress
- Document training settings and results for each iteration
