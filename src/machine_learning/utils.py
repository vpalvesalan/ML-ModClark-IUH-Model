import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

def get_linear_model_formula(pipeline: Pipeline, target_name: str) -> str:
    """
    Extracts parameters from a fitted scikit-learn pipeline with a linear model
    and returns a formatted string of its mathematical formula.

    Args:
        pipeline (Pipeline): The fitted scikit-learn pipeline object.
        target_name (str): The name of the target variable (e.g., 'TC').

    Returns:
        str: A markdown formatted string of the model's equation.
    """
    # Verify the model is linear and has coefficients
    if not hasattr(pipeline.named_steps.get('regressor'), 'coef_'):
        regressor_name = type(pipeline.named_steps.get('regressor')).__name__
        return f"Cannot generate formula: The model '{regressor_name}' is not a linear model."

    # --- 1. Extract Components from Pipeline ---
    preprocessor = pipeline.named_steps.get('preprocessor')
    regressor = pipeline.named_steps.get('regressor')
    
    feature_transformer = preprocessor.named_steps.get('feature_transformer')
    scaler = preprocessor.named_steps.get('scaler')

    intercept = regressor.intercept_
    coeffs = regressor.coef_
    
    # Get parameters from the StandardScaler
    scaler_means = scaler.mean_
    scaler_stds = scaler.scale_
    
    # Get feature names after the ColumnTransformer step
    ct_output_names = feature_transformer.get_feature_names_out()

    # --- 2. Build Each Term of the Formula ---
    terms = []
    for i, full_name in enumerate(ct_output_names):
        # Skip features that were eliminated (coefficient is zero)
        if np.isclose(coeffs[i], 0):
            continue

        # Get parameters for this specific feature
        coeff = coeffs[i]
        mean = scaler_means[i]
        std = scaler_stds[i]

        # Determine the initial transformation (log, passthrough, etc.)
        parts = full_name.split('__')
        transform_type = parts[0]
        original_name = parts[-1]

        if transform_type == 'log':
            inner_term = f"log({original_name})"
        else:  # This handles 'remainder' or 'passthrough'
            inner_term = original_name
            
        # Construct the full term: Coeff * ( (Transform(x) - Mean) / Std_Dev )
        scaled_term = f"\\frac{{ {inner_term} - {mean:.4f} }}{{ {std:.4f} }}"
        full_term = f"{coeff:+.4f} \\times \\left( {scaled_term} \\right)"
        terms.append(full_term)

    # --- 3. Assemble the Final Markdown String ---
    # The model predicts the log of the target, so we wrap in exp()
    formula_body = " \\\\\n & ".join(terms)

    markdown_string = f"""### {target_name} Prediction Formula

The model predicts the **natural logarithm** of **{target_name}**. To get the final predicted value, you must apply the exponential function `exp()` to the equation's result.

$${target_name}_{{predicted}} = \\exp \\left( Intercept + \\sum (Coefficient \\times Transformed\\_Feature) \\right)$$

---
### Detailed Equation

$${target_name}_{{predicted}} = \\exp \\left( {intercept:.4f} \\\\\n & {formula_body} \\right)$$

Where:
- **`log()`**: Represents the natural logarithm.
- Each feature is first transformed (if applicable), then standardized using its specific mean ($$\\mu$$) and standard deviation ($$\\sigma$$) derived from the training data.
"""
    return markdown_string