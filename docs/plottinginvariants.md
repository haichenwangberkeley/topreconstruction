# HistogramPlottingInvariants.md

This document defines mandatory invariants for all 1D histogram plots.
All coding agents must follow these rules unless explicitly overridden by the user.

These invariants are designed for statistical correctness, visual clarity, and consistency across analyses.

---

# 1. Scope

These rules apply to:

- All 1D histograms
- All overlaid histogram comparisons
- All ratio-panel plots

They apply regardless of plotting backend (ROOT, matplotlib, mplhep, boost-histogram, etc.).

---

# 2. Histogram Styling (No Filled Areas)

- Histograms must NOT use filled colors.
- The area under the histogram must remain empty.
- Histograms must be drawn as line-only (“step”) style.
- Distinction between multiple distributions must be achieved using:
  - Different line colors
  - Different line styles (solid, dashed, dotted, dash-dot, etc.)
- Line width must be clearly visible but not excessive.

Filled stacked histograms are forbidden unless explicitly requested.

---

# 3. Binning Rules

- Default number of bins: 20
- All overlaid histograms must use identical bin edges.
- Binning must be fixed before computing uncertainties and ratios.
- Unless explicitly specified, automatic bin selection must not change between overlaid distributions.

---

# 4. Intelligent X-Axis Range Selection

The x-axis limits must be computed automatically using statistical range:

For each distribution:

    mean_i = mean(distribution_i)
    std_i  = standard_deviation(distribution_i)

Define:

    lower_i = mean_i - 3 * std_i
    upper_i = mean_i + 3 * std_i

Across all distributions:

    global_lower = min(lower_i)
    global_upper = max(upper_i)

The x-axis range MUST be:

    [global_lower, global_upper]

This ensures consistent, statistically meaningful visualization.

---

# 5. Statistical Uncertainties (Mandatory)

## 5.1 Bin-Level Uncertainty Definition

Every histogram MUST display bin-by-bin statistical uncertainties.

For unweighted entries:

    sigma_bin = sqrt(N_bin)

For weighted entries:

    sigma_bin = sqrt(sum_of_weights_squared_in_bin)

Gaussian symmetric uncertainties must be used unless explicitly overridden.

Uncertainties must be computed using the final bin contents (after any normalization).

---

# 6. Normalization Consistency

All histograms within a figure must use consistent normalization:

- Either raw counts
- Or normalized density
- Or area-normalized

If normalization is applied:

- Statistical uncertainties must be scaled consistently.
- Error propagation must use post-normalization values.

The normalization choice must be reflected in axis labeling.

---

# 7. Overlay Behavior

When two or more histograms are drawn:

- A ratio panel MUST be included below the main panel.
- The ratio panel shares the same x-axis.
- The ratio is defined as:

    R_i = A_i / B

Where:
- A_i = bin content of distribution i
- B   = bin content of nominal distribution

If nominal distribution is not specified:
- The FIRST distribution provided is the nominal.

---

# 8. Ratio Panel Requirements

## 8.1 Ratio Axis Range

The ratio panel y-axis range MUST be fixed to:

    [0.5, 1.5]

Values outside this range may be clipped.

A horizontal reference line at R = 1.0 must be drawn.

---

## 8.2 Statistical Error Propagation in Ratio (Mandatory)

Statistical uncertainties MUST be propagated assuming independence between numerator and denominator.

For:

    R = A / B

The propagated uncertainty must be:

    sigma_R = R * sqrt( (sigma_A / A)^2 + (sigma_B / B)^2 )

Where:
- A = numerator bin content
- B = nominal bin content
- sigma_A = statistical uncertainty of A
- sigma_B = statistical uncertainty of B

This formula assumes statistical independence between A and B.

---

## 8.3 Special Cases

If B = 0:
- The ratio must NOT be computed.
- The bin must be masked or omitted in the ratio panel.
- No NaN or infinite values are allowed.

If A = 0 and B ≠ 0:
- Ratio = 0
- Uncertainty must be computed safely without division errors.

Division-by-zero must never produce undefined plot values.

---

## 8.4 Nominal Reference in Ratio Panel

The nominal distribution must appear in the ratio panel as:

- A reference band centered at 1.0
- With its propagated statistical uncertainty

The nominal must not be silently omitted.

---

# 9. Error Bar Visibility

- All histograms must visibly display error bars.
- Error bars must not be hidden behind lines.
- Marker-based histograms must include visible uncertainty bars.
- Line-only histograms must include overlaid uncertainty bars.

Statistical uncertainty must never be omitted unless explicitly requested.

---

# 10. Backend Independence

These invariants are backend-agnostic.

Whether implemented in:
- ROOT
- matplotlib
- mplhep
- boost-histogram
- any other plotting framework

The statistical and stylistic rules above remain mandatory.

---

# 11. Override Policy

These invariants may only be modified if:

- The user explicitly overrides a specific rule.
- The plot type is not a 1D histogram.

Otherwise, all invariants are mandatory.

---

End of specification.