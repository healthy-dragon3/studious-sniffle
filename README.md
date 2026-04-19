# Fannie Mae Loan-Level Default & Prepayment Modeling

Loan-level credit and prepayment modeling on 14M+ Fannie Mae mortgages originated in 2019, comparing academic survival analysis against industry-style approaches used by MBS trading desks.

## What this does

Builds four parallel models on the same loan-level data, each predicting either default or prepayment:

1. **Cox proportional hazards (default)** — survival analysis with time-varying covariates
2. **Monthly logistic regression (default)** — each loan-month as an independent Bernoulli trial, closer to how credit teams model CRT/CAS deals in practice
3. **Cox proportional hazards (prepayment)** — with piecewise-linear refi incentive features
4. **Structural S-curve + burnout (prepayment)** — the functional form used by ADCo, Yield Book, and other production prepay models

All four were trained on 2019Q1 and validated out-of-sample on 2019Q2, giving a clean apples-to-apples comparison of predictive power and generalization.

## Data

- Fannie Mae Single-Family Loan Performance Dataset, 2019Q1 and 2019Q2
- 2019Q1: 341,865 loans, 11.4M loan-months, 594 defaults, 274,959 prepayments
- 2019Q2: 406,893 loans, 13.8M loan-months, 549 defaults, 313,881 prepayments
- Covers origination through end of 2024 (most loans observed through the 2020-21 COVID refi wave and into the 2023-24 rate lock-in period)

## Feature engineering

Built from raw loan-month data:
- `spread` — origination rate minus national 30-year fixed rate at origination (from FRED MORTGAGE30US)
- `cvr` — origination rate divided by current national rate (refi incentive ratio)
- `cvr_itm` / `cvr_otm` — piecewise-linear decomposition of cvr into in-the-money (refi incentive) and out-of-the-money (rate lock-in)
- `current_ltv` — loan-to-value ratio evolving monthly as principal pays down, using forward-filled unpaid balance to avoid data leakage on termination rows
- Standard loan covariates: original LTV, FICO, DTI, loan age, state, occupancy

## Final results

Out-of-sample AUC on 2019Q2 loans (trained on 2019Q1):

| Model | Q1 train | Q2 out-of-sample | Generalization delta |
|---|---|---|---|
| Default: Cox v3 (academic) | 0.873 | 0.787 | -0.086 |
| Default: Logistic (industry) | 0.892 | 0.876 | -0.015 |
| Prepay: Cox v3b (academic) | 0.713 | 0.649 | -0.064 |
| Prepay: S-curve (industry) | 0.659 | 0.666 | +0.007 |

## Key finding

The industry-style models generalize meaningfully better than the academic Cox models. The monthly logistic default model transferred from Q1 to Q2 with almost no AUC loss, while Cox v3 lost 8.6 points. The structural S-curve prepayment model actually improved slightly out-of-sample despite lower in-sample fit, which is the textbook bias-variance tradeoff showing up in real data: imposing correct economic structure reduces variance at the cost of some in-sample bias, and pays off on new data.

The estimated S-curve parameters match industry benchmarks closely: turnover rate 0.77% monthly (9.2% CPR annualized), refi S-curve center at 27 basis points of incentive, steep slope of 17.7 indicating sharp borrower response once refi savings materialize.

## Tech stack

- Python 3.12, pandas, NumPy
- `lifelines` for Cox PH fitting
- `scikit-learn` for logistic regression and AUC
- `scipy.optimize` for non-linear S-curve fitting
- `matplotlib` for diagnostic plots
- `pandas-datareader` for FRED mortgage rate pulls

## Repo structure
fnma_modeling/
├── fnma_default_model.ipynb      main notebook, end-to-end
├── 2019Q1_panel.parquet          cleaned Q1 loan-month panel
├── 2019Q2_panel.parquet          cleaned Q2 loan-month panel
├── models/
│   ├── cox_v1_sample.pkl         default: oltv + fico
│   ├── cox_v2_sample.pkl         default: + spread
│   ├── cox_v3_sample.pkl         default: + cvr + current_ltv
│   ├── cox_pm_v1_sample.pkl      prepay: oltv + fico
│   ├── cox_pm_v2_sample.pkl      prepay: + spread
│   ├── cox_pm_v3_sample.pkl      prepay: + cvr + current_ltv (linear)
│   ├── cox_pm_v3b_sample.pkl     prepay: + non-linear cvr_itm / cvr_otm
│   ├── logit_default_final.pkl   monthly default logistic
│   └── scurve_prepay_final.pkl   S-curve + burnout prepay parameters
├── results/
│   ├── final_results.json        Q2 out-of-sample AUCs
│   ├── models_summary_Q1.json    Q1 in-sample fit metrics
│   └── *.csv                     per-model coefficient tables
└── README.md

## Notes and limitations

- Prepayment AUCs around 0.65-0.67 are low relative to production models (ADCo, Yield Book typically 0.80+). The gap comes from several simplifications: no MSA-level house price indices, no calendar-time fixed effects to capture the 2020-21 refi wave's concentrated timing, no pool-level burnout tracking, and no seasonality controls.
- The default competing-risks framework treats prepayment as censoring and vice versa (cause-specific hazards), which is valid when the two risks are conditionally independent given covariates. Joint estimation with unobserved heterogeneity (An & Qi 2012) would be more rigorous but requires custom code beyond the standard lifelines implementation.
- 2019 vintage loans cover a specific economic regime (COVID rate collapse followed by 2023-24 rate spike). Findings may not transfer to materially different rate environments.
- Default event counts are small (under 600 per quarter) because 2019 underwriting was strong and COVID forbearance absorbed much of what would have been foreclosure activity. Standard errors on coefficients reflect this.

## References

- Cox, D.R. (1972). Regression Models and Life-Tables. JRSS-B.
- Richard, S.F. and Roll, R. (1989). Prepayments on Fixed-Rate Mortgage-Backed Securities. JPM.
- An, X. and Qi, L. (2012). Default and Prepayment Modeling in Participating Mortgages. Journal of Banking & Finance.
- Frame, W.S. et al. on OFHEO model performance during 2007-2008.
