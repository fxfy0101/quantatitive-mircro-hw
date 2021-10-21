# %%c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS
from stargazer.stargazer import Stargazer
# %%
df = pd.read_stata(r"/Users/yong/Documents/Data/NELS_1988-00_v1_0_Stata_Datasets/NELS_88_00_BYF4STU_V1_0.dta")
df = df[df.F4UNI2A == 1]
# %%
### score gain calculation
score = df[["BY2XMSTD", "F12XMSTD", "F22XMSTD"]].apply(lambda x: x.replace([99.98, 99.99, -9.00], np.nan))
score["STU_ID"] = df["STU_ID"]
score["gain_1"] = score["F12XMSTD"] - score["BY2XMSTD"]
score["gain_2"] = score["F22XMSTD"] - score["F12XMSTD"]
score = score.drop(["BY2XMSTD", "F12XMSTD", "F22XMSTD"], axis=1)
# %%
### divorce indicators
divorce = df[["STU_ID", "F1S99C", "F2S96B"]].copy()
divorce_1_dummies = pd.get_dummies(divorce.F1S99C.map({1: "yes_1", 2: "no_1", 8: "missing_1", 7: "unknown_1", 9: "unknown_1"}))
divorce_2_dummies = pd.get_dummies(divorce.F2S96B.map({1: "yes_2", 2: "no_2", 8: "missing_2", 7: "unknown_2", 9: "unknown_2"}))
divorce_dummies = pd.concat([divorce, divorce_1_dummies, divorce_2_dummies], axis=1)
divorce_dummies.drop(["F1S99C", "F2S96B"], axis=1, inplace=True)
# %%
### long form panel data, in Python's format requirement
wide_panel = pd.merge(score, divorce_dummies, how="left", on="STU_ID")
long_panel = pd.wide_to_long(wide_panel, ["gain", "yes", "no", "missing", "unknown"], i="STU_ID", j="wave", sep="_").sort_index(level="STU_ID")
score_gain_panel = long_panel.rename({"gain": "Test_score_gain", "yes": "divorce", "missing": "divorce_missing", "unknown": "divorce_unknown"}, axis=1).drop("no", axis=1)
# %%
### race indicators
### 1: asian and pacific islanders, 2: hispanic, 3: black, 4: white, 5: american indian
### 6: multiple responses, 7: refusal, 8: missing, 9: legitimate skip
### categories 6, 7, and 9 are combined into unknown category
### gender indicators
### 1: male, 2: female, 7: refusal, 8: missing, 9: legitimate skip
### categories 7, 8 and 9 are combined into missing category
### could use get_dummies here
race_dummies = pd.get_dummies(df.BYS31A.map({1: "race_asian", 2: "race_hispanic", 3: "race_black", 4: "race_white", 5: "race_american_indian", 6: "race_unknown", 7: "race_unknown", 8: "race_missing", 9: "race_unknown"}))
gender_dummies = pd.get_dummies(df.BYS12.map({1: "gender_male", 2: "gender_female", 7: "gender_missing", 8: "gender_missing", 9: "gender_missing"}))
race_gender = pd.concat([df[["STU_ID"]], gender_dummies, race_dummies], axis=1)
# %%
# pooled OLS data
score_gain_pooled_ols = score_gain_panel.reset_index().drop("wave", axis=1)
score_gain_pooled_ols = pd.merge(score_gain_pooled_ols, race_gender, how="left", on="STU_ID")
# %%
### part i
### simple OLS
ols_model = smf.ols("Test_score_gain ~ divorce + divorce_missing + divorce_unknown + gender_male + gender_missing + race_hispanic + race_black + race_asian + race_american_indian + race_missing + race_unknown", data=score_gain_pooled_ols)
ols_results = ols_model.fit(cov_type="cluster", cov_kwds={"groups": score_gain_pooled_ols.STU_ID[score_gain_pooled_ols.Test_score_gain.notnull()]})
print(Stargazer([ols_results]).render_latex())
# %%
### part ii
### fixed effects
fixed_effects = PanelOLS(score_gain_panel[["Test_score_gain"]], score_gain_panel[["divorce", "divorce_missing", "divorce_unknown"]], entity_effects=True).fit(cov_type="clustered", cluster_entity=True)
print(fixed_effects.summary.as_latex())
