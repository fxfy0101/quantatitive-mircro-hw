# %%
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
score = df[["STU_ID", "BY2XMSTD", "F12XMSTD", "F22XMSTD"]].copy()
score.BY2XMSTD.replace([99.98, 99.99, -9.00], np.nan, inplace=True)
score.F12XMSTD.replace([99.98, 99.99, -9.00], np.nan, inplace=True)
score.F22XMSTD.replace([99.98, 99.99, -9.00], np.nan, inplace=True)
score["gain_1"] = score["F12XMSTD"] - score["BY2XMSTD"]
score["gain_2"] = score["F22XMSTD"] - score["F12XMSTD"]
score = score.drop(["BY2XMSTD", "F12XMSTD", "F22XMSTD"], axis=1)
# %%
### divorce indicators
divorce = df[["STU_ID", "F1S99C", "F2S96B"]].copy()
divorce["yes_1"] = divorce.F1S99C.apply(lambda x: 1 if x == 1 else 0)
divorce["no_1"] = divorce.F1S99C.apply(lambda x: 1 if x == 2 else 0)
divorce["missing_1"] = divorce.F1S99C.apply(lambda x: 1 if x == 8 else 0)
divorce["unknown_1"] = divorce.F1S99C.apply(lambda x: 1 if x == 7 or x == 9 else 0)
divorce["yes_2"] = divorce.F2S96B.apply(lambda x: 1 if x == 1 else 0)
divorce["no_2"] = divorce.F2S96B.apply(lambda x: 1 if x == 2 else 0)
divorce["missing_2"] = divorce.F2S96B.apply(lambda x: 1 if x == 8 else 0)
divorce["unknown_2"] = divorce.F2S96B.apply(lambda x: 1 if x == 7 or x == 9 else 0)
divorce.drop(["F1S99C", "F2S96B"], axis=1, inplace=True)
# %%
### long form panel data, in Python's format requirement
wide_panel = pd.merge(score, divorce, how="left", on="STU_ID")
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
variables = {"BYS31A": "race", "BYS12": "gender"}
nels_88 = df[["STU_ID", "BYS31A", "BYS12"]]
nels_88 = nels_88.rename(variables, axis=1)
nels_88['gender_female'] = nels_88.gender.apply(lambda x: 1 if x == 2 else 0)
nels_88['gender_male'] = nels_88.gender.apply(lambda x: 1 if x == 1 else 0)
nels_88['gender_missing'] = nels_88.gender.apply(lambda x: 1 if x == 7 or x == 8 or x == 9 else 0)
nels_88["race_asian"] = nels_88.race.apply(lambda x: 1 if x == 1 else 0)
nels_88["race_hispanic"] = nels_88.race.apply(lambda x: 1 if x == 2 else 0)
nels_88["race_black"] = nels_88.race.apply(lambda x: 1 if x == 3 else 0)
nels_88["race_white"] = nels_88.race.apply(lambda x: 1 if x == 4 else 0)
nels_88["race_american_indian"] = nels_88.race.apply(lambda x: 1 if x == 5 else 0)
nels_88["race_missing"] = nels_88.race.apply(lambda x: 1 if x == 8 else 0)
nels_88["race_unknown"] = nels_88.race.apply(lambda x: 1 if x == 6 or x == 7 or x == 9 else 0)
race_gender = nels_88.drop(["race", "gender"], axis=1)
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
fixed_effects = PanelOLS(score_gain_panel[["Test_score_gain"]], score_gain_panel[["divorce", "divorce_missing", "divorce_unknown"]]).fit(cov_type="clustered", cluster_entity=True)
print(fixed_effects.summary.as_latex())