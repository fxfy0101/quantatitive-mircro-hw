# %%c
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from stargazer.stargazer import Stargazer
# %%
df = pd.read_stata(r"/Users/yong/Documents/Data/NELS_1988-00_v1_0_Stata_Datasets/NELS_88_00_BYF4STU_V1_0.dta")
# %%
## reconstruct dataset in assignment 1
df1 = df[df.F4UNI2A == 1]
variables = {"BYS31A": "race", "BYS12": "gender", "BYS34B": "motheduc", "BYFAMINC": "faminc", "BY2XCOMP": "score"}
df2 = df1[["STU_ID", "BYS31A", "BYS12", "BYS34B", "BYFAMINC", "BY2XCOMP"]]
nels_88 = df2.rename(variables, axis=1)
nels_88.score.replace([99.98, 99.99, -9.00], np.nan, inplace=True)
### race dummies
race_dummies = pd.get_dummies(nels_88.race.map({1: "race_asian", 2: "race_hispanic", 3: "race_black", 
                                                4: "race_white", 5: "race_american_indian", 6: "race_unknown", 
                                                7: "race_unknown", 8: "race_missing", 9: "race_unknown"}))
### gender dummies
gender_dummies = pd.get_dummies(nels_88.gender.map({1: "gender_male", 2: "gender_female", 7: "gender_missing", 
                                                    8: "gender_missing", 9: "gender_missing"}))
### mother's education dummies
motheduc_dummies = pd.get_dummies(nels_88.motheduc.map({1: "motheduc_less_hs_degree", 2: "motheduc_hs_degree", 
                                                        3: "motheduc_some_college", 4: "motheduc_some_college", 
                                                        5: "motheduc_fouryr_degree", 6: "motheduc_graduate_degree", 
                                                        7: "motheduc_graduate_degree", 8: "motheduc_unknown", 
                                                        97: "motheduc_unknown", 98: "motheduc_missing", 99: "motheduc_unknown"}))
### family income dummies
faminc_dummies = pd.get_dummies(nels_88.faminc.map({1: "inc_less_10", 2: "inc_less_10", 3: "inc_less_10", 4: "inc_less_10", 
                                                    5: "inc_less_10", 6: "inc_less_10", 7: "inc_10_20", 8: "inc_10_20", 
                                                    9: "inc_20_35", 10: "inc_20_35", 11: "inc_35_50", 12: "inc_50_75", 
                                                    13: "inc_75_100", 14: "inc_100_200", 15: "inc_more_200", 
                                                    98: "inc_missing", 99: "inc_unknown"}))
nels_88 = pd.concat([nels_88, race_dummies, gender_dummies, motheduc_dummies, faminc_dummies], axis=1)
nels_88 = nels_88.drop(["race", "gender", "motheduc", "faminc"], axis=1)
# %%
### language indicator
### BYS18: first language spoke
### BYS27B: how well do you speak english
languages = df[["STU_ID", "BYS18", "BYS27B"]].copy()
### 98 missing, drop them; 8 missing, drop them
languages = languages[~((languages.BYS18 == 98) | (languages.BYS27B == 8))]
languages["eng_well"] = languages.BYS27B.apply(lambda x: 1 if x == 1 or x == 9 else 0)
languages["eng"] = languages.BYS18.apply(lambda x: 1 if x == 1 or x == 99 else 0)
# %%
### final dataset
iv_data = pd.merge(nels_88, languages, how="left", on="STU_ID")
# %%
### part 1
### simple OLS
sols_score = smf.ols("score ~ eng_well", data=iv_data).fit()
sols_score.summary()
# %%
### part 1
### instrument variable
siv_score = IV2SLS.from_formula("score ~ 1 + [eng_well ~ eng]", data=iv_data).fit()
siv_score.summary
# %%
### part 1
### instrument variable
### first stage regression
siv_score.first_stage
# %%
### part 2
### OLS
### base group: race_black, gender_male, motheduc_less_hs_degree, inc_less_10
ols_formula = "score ~ eng_well + race_asian + race_hispanic + race_white + race_american_indian + race_missing + race_unknown + gender_female + gender_missing + motheduc_hs_degree + motheduc_some_college + motheduc_fouryr_degree + motheduc_graduate_degree + motheduc_missing + motheduc_unknown + inc_10_20 + inc_20_35 + inc_35_50 + inc_50_75 + inc_75_100 + inc_100_200 + inc_more_200 + inc_missing"
ols_score = smf.ols(ols_formula, data=iv_data).fit()
ols_score.summary()
# %%
### part 2
### instrument variable
iv_formula = "score ~ 1 + [eng_well ~ eng] + race_asian + race_hispanic + race_white + race_american_indian + race_missing + race_unknown + gender_female + gender_missing + motheduc_hs_degree + motheduc_some_college + motheduc_fouryr_degree + motheduc_graduate_degree + motheduc_missing + motheduc_unknown + inc_10_20 + inc_20_35 + inc_35_50 + inc_50_75 + inc_75_100 + inc_100_200 + inc_more_200 + inc_missing"
iv_score = IV2SLS.from_formula(iv_formula, data=iv_data).fit()
iv_score.summary
# %%
### first stage regression
iv_score.first_stage