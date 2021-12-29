# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from stargazer.stargazer import Stargazer
# %%
### import dataset
df = pd.read_stata(r"/Users/yong/Documents/Data/NELS_1988-00_v1_0_Stata_Datasets/NELS_88_00_BYF4STU_V1_0.dta")
### include only base-year eligible records
### filter data based on variable F$UNI2A (=1 base-year eligible)
df1 = df[df.F4UNI2A == 1]
# %%
### select variables
### race (BYS31A), gender (BYS12), mother's education (BYS34B), whether had a computer (BYS35H), 
### clothes dryer (BYS35J), microwave oven (BYS35L) in the base year
### family income (BYFAMINC)
### composite standardized test score (BY2XCOMP)
### bygrads

variables = {"BYS31A": "race", "BYS12": "gender", "BYS34B": "motheduc", "BYS35H": "computer", "BYS35J": "dryer", 
             "BYS35L": "microwave", "BYFAMINC": "faminc", "BY2XCOMP": "score", "BYGRADS": "bygrads"}
df2 = df1[["STU_ID", "BYS31A", "BYS12", "BYS34B", "BYS35H", "BYS35J", "BYS35L", "BYFAMINC", "BY2XCOMP", "BYGRADS"]]
nels_88 = df2.rename(variables, axis=1)
# %%
### code indicator variables

### race indicators
### 1: asian and pacific islanders, 2: hispanic, 3: black, 4: white, 5: american indian
### 6: multiple responses, 7: refusal, 8: missing, 9: legitimate skip
### categories 6, 7, and 9 are combined into unknown category
race_dummies = pd.get_dummies(nels_88.race)
race_dummies["race_unknown"] = race_dummies[6] + race_dummies[7] + race_dummies[9]
race_dummies = race_dummies.rename({1: "race_asian", 2: "race_hispanic", 3: "race_black", 4: "race_white",
                                    5: "race_american_indian", 8: "race_missing"}, axis=1).drop([6, 7, 9], axis=1)
# %%
### gender indicators
### 1: male, 2: female, 7: refusal, 8: missing, 9: legitimate skip
### categories 7, 8 and 9 are combined into missing category
gender_dummies = pd.get_dummies(nels_88.gender)
gender_dummies["gender_missing"] = gender_dummies[7] + + gender_dummies[8] + gender_dummies[9]
gender_dummies = gender_dummies.rename({1: "gender_male", 2: "gender_female"}, axis=1).drop([7, 8, 9], axis=1)
# %%
### mother's education
### 1: not finish high school, 2: graduated high school, 3: junior college, 4: college less than 4 yrs
### 5: graduated college, 6: master's degree, 7: phd or equivalent, 8: don't know, 97: refusal
### 98: missing, 99: legitimate skip
### categories 3 and 4 are combined into some college category
### categories 6 and 7 are combined into graduate degree
### categories 8, 97, and 99 are combined into unknown category
motheduc_dummies = pd.get_dummies(nels_88.motheduc)
motheduc_dummies["motheduc_some_college"] = motheduc_dummies[3] + motheduc_dummies[4]
motheduc_dummies["motheduc_graduate_degree"] = motheduc_dummies[6] + motheduc_dummies[7]
motheduc_dummies["motheduc_unknown"] = motheduc_dummies[8] + motheduc_dummies[97] + motheduc_dummies[99]
motheduc_dummies = motheduc_dummies.drop([3, 4, 6, 7, 8, 97, 99], axis=1).rename({1: "motheduc_less_hs_degree", 
                                                                                  2: "motheduc_hs_degree", 
                                                                                  5: "motheduc_fouryr_degree", 
                                                                                  98: "motheduc_missing"},
                                                                                  axis=1)
# %%
### computer
### 1: have, 2: do not have, 6: multiple response, 8: missing, 9: legitimate skip
### categories 6, 8, and 9 are combined into missing category
computer_dummies = pd.get_dummies(nels_88.computer)
computer_dummies["computer_missing"] = computer_dummies[6] + computer_dummies[8] + computer_dummies[9]
computer_dummies = computer_dummies.drop([6, 8, 9], axis=1).rename({1: "have_computer", 2: "no_computer"}, axis=1)
# %%
### clothes dryer
### 1: have, 2: do not have, 6: multiple response, 8: missing, 9: legitimate skip
### categories 6, 8, and 9 are combined into missing category
dryer_dummies = pd.get_dummies(nels_88.dryer)
dryer_dummies["dryer_missing"] = dryer_dummies[6] + dryer_dummies[8] + dryer_dummies[9]
dryer_dummies = dryer_dummies.drop([6, 8, 9], axis=1).rename({1: "have_dryer", 2: "no_dryer"}, axis=1)
# %%
### microwave
### 1: have, 2: do not have, 6: multiple response, 8: missing, 9: legitimate skip
### categories 6, 8, and 9 are combined into missing category
microwave_dummies = pd.get_dummies(nels_88.microwave)
microwave_dummies["microwave_missing"] = microwave_dummies[6] + microwave_dummies[8] + microwave_dummies[9]
microwave_dummies = microwave_dummies.drop([6, 8, 9], axis=1).rename({1: "have_microwave", 2: "no_microwave"}, axis=1)
# %%
### family income
### 1: none, 2: less than $1,000, 3: $1,000-$2,9999, 4: $3,000-$4,9999
### 5: $5,000-$7,499, 6: $7,500-$9,999, 7: $10,000-$14,999
### 8: $15,000-$19,999, 9: $20,000-$24,999, 10: $25,000-$34,999
### 11: $35,000-$49,999, 12: $50,000-$74,999, 13: $75,000-$99,999
### 14: $100,000-199,999, 15: $200,000 OR MORE
### 98: missing, 99: legitimate skip
### categories 2, 3, 4, 5, and 6 are combined into category 10,000 or less
### categories 7 and 8 are combined into 10,000-20,000
### categories 9 and 10 are combined into 20,000-35,000
### categories 1 and 99 are combined into unknown
faminc_dummies = pd.get_dummies(nels_88.faminc)
faminc_dummies["inc_less_10"] = faminc_dummies[2] + faminc_dummies[3] + faminc_dummies[4] + faminc_dummies[5] + faminc_dummies[6]
faminc_dummies["inc_10_20"] = faminc_dummies[7] + faminc_dummies[8]
faminc_dummies["inc_20_35"] = faminc_dummies[9] + faminc_dummies[10]
faminc_dummies["inc_unknown"] = faminc_dummies[1] + faminc_dummies[99]
faminc_dummies = faminc_dummies.rename({11: "inc_35_50", 12: "inc_50_75", 13: "inc_75_100", 14: "inc_100_200", 15: "inc_more_200", 98: "inc_missing"}, axis=1).drop(list(range(1, 11)) + [99], axis=1)
# %%
### combine dataset
nels_88 = pd.concat([nels_88, gender_dummies, race_dummies, motheduc_dummies, 
                     computer_dummies, dryer_dummies, microwave_dummies, faminc_dummies], axis=1)
nels_88 = nels_88.drop(["race", "gender", "motheduc", "computer", "dryer", "microwave", "faminc"], axis=1)
# %%
### identify missing outcome variables
### test score is the outcome variable
### 99.98: missing, 99.99: test not completed, -9.00: legitimate skip
### for bygrads
### 9.8: missing, 9.9: legitimate skip
nels_88.score.replace([99.98, 99.99, -9.00], np.nan, inplace=True)
nels_88.bygrads.replace([9.8, 9.9], np.nan, inplace=True)
# %%
### basic regression
### part (1)
### the base group is have no computer, dryer and microwave
grades_reg_1 = smf.ols("bygrads ~ have_computer + have_dryer + have_microwave", data=nels_88).fit()
grades_reg_1.summary()
# %%
### part (2)
### the base group is have no computer, dryer, and microwave and gender is male and race is asian
grades_reg_2 = smf.ols("bygrads ~ have_computer + have_dryer + have_microwave + gender_female + race_hispanic + race_black + race_white + race_american_indian", data=nels_88).fit()
grades_reg_2.summary()
# %%
### the base group is have no computer, dryer, and microwave and income less than 10,000
grades_reg_3 = smf.ols("bygrads ~ have_computer + have_dryer + have_microwave + inc_10_20 + inc_20_35 + inc_35_50 + inc_50_75 + inc_75_100 + inc_100_200 + inc_more_200", data=nels_88).fit()
grades_reg_3.summary()
# %%
### the base group is have no computer, dryer, and microwave
### gender is male and race is asian
### income less than 10,000 and mother's education is less than high school degress
grades_reg_4 = smf.ols("bygrads ~ have_computer + have_dryer + have_microwave + gender_female + race_hispanic + race_black + race_white + race_american_indian + inc_10_20 + inc_20_35 + inc_35_50 + inc_50_75 + inc_75_100 + inc_100_200 + inc_more_200 + motheduc_hs_degree + motheduc_some_college + motheduc_fouryr_degree + motheduc_graduate_degree", data=nels_88).fit()
grades_reg_4.summary()
# %%
reg_obj = Stargazer([grades_reg_1, grades_reg_2, grades_reg_3, grades_reg_4])
reg_obj.covariate_order(["have_computer", "have_dryer", "have_microwave", "gender_female", 
                         "race_hispanic", "race_black", "race_white", "race_american_indian", 
                         "inc_10_20", "inc_20_35", "inc_35_50", "inc_50_75", "inc_75_100", 
                         "inc_100_200", "inc_more_200", "motheduc_hs_degree", "motheduc_some_college", 
                         "motheduc_fouryr_degree", "motheduc_graduate_degree"])
reg_obj.dependent_variable_name("Students' Reported Grades")
print(reg_obj.render_latex())