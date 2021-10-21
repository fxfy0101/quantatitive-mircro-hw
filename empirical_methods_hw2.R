library(tidyverse)
library(plm)
library(AER)
library(stargazer)

### import dataset
df <- read.csv("~/Documents/Data/Nels_88_00_byf4_stu_v1_0.csv")
score <- df %>%
  filter(F4UNI2A == 1) %>%
  select(STU_ID, BY2XMSTD, F12XMSTD, F22XMSTD, F1S99C, F2S96B)

score_1 <- score %>%
  mutate(BY2XMSTD = replace(BY2XMSTD, BY2XMSTD %in% c(99.99, 99.98, -9.00), NA),
         F12XMSTD = replace(F12XMSTD, F12XMSTD %in% c(99.99, 99.98, -9.00), NA),
         F22XMSTD = replace(F22XMSTD, F22XMSTD %in% c(99.99, 99.98, -9.00), NA)) %>%
  mutate(gain1 = F12XMSTD - BY2XMSTD,
         gain2 = F22XMSTD - F12XMSTD,
         divorceyes1 = ifelse(F1S99C == 1, 1, 0),
         divorceno1 = ifelse(F1S99C == 2, 1, 0),
         divorcemissing1 = ifelse(F1S99C == 8, 1, 0),
         divorceunknown1 = ifelse(F1S99C %in% c(7, 9), 1, 0),
         divorceyes2 = ifelse(F2S96B == 1, 1, 0),
         divorceno2 = ifelse(F2S96B == 2, 1, 0),
         divorcemissing2 = ifelse(F2S96B == 8, 1, 0),
         divorceunknown2 = ifelse(F2S96B %in% c(7, 9), 1, 0)) %>%
  select(-c(1:6))

long_panel_score <- pivot_longer(score_1, everything(), names_to = c(".value", "period"), names_pattern = "([a-z]*)(1|2)")
long_panel_score$STU_ID <- rep(score$STU_ID, each = 2)

race_gender <- df %>%
  filter(F4UNI2A == 1) %>%
  select(STU_ID, BYS31A, BYS12) %>%
  mutate(gender_male = ifelse(BYS12 == 1, 1, 0),
         gender_female = ifelse(BYS12 == 2, 1, 0),
         gender_missing = ifelse(BYS12 %in% c(7, 8, 9), 1, 0),
         race_asin = ifelse(BYS31A == 1, 1, 0),
         race_hispanic = ifelse(BYS31A == 2, 1, 0),
         race_black = ifelse(BYS31A == 3, 1, 0),
         race_white = ifelse(BYS31A == 4, 1, 0),
         race_american_indian = ifelse(BYS31A == 5, 1, 0),
         race_unknown = ifelse(BYS31A %in% c(6, 7, 9), 1, 0),
         race_missing = ifelse(BYS31A == 8, 1, 0)) %>%
  select(-c(2:3))

long_panel_student <- left_join(long_panel_score, race_gender, by = "STU_ID")

pols <- lm(gain ~ divorceyes + divorcemissing + divorceunknown + gender_female + gender_missing + race_hispanic + race_black + race_white + race_american_indian + race_unknown + race_missing, data = long_panel_student)
stargazer(pols, type = "latex", se = list(sqrt(diag(vcovCL(pols, cluster = ~ STU_ID)))))

long_panel_student_fe <- pdata.frame(long_panel_student, index = c("STU_ID", "period"))
fe <- plm(gain ~ divorceyes + divorcemissing + divorceunknown, data = long_panel_student_fe, model = "within")
stargazer(fe, type = "latex", se = list(sqrt(diag(plm::vcovHC(fe, cluster = c("group"))))))
