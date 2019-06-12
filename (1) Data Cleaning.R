# STAT 359: Final project
# (1) Data cleaning R script 
# John Lee


# load packages
library(tidyverse)
library(foreign)
library(forcats) # for factor vars
library(modelr)
library(haven)
library(skimr)



# Part 1: clean the data -------------------------------

# Steps
# (1) choose the variables i want and just load those + save in a RDS
# (2) recode the vars I want (e.g., as numeric or factor)
  # smart coding strategy: all of the values I want to omit, just code as NA (then i can use na.omit later)

set.seed(3)


# load data (just load the RDS file now)
# anes2016_full_dat <- read.dta("ANES data/anes_timeseries_2016_Stata12.dta")
# write_rds(anes2016_full_dat, "anes2016_full_dat.rds")

anes2016_full_dat <- read_rds("anes2016_full_dat.rds")

# Check values - for a given var
anes2016_full_dat %>% count(V161361x) %>% View

anes2016_full_dat %>% count(V162034a)

# just select the vars of interest 
anes2016_reduced_dat <- anes2016_full_dat %>%
  select(
    V162034a, #DV: whom people voted for president in 2016
    V161342, # gender
    V161310x, # race
    V161267, # age
    V161270, # education
    V161361x, # income band
    V161268, # marital status
    V161266c, # born again christian
    V161266d, # evangelical christian 
    V161241, # religiosity
    V161266m, # athiest 
    V161095, # Dem party FT
    V161096, # GOP FT
    V161126, # ideol self-place
    V161127, # ideol, follow-up
    V161003, # pays attention to politics
    V161006, # whom you voted for in the 2012 pres election
    V161113, # support obamacare
    V161187, # support gun control; 
    V161196x, # support building a wall 
    V161232 # abortion
  )


# Create the custom functions to recode the var values 

# Resource if there are problems with case_when - https://github.com/tidyverse/dplyr/issues/3202

rc_votedfortrump <- function(x){
  case_when(
    x == "1. Hillary Clinton" ~ 0,
    x == "2. Donald Trump" ~ 1,
    x == "3. Gary Johnson" ~ 0,
    x == "4. Jill Steiin" ~ 0,
    x == "5. Other candidate SPECIFY" ~ 0,
    TRUE ~ NA_real_
  )
}

rc_male <- function(x){
  case_when(
    x == "1. Male" ~ 1,
    x == "2. Female" ~ 0,
    x == "3. Other" ~ 0,
    TRUE ~ NA_real_
  )
}

rc_race <- function(x){
  case_when(
    x == "1. White, non-Hispanic" ~ "White",
    x == "2. Black, non-Hispanic" ~ "Black",
    x == "3. Asian, native Hawaiian or other Pacif Islr,non-Hispanic" ~ "Asian",
    x == "4. Native American or Alaska Native, non-Hispanic" ~ "Native American",
    x == "5. Hispanic" ~ "Hispanic",
    x == "6. Other non-Hispanic incl multiple races [WEB: blank 'Other' counted as a race]" ~ "Other",
    TRUE ~ NA_character_
  )
}

rc_educ  <- function(x){
  case_when(
    x == "-9. Refused" ~ NA_character_,
    x == "13. Bachelor's degree (for example: BA, AB, BS)" ~ "College Grad",
    x == "14. Master's degree (for example: MA, MS, MENG, MED, MSW, MBA)" ~ "Grad degree",
    x == "15. Professional school degree (for example: MD, DDS, DVM, LLB, JD)" ~ "Grad degree",
    x == "16. Doctorate degree (for example: PHD, EDD)" ~ "Grad degree",
    TRUE ~ "Non-BA"
  )
}

rc_incomeb  <- function(x){
  ifelse(x == "-9. Refused" | x == "-5. Interview breakoff (sufficient partial IW)",
         NA, x)
}

rc_divorced <- function(x){
  case_when(
    x == "4. Divorced" ~ 1,
    x == "-9. Refused" ~ NA_real_,
    TRUE ~ 0
  )
}

rc_fundamentalist <- function(x){
  case_when(
    x == "0. Not selected" ~ 0,
    x == "1. Selected" ~ 1,
    TRUE ~ NA_real_
  )
}

rc_religious <- function(x){
  case_when(
    x == "2. Not important" ~ 0,
    x == "1. Important" ~ 1,
    TRUE ~ NA_real_
  )
}

rc_athiest <- function(x){
  case_when(
    x == "0. Not selected" ~ 0,
    x == "1. Selected" ~ 1,
    TRUE ~ NA_real_
  )
}

rc_ideology  <- function(x){
  ifelse(x == "-9. Refused" | x == "-8. Don't know (FTF only)" |
           x == "99. Haven't thought much about this (FTF ONLY: DO NOT PROBE)",
         NA, as.numeric(x)-2)
}

rc_ideol2 <- function(x){
  case_when(
    x == "1. Liberal" ~ 2,
    x == "2. Conservative" ~ 6,
    x == "3. Moderate" ~ 4,
    TRUE ~ NA_real_
  )
}

rc_pol_interest <- function(x){
  case_when(
    x == "1. Always" ~ 1,
    TRUE ~ 0
  )
}

rc_2012romney <- function(x){
  case_when(
    x == "2. Mitt Romney" ~ 1,
    x == "1. Barack Obama" ~ 0,
    x == "5. Other SPECIFY" ~ 0,
    TRUE ~ NA_real_
  )
}

rc_support_policy <- function(x){
  case_when(
    x == "1. Favor" ~ 1,
    x == "2. Oppose" ~ 0,
    x == "3. Neither favor nor oppose" ~ 0,
    TRUE ~ NA_real_
  )
}

rc_support_gcontrol <- function(x){
  case_when(
    x == "1. More difficult" ~ 1,
    x == "2. Easier" ~ 0,
    x == "3. Keep these rules about the same" ~ 0,
    TRUE ~ NA_real_
  )
}

rc_oppose_wall <- function(x){
  ifelse(x == "-9. RF (-9) in V161196 or V161196a" | x == "-8. DK (-8) in V161196 or V161196a (FTF only)",
         NA, as.numeric(x)-2)
}

rc_pro_choice <- function(x){
  case_when(
    x == "1. By law, abortion should never be permitted." ~ 0,
    x == "2. By law, only in case of rape, incest, or woman's life in danger." ~ 0,
    x == "3. By law, for reasons other than rape, incest, or woman's life in danger if need established" ~ 0,
    x == "4. By law, abortion as a matter of personal choice." ~ 1,
    x == "5. Other SPECIFY" ~ 0,
    TRUE ~ NA_real_
  )
}


# Rename + recode the vars 
anes2016_rev_dat <- anes2016_reduced_dat %>%
  mutate(votedfortrump = rc_votedfortrump(V162034a) %>% as.factor,
         male = rc_male(V161342) %>% as.numeric,
         race = rc_race(V161310x) %>% as.factor,
         age = ifelse(V161267 < 0, NA, V161267),
         educ = rc_educ(V161270) %>% as.factor,
         income_band = rc_incomeb(V161361x) %>% as.numeric,
         divorced = rc_divorced(V161268) %>% as.numeric,
         born_again = rc_fundamentalist(V161266c) %>% as.numeric,
         evangelical = rc_fundamentalist(V161266d) %>% as.numeric,
         religious = rc_religious(V161241) %>% as.numeric,
         athiest = rc_athiest(V161266m) %>% as.numeric,
         dem_FT = ifelse(V161095 < 0, NA, V161095),
         repub_FT = ifelse(V161096 < 0, NA, V161096),
         ideology = rc_ideology(V161126),
         ideol_followup = rc_ideol2(V161127),
         conserv_scale = ifelse(is.na(ideology), ideol_followup, ideology),
         pol_interest = rc_pol_interest(V161003)  %>% as.numeric,
         romney2012 = rc_2012romney(V161006) %>% as.numeric,
         support_obamacare = rc_support_policy(V161113) %>% as.numeric,
         support_gcontrol = rc_support_gcontrol(V161187) %>% as.numeric,
         oppose_wall = rc_oppose_wall(V161196x) %>% as.numeric,
         pro_choice = rc_pro_choice(V161232) %>% as.numeric
         )


# Check the recoded values
anes2016_rev_dat %>% count(pro_choice) 

# Use listwise deletion to get to the final full dataset 
anes2016_rev_dat <- na.omit(anes2016_rev_dat %>% select(-ideology, -ideol_followup, -romney2012)) %>%
  as_tibble %>%
  # Only filter in the recoded vars of interest 
  select(
    votedfortrump, # DV
    male, # demographic
    race, # demographic
    age, # demographic
    educ, # SES
    income_band, # SES
    divorced, # social, cultural
    born_again, # social, cultural
    evangelical, # social, cultural
    religious, # social, cultural
    athiest, # social, cultural
    dem_FT, # general partisanship
    repub_FT, # general partisanship
    conserv_scale, # ideology
    pol_interest,
    support_obamacare, # policy pref
    support_gcontrol, # policy pref
    oppose_wall, # policy pref
    pro_choice # policy pref
  )
  

# Inspect the final dataset 
anes2016_rev_dat %>% skim

anes2016_rev_dat %>% count(age)


# Now we need to transform the data ----------------------------------------------
# Notes: 
# (1) for now, just standardize the continuous predictors + drop unused levels for factors; 
# (2) I'll do OHE whenever necessary (e.g., boosted trees, NN)

# Drop unused levels for factor vars, explicitly set ref levels
anes2016_rev_dat <- anes2016_rev_dat %>% 
  mutate(
    educ = droplevels(educ),
    race = droplevels(race),
    votedfortrump = droplevels(votedfortrump),
    educ = relevel(educ, ref = "Non-BA"), 
    race = relevel(race, ref = "White"), 
    votedfortrump = relevel(votedfortrump, ref = "0") 
    )

# Standardize the continuous predictors
std_anes2016_rev <- anes2016_rev_dat %>% 
  select(age,
         conserv_scale,
         dem_FT,
         income_band,
         oppose_wall,
         repub_FT
         ) %>%
  scale() %>% # standardize the continuous predictors (mean = 0, sd = 1)
  as_tibble()

# Make sure the scaling worked (it did)
std_anes2016_rev %>% skim

# Create a subset with just the cat vars 
cat_vars <- anes2016_rev_dat %>% 
  select(-age,
         -conserv_scale,
         -dem_FT,
         -income_band,
         -oppose_wall,
         -repub_FT
  )

# Create a final combined df
final_anes2016_dat <- base::cbind(cat_vars, std_anes2016_rev) %>% as_tibble()

# Inpsect the final combined df
final_anes2016_dat %>% skim

final_anes2016_dat 

# final step for the R script: Create the training and test sets (let's do a 70-30 split)

train_set <- final_anes2016_dat %>% sample_frac(0.7) # 1,676 obs
test_set <- final_anes2016_dat %>% setdiff(train_set) # 718 obs

write_rds(train_set, "train_set.rds")
write_rds(test_set, "test_set.rds")



