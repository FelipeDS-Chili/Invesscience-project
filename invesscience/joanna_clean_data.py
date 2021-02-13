
import pandas as pd
import numpy as np
import os
from invesscience.joanna_merge_2 import get_training_data

## Categories
def clean_cat(x):
    categories = ["software","web", "mobile", "biotech", "advertising", "enterprise", "cleantech", "hardware", "network_hosting", "semiconductor", "search",\
        "security", "public_relations", "finance", "consulting", "analytics", "social", "music", "medical", "ecommerce"]
    if pd.isnull(x):
        return np.nan
    elif x in categories:
        return x
    else:
        return "other_category"

## Location

def clean_country(x):
    countries = ["FRA","USA", "GBR", "CAN", "ISR", "DEU", "CHN", "SWE", "IND"]
    if pd.isnull(x):
        return np.nan
    elif x in countries:
        return x
    else:
        return "other_country"

def clean_state(x):
    states = ['CA', 'NY', "MA", "WA", "TX", "CO", "VA", "IL", "PA", "MD", "NC", "NJ", "FL"]
    if pd.isnull(x):
        return np.nan
    elif x=="not_US":
        return "not_US"
    elif x in states:
        return x
    else:
        return "other_state"


## Dates
def clean_date(x):
    if pd.isnull(x):
        return np.nan
    elif str(x) > "1995":
        return str(x)
    else:
        return "pre_1996"

def clean_series(x):
    if pd.isnull(x):
        return np.nan
    elif x >= 10:
        return 10
    else:
        return x

def clean_date_series(x):
    if pd.isnull(x):
        return np.nan
    elif str(x) > "1998":
        return str(x)
    else:
        return "pre_1999"

##Series reference
def clean_participants(x):
    if pd.isnull(x):
        return np.nan
    elif x==0:
        return np.nan
    elif x <= 4:
        return int(x)
    else:
        return 5

def clean_amount(x):
    if pd.isnull(x):
        return np.nan
    elif x == 0:
        return np.nan
    elif x< 1000000:
        return 1000000
    elif x< 3000000:
        return 3000000
    elif x< 5000000:
        return 5000000
    elif x< 10000000:
        return 10000000
    elif x< 20000000:
        return 20000000
    elif x< 50000000:
        return 50000000
    else:
        return 100000000

def clean_time(x):
    if pd.isnull(x):
        return np.nan
    elif x <=1:
        return "1_less"
    elif x>=10:
        return "10_more"
    else:
        return "2_9"

def clean_rounds(x):
    if pd.isnull(x):
        return np.nan
    elif x==0:
        return 0
    elif x == 1:
        return 1
    else:
        return 2


## Founders
def clean_founder(x):
    if pd.isnull(x):
        return np.nan
    elif x <= 3:
        return int(x)
    else:
        return 4

def clean_female(x):
    if pd.isnull(x):
        return np.nan
    elif x == 0:
        return 0
    else:
        return 1

def clean_school(x):
    if pd.isnull(x):
        return np.nan
    elif x >0:
        return 1
    else:
        return 0

def clean_study(x):
    if pd.isnull(x):
        return np.nan
    elif x >0:
        return 1
    else:
        return 0

def clean_degree_count(x):
    if pd.isnull(x):
        return np.nan
    if x>=3:
        return 3
    else:
        return round(x)

def clean_founded_before(x):
    if pd.isnull(x):
        return np.nan
    elif x == 0:
        return 0
    else:
        return 1

def clean_worked_before(x):
    if pd.isnull(x):
        return np.nan
    elif x >=2:
        return 2
    else:
        return round(x)



def clean_training_data(companies, reference="a"):
    '''need to add the missing data Felipe completed'''
    filled_1 = pd.read_csv("../raw_data/datanamed_completed.csv", sep=";", header=1)[["id","state_code"]].set_index("id")
    filled_2 = pd.read_csv("../raw_data/countries_filled.csv", sep=";")[["id","country_code"]].set_index("id")
    filled_3 = pd.read_csv("../raw_data/categories_filled.csv", sep=";")[["id","category_code"]].set_index("id")
    filled_4 = pd.read_csv(os.path.join("..", 'raw_data' , 'last_complete_a.csv'), sep=';').set_index("id")



    companies.country_code = companies.set_index("id").country_code.fillna(filled_2.country_code).reset_index().country_code
    companies.country_code = companies.set_index("id").country_code.fillna(filled_4.country_code).reset_index().country_code

    companies.loc[companies.country_code!="USA","state_code"] = "not_US"
    companies.loc[companies.country_code.isnull(),"state_code"] = np.nan

    companies.state_code = companies.set_index("id").state_code.fillna(filled_1.state_code).reset_index().state_code
    companies.state_code = companies.set_index("id").state_code.fillna(filled_4[filled_4.country_code=="USA"].state_code).reset_index().state_code


    companies.category_code = companies.set_index("id").category_code.fillna(filled_3.category_code).reset_index().category_code


    '''neeed to check if same trends for series 0'''
    #cleaning categories - remove some to put in "other"
    companies.category_code = companies.category_code.map(clean_cat).replace('NaT', np.nan)

    #cleaning country to most frequent and put rest in other_country
    companies.country_code = companies.country_code.map(clean_country).replace('NaT', np.nan)

    #cleaning state to most frequent and put rest in other_state
    companies.state_code = companies.state_code.map(clean_state).replace('NaT', np.nan)

    #cleaning founded_at to year and becomes cat variable, pre_1996 and after
    companies["founded_at"] = companies.founded_at.dt.strftime('%Y')
    companies["founded_at"] = companies.founded_at.map(clean_date).replace('NaT', np.nan)

    #new variable time since series a (when somoene enters a company in the product page, can specify within 5 years or 10 years or other >> will be the time_diff_series_a_now)
    companies[f"time_diff_series_{reference}_now"] = round(-(companies[f"date_series_{reference}"] - pd.to_datetime("31-12-2013"))/np.timedelta64(12, 'M')).map(clean_series).replace('NaT', np.nan)

    #cleaning date series reference to year and becomes cat variable pre_1999 and after
    companies[f"date_series_{reference}"] = companies[f"date_series_{reference}"].map(lambda x: np.nan if pd.isnull(x) else x).dt.strftime('%Y').map(clean_date_series).replace('NaT', np.nan)

    # cleaning participants series reference, 5+ =5
    companies[f"participants_{reference}"] = companies[f"participants_{reference}"].map(clean_participants).replace('NaT', np.nan)

    # cleaning amount series reference in 7 buckets and removing 0s
    companies[f"raised_amount_usd_{reference}"] = companies[f"raised_amount_usd_{reference}"].map(clean_amount).replace('NaT', np.nan)

    # cleaning time to series reference in 3 buckets, becomes categorical
    companies[f"timediff_founded_series_{reference}"]= companies[f"timediff_founded_series_{reference}"].map(clean_time).replace('NaT', np.nan)

    #cleaning founder to new cat 4+ (=4)
    companies.founder_count = companies.founder_count.map(clean_founder).replace('NaT', np.nan)

    #cleaning female_ratio to 0 or 1
    companies["female_ratio"] = companies["female_ratio"].map(clean_female).replace('NaT', np.nan)

    #putting top 5 and 20 together
    companies["top_20_bool"] = (companies["top_5_bool"] + companies["top_20_bool"]).map(clean_school).replace('NaT', np.nan)

    #graduate and undergrad as a bool
    companies["graduate"] = companies.graduate.map(clean_study).replace('NaT', np.nan)
    companies["undergrad"] = companies.undergrad.map(clean_study).replace('NaT', np.nan)
    companies["professional"] = companies.undergrad.map(clean_study).replace('NaT', np.nan)

    # degree count 4 categories 0 - 3
    companies["degree_count"] = companies.degree_count.map(clean_degree_count).replace('NaT', np.nan)

    #companies founded before to bool
    companies["mean_comp_founded_before"] = companies["mean_comp_founded_before"].map(clean_founded_before).replace('NaT', np.nan)

    #companies worked before to 0 - 1 - 2
    companies["mean_comp_worked_before"] = companies["mean_comp_worked_before"].map(clean_worked_before).replace('NaT', np.nan)

    #drop useless features
    companies = companies.drop(["n_female_founders","top_5_bool","top_5","top_20", "top_50_bool","top_50",f"date_series_{reference}"
        ,"mean_comp_founded_ever", "phd", "MBA", "cs"], axis=1)

    if type(reference) == str:
        # clean rounds before a to 0, 1, 2+
        companies["rounds_before_a"] = companies.rounds_before_a.map(clean_rounds).replace('NaT', np.nan)
        companies = companies.drop(["participants_before_a", "raised_before_a"], axis=1)

    return companies



if __name__ == "__main__":
    companies = get_training_data(reference="a",cut="2014")
    print(companies.head(10))
    print(companies.shape)
    print(companies.columns)
    print(companies.info())
    print(companies.state_code.value_counts())
    print(companies.country_code.value_counts())
    companies = clean_training_data(companies,reference="a")
    print(companies.info())
    print(companies.state_code.value_counts())
    print(companies.country_code.value_counts())





