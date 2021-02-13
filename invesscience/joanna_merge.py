'''for mlflow'''
import pandas as pd
import numpy as np
import os
from invesscience.joanna_target import get_company_target
from invesscience.joanna_12 import comps_founded_before
from invesscience.joanna_21 import merge_company_level
from invesscience.felipe_1 import time_serie_investment_new
from invesscience.felipe_10 import n_female_founders
from invesscience.joanna_18 import merge_company_level_uni
from invesscience.joanna_14 import comps_worked_before



def get_training_data(reference="a", cut='2009'):




    path = os.path.dirname(os.path.dirname(__file__))
    acq = pd.read_csv(os.path.join(path,"raw_data","acquisitions.csv"))
    ipos = pd.read_csv(os.path.join(path,"raw_data","ipos.csv"))
    rounds = pd.read_csv(os.path.join(path,"raw_data","funding-rounds.csv"))
    companies = pd.read_csv(os.path.join(path,"raw_data","companies.csv"))
    relationships = pd.read_csv(os.path.join(path,"raw_data","relationships.csv"))
    founders = pd.read_csv(os.path.join(path,"raw_data","founders.csv"))
    people = pd.read_csv(os.path.join(path,"raw_data","people.csv"))
    degrees = pd.read_csv(os.path.join(path,"raw_data","degrees.csv")).drop(columns=["updated_at","created_at"])
    ranking = pd.read_csv(os.path.join(path,"raw_data","support","2018-university-ranking-2.csv"))
    #get company table with target
    companies=get_company_target(ipos, acq, rounds,companies,reference)

    # initial feature selection
    companies = companies[["id",'normalized_name','category_code', "founded_at",\
    'description', "closed_at", "status", 'exit', 'exit_date',\
      'country_code', "state_code",\
      f"date_series_{reference}", f"participants_{reference}", f"raised_amount_usd_{reference}"]]

    #filter series a before certain date
    companies = companies[companies[f"date_series_{reference}"]<cut]
    #print(companies.head())
    #print(companies.shape)

    #get time between founding date and reference round
    companies.founded_at = pd.to_datetime(companies.founded_at)
    companies[f'timediff_founded_series_{reference}'] = (companies[f"date_series_{reference}"] - companies.founded_at)/np.timedelta64(12, 'M')
    #print(companies.head())
    #print(companies.shape)

    # get number of rounds before reference round (if reference is str if not not usefull)
    if type(reference) == str:
        companies = time_serie_investment_new(rounds, companies, reference)

    #print(companies.sort_values(by="rounds_before_a",ascending=False).head())
    #print(companies.shape)

    #get diplomas of founding team
    companies = merge_company_level(people, degrees,companies,relationships)

    #get university ranking of founding team
    companies = merge_company_level_uni(people, degrees, companies, relationships, ranking)

    #print(companies.head())
    #print(companies.shape)

    #get female ratio in founders
    companies = n_female_founders(companies, founders, relationships)

    #print(companies.head())
    #print(companies.shape)

    # get mean companies founded before and total by founders
    companies = comps_founded_before(companies, relationships, founders)#.sort_values(by="female_ratio",ascending=False)

    # get mean companies worked at before by founders
    companies = comps_worked_before(companies, relationships, founders)

    # target encoding
    exits ={"acquisition":1, "ipo":1, "no exit":0}
    companies["target"] = companies.exit.map(exits)

    return companies



if __name__ == "__main__":
    companies = get_training_data(reference="a")
    print(companies.head(10))
    print(companies.shape)
    print(companies.columns)
    print(companies.info())


