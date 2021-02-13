import pandas as pd
import numpy as np
import os

def comps_founded_before(companies, relationships, founders):
    '''Modifying companies and founder table'''
    founding = relationships[relationships.founder]
    founding = founding.rename(columns={"relationship_object_id":"id"})
    merged = companies.merge(founding, how="left", on="id")
    merged.founded_at = pd.to_datetime(merged.founded_at)

    # Number of companies founded in total
    companies_founded = merged[["person_object_id","founder"]].groupby("person_object_id"\
                    ,as_index=False).sum().rename(columns={"founder":"founded_companies"})
    companies_founded.founded_companies = companies_founded.founded_companies.astype("int")

    ##new table founders to return with new column founded_companies
    founders = founders.merge(companies_founded, how="left", on="person_object_id")

    ##new column to companies with mean_comp_founded_ever
    tmp_2 = merged.merge(companies_founded, how="left", on="person_object_id")[["id","founded_companies"]]\
    .groupby("id", as_index=False).mean()
    tmp_2 = tmp_2.rename(columns={"founded_companies":"mean_comp_founded_ever"})
    companies = companies.merge(tmp_2)

    # Number of companies founded before specific one
    tmp = merged.sort_values(by=["person_object_id","founded_at"]).groupby("person_object_id").cumcount()
    tmp = pd.concat([merged, tmp], axis=1).sort_values(by=["person_object_id","founded_at"])
    tmp = tmp.rename(columns={0:"founded_count"})
    tmp.loc[tmp.person_object_id.isnull(),'founded_count']=np.nan
    tmp = tmp[['id',"founded_count"]].groupby("id",as_index=False).mean()\
    .rename(columns={"founded_count":"mean_comp_founded_before"})
    ##new column to companies with mean_comp_founded_before
    companies = companies.merge(tmp, how="left", on="id")

    #print(companies.shape)
    #print(founders.shape)
    #print(companies.head())
    #print(founders.head())

    return companies

if __name__ == "__main__":
    companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))
    relationships = pd.read_csv(os.path.join('..',"raw_data","relationships.csv"))
    founders = pd.read_csv(os.path.join('..',"raw_data","founders.csv"))

    comps_founded_before(companies, relationships, founders)
