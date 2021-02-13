
import pandas as pd
import numpy as np
import os


def n_female_founders(companies, founders, relationships):
    '''get your numbers of females for each company ;) '''

    merge_1 = relationships.merge(founders[['person_object_id', 'gender']], how = 'left', on = 'person_object_id')
    merge_1 = merge_1[merge_1.founder ==True]
    merge_1 ['n_female_founders'] = merge_1.gender.map(lambda x: 1 if x=='Female' else 0)
    merge_1 = merge_1.groupby('relationship_object_id',as_index=False).sum()[['relationship_object_id','n_female_founders']]
    merge_1 = merge_1.rename(columns = {'relationship_object_id':'id'})
    companies = companies.merge(merge_1, how = 'left', on = 'id')
    companies["female_ratio"] = companies.n_female_founders/companies.founder_count

    return companies

if __name__ == "__main__":
    people = pd.read_csv(os.path.join('..',"raw_data","people.csv"))
    relationships = pd.read_csv(os.path.join('..',"raw_data","relationships.csv"))
    companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))
    companies = n_female_founders(companies, founders, relationships)
    print(companies.head())
