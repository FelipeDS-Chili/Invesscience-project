
import pandas as pd
import numpy as np
import os

def to_drop_1(x):
    titles= ["board member", "advisor", "board", "investor", "chairman"\
             , "board of directors", "executive chairman", "investor", "angel", "angel investor",
            'board of advisors', "board of advisory", "member board of directors", "board observer",
            "board director", "advisory board", "member", "board director", "investor and advisor"]
    if 'advistor' in x:
        return True
    if "advisory" in x:
        return True
    if "investor" in x:
        return True
    if "observer" in x:
        return True
    for c in titles:
        if x==c:
            return True
    else:
        return False

def comps_worked_before(companies, relationships, founders):
    relationships = relationships.rename(columns={"relationship_object_id":"id"})
    relationships = relationships
    merged = companies.merge(relationships, how="left", on="id")
    merged.founded_at = pd.to_datetime(merged.founded_at)
    merged["to_drop"] = merged.title.astype(str).map(lambda x: to_drop_1(x.lower()))
    merged = merged[merged.to_drop==False]
    merged = merged.sort_values(by="founder", ascending=False).drop_duplicates(["id","person_object_id"])

    # Number of companies worked before specific one
    tmp = merged.sort_values(by=["person_object_id","founded_at"]).groupby("person_object_id").cumcount()
    tmp = pd.concat([merged, tmp], axis=1).sort_values(by=["person_object_id","founded_at"])
    tmp = tmp.rename(columns={0:"worked_count"})
    tmp.loc[tmp.person_object_id.isnull(),'worked_count']=np.nan
    tmp = tmp[tmp.founder==True]

    tmp = tmp[['id',"worked_count"]].groupby("id",as_index=False).mean()\
    .rename(columns={"worked_count":"mean_comp_worked_before"})
    print(tmp.head())
    companies = companies.merge(tmp, how="left", on="id")
    print(companies.head())

    return companies


if __name__ == "__main__":
    companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))
    relationships = pd.read_csv(os.path.join('..',"raw_data","relationships.csv"))
    founders = pd.read_csv(os.path.join('..',"raw_data","founders.csv"))

    comps_worked_before(companies, relationships, founders)

