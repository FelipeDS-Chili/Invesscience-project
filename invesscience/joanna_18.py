import pandas as pd
import numpy as np
import os

def degree_standardisation(x):
    if "harvard" in x:
        return "harvard university"
    if "harverd" in x:
        return "harvard university"
    if "stanford" in x:
        return "stanford university"
    if "massachusetts institute of technology (mit)" in x:
        return "massachusetts institute of technology (mit)"
    if "massachusetts institute of technology" in x:
        return "massachusetts institute of technology (mit)"
    if "mit " in x:
        return "massachusetts institute of technology (mit)"
    if " mit" in x:
        return "massachusetts institute of technology (mit)"
    if "(mit)" in x:
        return "massachusetts institute of technology (mit)"
    if "california institue of technology" in x:
        return "california institue of technology (caltech)"
    if "caltech" in x:
        return "california institue of technology (caltech)"
    if "cambridge" in x:
        return "university of cambridge"
    if "oxford" in x:
        return "university of oxford"
    if "university college london" in x:
        return "(ucl) university college london"
    if "imperial college" in x:
        return "imperial college london"
    if "university of chicago" in x:
        return "university of chicago"
    if "chicago university" in x:
        return "university of chicago"
    if "eth zurich" in x:
        return "eth zurich (swiss federal institute of technology)"
    if "swiss federal institute of technology" in x:
        return "eth zurich (swiss federal institute of technology)"
    if "nanyang" in x:
        return "nanyang technological university (NTU)"
    if "epfl" in x:
        return "ecole polytechnique fédérale de lausanne (epfl)"
    if all(c in x for c in ["ecole polytechnique","lausanne"]):
        return "ecole polytechnique fédérale de lausanne (epfl)"
    if 'princeton' in x:
        return "princeton university"
    if "cornell" in x:
        return "cornell university"
    if all(c in x for c in ["national university","singapore"]):
        return 'national university of singapore (nus)'
    if "yale" in x:
        return "yale university"
    if "john hopkins" in x:
        return "joh hopkins university"
    if "columbia" in x:
        return "columbia university"
    if "pennsylvania" in x:
        return "university of pennsylvania"
    if "upenn" in x:
        return "university of pennsylvania"
    if "wharton" in x:
        return "university of pennsylvania"
    if "duke university" in x:
        return "duke university"
    if "michigan" in x:
        return "university of michigan"
    if "australian national university" in x:
        return "australian national unversity (anu)"
    if all (c in x for c in ["king","college"]):
        return "king's college london (kcl)"
    if all (c in x for c in ["university","edinburgh"]):
        return "university of edinburgh"
    if "tsinghua" in x:
        return "tsinghua university"
    if "cuhk" in x:
        return 'the chinese university of hong kong (cuhk)'
    if "hkust" in x:
        return "the hong kong university of science and technology (hkust)"
    if "hku" in x:
        return "university of hong kong (hku)"
    if all (c in x for c in ["hong kong", "technology", "science"]):
        return "the hong kong university of science and technology (hkust)"
    if all (c in x for c in ["hong kong", "chinese"]):
        return "the chinese university of hong kong (cuhk)"
    if all (c in x for c in ["hong kong", "city"]):
        return 'city university of hong kong'
    if "hong kong" in x:
        return "university of hong kong (hku)"
    if "berkeley" in x:
        return 'university of california, berkeley (ucb)'
    if "northwestern" in x:
        return "northwestern university"
    #if all(c in x for c in ["tokyo","technology"]):
     #   return "tokyo institute of technology"
    if "tokyo" in x:
        return 'the university of tokyo'
    if "toronto" in x:
        return "university of toronto"
    if "mcgill" in x:
        return "mcgill university"
    if "ucla" in x:
        return "university of california, los angeles (ucla)"
    if all (c in x for c in ["california", "los angeles", "university"]):
        return "university of california, los angeles (ucla)"
    if "manchester" in x:
        return "the university of manchester"
    if "london school of economics" in x:
        return "london school of economics and political science (lse)"
    if "kyoto" in x:
        return "kyoto university"
    if "seoul" in x:
        return "seoul national university (snu)"
    if "peking university" in x:
        return "peking university"
    if "ucsd" in x:
        return "university of california, san diego (ucsd)"
    if all (c in x for c in ["california", "san diego", "university"]):
        return "university of california, san diego (ucsd)"
    if "fudan" in x:
        return "fudan university"
    if "kaist" in x:
        return "kaist - korea advanced institute of science and technology"
    if "korea advanced institute of science and technology" in x:
        return "kaist - korea advanced institute of science and technology"
    if all (c in x for c in ["university","melbourne"]):
        return "the university of melbourne"
    if "ecole normale sup" in x:
        return "ecole normale supérieure, paris, (ens paris)"
    if "école normale sup" in x:
        return "ecole normale supérieure, paris, (ens paris)"
    if "ens paris" in x:
        return "ecole normale supérieure, paris, (ens paris)"
    if all (c in x for c in ["university","bristol"]):
        return "university of bristol"
    if "new south wales" in x:
        return 'the university of new south wales (unsw)'
    if "unsw" in x:
        return 'the university of new south wales (unsw)'
    if "carnegie mellon" in x:
        return "carnegie mellon university"
    if 'queensland' in x:
        return 'the university of queensland (uq)'
    if all(c in x for c in ["university", "sydney"]):
        return 'the university of sydney'
    #if "british columbia" in x:
        #return "university of british columbia"
    #if "new york university" in x:
      #  return "new york university (nyu)"
   # if "nyu" in x:
        #return "new york university (nyu)"
    #if "brown" in x:
        #return "brown university"
    #if "delft" in x:
        #return 'delft university of technology'
   # if all(c in x for c in ["wisconsin",'masion']):
        #return "university of wisconsin-madison"
    else:
        return x.lower()

def add_ranking(degrees,ranking):
    ranking["Institution Name"] = ranking["Institution Name"].map(lambda x: x.lower().strip())
    ranking = ranking.rename(columns={"Institution Name":"institution"})
    degrees.institution = degrees.institution.astype(str).map(lambda x: degree_standardisation(x.lower()))
    degrees = degrees.merge(ranking[["institution","2018"]], how="left", on="institution").rename(columns={"2018":"ranking"})
    degrees.ranking = degrees.ranking.astype(str).map(lambda x: int(x.strip("=").split("-")[0]) if x!= "nan" else np.nan)
    degrees["top_5"] = degrees.ranking.map(lambda x: 1 if x <= 5 else 0)
    degrees["top_20"] = degrees.ranking.map(lambda x: 1 if x <= 20 else 0)
    degrees["top_50"] = degrees.ranking.map(lambda x: 1 if x <= 50 else 0)
    return degrees


def merge_people_level_uni(people, degrees,ranking):
    #print(people.shape)
    degrees = add_ranking(degrees,ranking)
    degrees_per_person = degrees.groupby("object_id").sum().drop(["id","ranking"], axis=1)
    degrees_per_person = degrees_per_person.astype(int).applymap(lambda x: 1 if x>0 else 0)
    degrees_per_person["top_50"] = (degrees_per_person["top_50"] - degrees_per_person["top_20"] - degrees_per_person["top_5"])\
    .astype(int).map(lambda x: max(x,0))
    degrees_per_person["top_20"] = (degrees_per_person["top_20"] - degrees_per_person["top_5"])\
    .astype(int).map(lambda x: max(x,0))
    degrees_per_person = degrees_per_person.reset_index().rename(columns={"object_id":"person_object_id"})
    people = people.merge(degrees_per_person,on="person_object_id", how="left")
    people.loc[:,degrees_per_person.columns] = people.loc[:,degrees_per_person.columns].fillna(0)
    #print(people.shape)
    #print(people.sample(20))
    return people


def merge_company_level_uni(people, degrees,companies, relationships, ranking):
    people = merge_people_level_uni(people, degrees,ranking)
    relationships = relationships[relationships.founder][["person_object_id","relationship_object_id"]]\
    .merge(people, on="person_object_id", how="left")

    tmp = relationships[["relationship_object_id","top_5",\
    "top_20","top_50"]]\
    .groupby("relationship_object_id",as_index=False).sum()
    tmp["top_5_bool"] = tmp.top_5.map(lambda x: 1 if x > 0 else 0)
    tmp["top_20_bool"] = tmp.top_20.map(lambda x: 1 if x > 0 else 0)
    tmp["top_50_bool"] = tmp.top_50.map(lambda x: 1 if x > 0 else 0)
    tmp2 = relationships.groupby("relationship_object_id",as_index=False)\
    .count()[["relationship_object_id","person_object_id"]]\
    .rename(columns={"person_object_id":"founder_count"})

    tmp3 = tmp.merge(tmp2, on="relationship_object_id")
    for i in ["top_5","top_20","top_50"]:
        tmp3[i] = tmp3[i]/ tmp3.founder_count
    tmp3 = tmp3.rename(columns={"relationship_object_id":"id"}).drop("founder_count", axis=1)
    companies = companies.merge(tmp3, how="left", on="id")
    #print(companies.shape)
    #print(companies.sample(20))
    return companies

if __name__ == "__main__":
    degrees = pd.read_csv(os.path.join('..',"raw_data","degrees.csv")).drop(columns=["updated_at","created_at"])
    people = pd.read_csv(os.path.join('..',"raw_data","people.csv"))
    relationships = pd.read_csv(os.path.join('..',"raw_data","relationships.csv"))
    companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))
    ranking = pd.read_csv(os.path.join("..","raw_data","support","2018-university-ranking-2.csv"))

    companies = merge_company_level_uni(people, degrees, companies, relationships, ranking)
    print(companies.sample(20))
