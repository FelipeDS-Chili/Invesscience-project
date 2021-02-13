import pandas as pd
import numpy as np
import os

#Degree level
def degree_clean(x):
    x_lower = str(x).lower()
    if x_lower =="degree":
        return "Undergraduate"
    if x_lower=="degrees":
        return "Undergraduate"
    if x_lower=="undergraduate degree":
        return "Undergraduate"
    if x_lower=="graduate degree":
        return "Graduate"
    if x_lower=="associate degree":
        return "Associate"
    if x_lower=="btech":
        return "BS"
    if x_lower=="mis":
        return "Undergraduate"
    if "phd" in x_lower:
        return "PhD"
    if "ph.d" in x_lower:
        return "PhD"


    if x == "MENG" or x=="MSE" or x=="ME" or x_lower=='master of engineering' or x_lower=="m.s."\
    or x_lower=="m.s" or x_lower=="m.e." or x_lower=="mcs" or x_lower=="mca" or x_lower=="msme"\
    or x_lower=="masc" or x_lower=="m.s.e" or x_lower=="master of science":
        return "MS"
    if x=="M.S.Eng":
        return 'MS'

    if x_lower=="master degree" or x_lower=="masters" or x_lower=="masters degree" or x_lower=="maitrise"\
    or x_lower=="magister" or x_lower=="deug":
        return "Master"

    if x_lower=="diploma" or x_lower=="diplom" or x_lower=="certificate" or x_lower=="certification"\
    or x_lower=='honours' or x_lower=="minor" or x_lower=='fellowship' or x_lower=="incomplete"\
    or x_lower=="honors" or x_lower=="laurea" or x_lower=="cfa" or x_lower=="unknown"\
    or x_lower=="major" or x_lower=="specialist" or x_lower=='advanced management program'\
    or x_lower=="exchange" or x_lower=="hsc" or x_lower=="entrepreneurship" \
    or x_lower=="international baccalaureate" or x_lower=="amp" or x_lower=="diplom-kaufmann" or \
    x_lower=="certified" or x_lower=="diplom kaufmann":
        return np.nan

    if x=="Bachelor of Arts (B.A.)" or x=="BA / BA (Hons)" or x=="AB" or x=="A.B.":
        return "BA"

    if x=="Bachelor of Science (B.S.)" or x=="BE" or x_lower=="bsme" or x=="BENG" or x_lower=="b.s"\
    or x_lower=="bachelors in engineering" or x_lower=="bachelors of engineering" or x_lower=="bca" or \
    x_lower=="bachelor of information technolo" or x_lower=="bmath" or x_lower=="bee":
        return "BS"
    if x=="Master of Business Administratio":
        return "MBA"
    if x == "B of Eng" or x=="Bachelor of Civil Engineering" or x=="B.S. in Electrical Engineering" \
    or x=="BS, Civil Engineering" or x_lower=="bsc" or x_lower=='bcs':
        return "BS"

    if x_lower=="bachelors" or x_lower=="bachelor of economics":
        return "Bachelor"
    if x_lower=='s.b.':
        return "BS"
    if "bachelor of business administrat" in x_lower:
        return 'BBA'
    if x_lower=='bachelor of technology (btech':
        return "BS"
    if x_lower=="chartered accountant" or x_lower=="cpa" or x_lower=="ca" or x_lower=="hnd"\
    or x_lower=="aca" or x_lower=="professional certificate" or x_lower=="bts":
        return "Professional"
    if x_lower=="doctor of philosophy" or x_lower=="postdoctoral" or x_lower=="post doc" or x_lower=="d.phil"\
    or x_lower=="dr" or x_lower=="dr." or x_lower=="dea" or x_lower=="dphil":
        return "PhD"
    if x_lower=="juris doctor" or x_lower=="doctor of law (jd)" or x_lower=="j.d.":
        return"JD"

    if "doctorate" in x_lower:
        return "PhD"
    if "bachelor of laws" in x_lower:
        return "LLB"
    if x_lower=="sb":
        return "BS"
    if "executive" in x_lower:
        return "Executive"
    if "high school" in x_lower:
        return np.nan
    if x_lower=="aa" or x_lower=="as" or x_lower=="aas" or x_lower=="dec":
        return "Associate"
    if x_lower=="doctor of medicine":
        return"MD"
    if x_lower=="s.m." or x_lower=="s.m" or x_lower=="sm":
        return "MS"
    if x_lower=="sb" or x_lower=='s.b.' or x_lower=="sc.b." or x_lower=="s.b" or x_lower=="scb":
        return "BS"
    if x=="Bachelor of Technology (B.Tech.)" or x=="Bachelors of Technology":
        return"BS"

    if "masters science" in x_lower:
        return "MS"
    if "m.tech" in x_lower:
        return "MTECH"
    if "m.d." in x_lower:
        return "MD"
    if "msc" in x_lower:
        return 'MS'
    if "/ms" in x_lower:
        return "MS"
    if ", ms" in x_lower:
        return "MS"
    if "m.eng" in x_lower :
        return "MS"
    if "ms " in x_lower:
        return "MS"
    if all(c in x_lower for c in ["post","doc"]):
        return "PhD"
    if all(c in x_lower for c in ["post","grad"]):
        return "Graduate"

    if "BA" in str(x):
        if "BBA" in str(x):
            return "BBA"
        elif "MBA" in str(x):
            return "MBA"
        else:
            return "BA"
    if "B.A" in str(x):
        if "B.B.A" in str(x):
            return "BBA"
        elif "M.B.A" in str(x):
            return "MBA"
        else:
            return "BA"

    if "bs " in x_lower:
        return "BS"
    if "bs." in x_lower:
        return 'BS'
    if "bme " in x_lower:
        return "BS"
    if all(c in x_lower for c in ["bachelor","art"]):
        return "BA"
    if all(c in x_lower for c in ["bachelor","science"]):
        return "BS"
    if all(c in x_lower for c in ["bachelor","degree"]):
        return "Bachelor"
    if "bachlor" in x_lower:
        return "Bachelor"
    if "bachellor" in x_lower:
        return "Bachelor"

    if "b.s." in x_lower:
        return "BS"
    if "b. s." in x_lower:
        return "BS"
    if "b.s." in x_lower:
        return "BS"
    if "b. s." in x_lower:
        return "BS"
    if "bs" in str(x).lower():
        return "BS"



    if x_lower=="mem" or x_lower=="mim":
        return "MM"
    if x_lower=="graduation" or x_lower=="diploma" or x_lower=="graduated" or \
    x_lower=="business administration" \
    or x_lower=="business" or x_lower =="marketing" or x_lower=="undergrad" or x_lower=="finance"\
    or x_lower=="economics" or x_lower=="electrical engineering" or x_lower=="law"\
    or x_lower=="management" or x_lower=="it" or x_lower=="cs" \
    or x_lower=="business management"\
    or x_lower=="license" or x_lower=="licence" or x_lower=="commerce" or x_lower=="licentiate"\
    or x_lower=="information technology" or x_lower=="licenciado" or x_lower=="computer science"\
    or x_lower=="law degree" or x_lower=="medical degree":
        x= 'Undergraduate'
    if x_lower=="graduate" or x_lower=="graduate diploma" or x_lower=='post graduate' \
    or x_lower=="postgraduate" or x_lower=="post graduate diploma" or x_lower=="advanced diploma"\
    or x_lower=="postgraduation" or x_lower=="post graduation" or x_lower=='graduate certificate'\
    or x_lower=="med" or x_lower=="pgdbm" or x_lower=="pg" or x_lower=="pgdm" or x_lower=="dess"\
    or x_lower=="engineering" or x_lower=="engineer" or x_lower=="dipl-ing"\
    or x_lower=="computer engineering" or x_lower=="mechanical engineering":
        return "Graduate"

    if "llb" in str(x).lower():
        return "LLB"

    if x_lower=="am":
        return "MA"

    if "certificate" in str(x).lower():
        return np.nan

    if x_lower=="bcom" or x_lower=="bm":
        x="BBA"
    if "bach" in str(x).lower():
        x="Bachelor"
    if "B." in str(x):
        if "M.B.A." in str(x):
            return "MBA"
        else:
            return "Bachelor"
    if "master" in str(x).lower():
        return "Master"
    if "M." in str(x):
        return "Master"
    if "engineering" in str(x).lower():
        x="Undergraduate"
    if "engineer" in str(x).lower():
        x="Undergraduate"
    if "baccalaureate" in str(x).lower():
        x=np.nan
    if "Ing" in str(x):
        x="Graduate"
    if "diploma" in str(x).lower():
        x=np.nan
    if "MA" in str(x):
        x="MA"
    if "ME" in str(x):
        x="MS"
    if "J.D" in str(x):
        x="JD"
    if "JD" in str(x):
        x="JD"
    if "MS" in str(x):
        x="MS"
    if "Eng" in str(x):
        x="Graduate"
    if "MPhil" in str(x):
        x="MPHIL"
    if "MBA " in str(x):
        x="MBA"
    if "hon" in str(x).lower():
        x=np.nan

    return x

def leveling(x):
    if x in ['BS', 'BA', 'Undergraduate', 'BBA',
        'Bachelor', 'BFA',
       'LLB']:
        return "undergrad"
    if x in [ 'Associate', 'Professional', 'Associate of Arts',
       'Associate of Science']:
        return "professional"
    if x in ['PhD']:
        return "phd"
    if x in ["MBA"]:
        return "MBA"
    if x in ['MS', 'JD', 'MA',
       'Master', 'Graduate', 'MD', 'Executive', 'LLM', 'MPHIL', 'MPA', 'MPH', 'MPP',
       'MM', 'MTECH', 'MPS', 'MHA']:
        return "graduate"

def degree_type_clean(degrees):
    #print(degrees.shape)
    degrees["degree_type"] = degrees.degree_type.map(degree_clean)
    degrees_clean = pd.DataFrame(degrees["degree_type"].value_counts())
    def cleaning(x):
        try:
            if degrees_clean.loc[x].degree_type >= 10:
                return x
            else:
                return np.nan
        except:
            return np.nan

    degrees["degree_type"]= degrees["degree_type"].map(cleaning)
    degrees["study_degree"] = degrees.degree_type.map(leveling)
    degrees = pd.concat([degrees,pd.get_dummies(degrees.study_degree)], axis=1)

    #print (degrees.sample(5))
    #print(degrees.shape)

    return degrees


def computer_science(x):
    for i in ["computer","informatic","information","it","internet technology", "cs ", " cs", "software"]:
        if i in x.lower():
            return 1
        return 0

def cs_topic(degrees):
    degrees["cs"] = degrees.subject.astype(str).map(computer_science)
    #print (degrees.sample(5))
    #print(degrees.shape)
    return degrees


def merge_people_level(people, degrees):
    #print(people.shape)
    degrees = degree_type_clean(degrees)
    degrees = cs_topic(degrees)
    degrees_per_person = degrees.groupby("object_id").sum().drop("id", axis=1)
    degrees_per_person["degree_count"] = degrees_per_person[["graduate","phd","professional","undergrad","MBA"]].sum(axis=1)
    degrees_per_person = degrees_per_person.reset_index().rename(columns={"object_id":"person_object_id"})
    degrees_per_person.cs = degrees_per_person.cs.map(lambda x: 1 if x >0 else 0)
    people = people.merge(degrees_per_person.fillna(value=0),on="person_object_id", how="left")
    people.loc[:,degrees_per_person.columns] = people.loc[:,degrees_per_person.columns].fillna(0)
    #print(people.shape)
    #print(people.sample(20))
    return people

def merge_company_level(people, degrees,companies, relationships):
    people = merge_people_level(people, degrees)
    relationships = relationships[relationships.founder][["person_object_id","relationship_object_id"]]\
    .merge(people, on="person_object_id", how="left")
    tmp = relationships[["relationship_object_id","phd",\
    "MBA","cs","graduate","undergrad","professional","degree_count"]]\
    .groupby("relationship_object_id",as_index=False).sum()
    tmp["MBA_bool"] = tmp.MBA.map(lambda x: 1 if x > 0 else 0)
    tmp["cs_bool"] = tmp.cs.map(lambda x: 1 if x > 0 else 0)
    tmp["phd_bool"] = tmp.phd.map(lambda x: 1 if x > 0 else 0)
    tmp2 = relationships.groupby("relationship_object_id",as_index=False)\
    .count()[["relationship_object_id","person_object_id"]]\
    .rename(columns={"person_object_id":"founder_count"})
    tmp3 = tmp.merge(tmp2, on="relationship_object_id")
    for i in ["phd","MBA","cs","graduate","undergrad","professional","degree_count"]:
        tmp3[i] = tmp3[i]/ tmp3.founder_count
    tmp3 = tmp3.rename(columns={"relationship_object_id":"id"})
    companies = companies.merge(tmp3, how="left", on="id")
    #print(companies.shape)
    #print(companies.sample(20))
    return companies


if __name__ == "__main__":
    degrees = pd.read_csv(os.path.join('..',"raw_data","degrees.csv")).drop(columns=["updated_at","created_at"])
    people = pd.read_csv(os.path.join('..',"raw_data","people.csv"))
    relationships = pd.read_csv(os.path.join('..',"raw_data","relationships.csv"))
    companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))
    merge_company_level(people, degrees,companies,relationships)

