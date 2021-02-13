from datetime import datetime
import altair as alt
import joblib
import pandas as pd
import pytz
import streamlit as st
import os
from invesscience.joanna_clean_data import clean_cat, clean_country, clean_state, clean_date, clean_series\
, clean_participants, clean_amount, clean_rounds, clean_study, clean_school, clean_female,\
clean_founded_before, clean_worked_before, clean_founder, clean_degree_count

df = pd.read_csv('companies_test_3.csv')

st.markdown(f"# Invesscience ⚡️ \n # Invest in the most promising start-ups 🦄")

def convert_bool(x):
    if x=="Yes":
        return 1
    else:
        return 0

def colors(x, cut):
    if x < (3/4)*cut:
        return "lower"
    elif x > (5/4)*cut:
        return "higher"
    else:
        return "average"

def create_chart_cat(column,title):
    #pivot_table
    pivot_categories = pd.pivot_table(df, values='id', index=[column],
                columns=['target'], aggfunc= "count", margins = True)
    pivot_categories = pivot_categories.fillna(0).rename(columns={0:'no_exit', 1:'acquisition_or_ipo'}).sort_values(by="All")
    #pivot_categories[0] = pivot_categories[0]/pivot_categories.All
    pivot_categories['acquisition_or_ipo'] = pivot_categories['acquisition_or_ipo']/pivot_categories.All
    cut = pivot_categories.loc["All","acquisition_or_ipo"]
    pivot_categories["likelihood"] = pivot_categories["acquisition_or_ipo"].map(lambda x: colors(x,cut))
    pivot_categories = pivot_categories[['acquisition_or_ipo',"likelihood"]]

    c = alt.Chart(pivot_categories.reset_index()).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    y=alt.Y("acquisition_or_ipo:Q", title="Chances of IPO or Acquisition [%]"),
    x=alt.X(f"{column}:O", title=title),
    color= alt.condition(alt.datum[column] =="All", alt.value("#314e6e"), "likelihood")
    #"likelihood"
    ).properties(width=700, height=400)
    st.write(c)

def create_chart_cat_neg(column,title):
    #pivot_table
    pivot_categories = pd.pivot_table(df, values='id', index=[column],
                columns=['target'], aggfunc= "count", margins = True)
    pivot_categories = pivot_categories.fillna(0).rename(columns={0:'no_exit', 1:'acquisition_or_ipo'}).sort_values(by="All")
    #pivot_categories[0] = pivot_categories[0]/pivot_categories.All
    pivot_categories['acquisition_or_ipo'] = pivot_categories['acquisition_or_ipo']/pivot_categories.All
    cut = pivot_categories.loc["All","acquisition_or_ipo"]
    pivot_categories["likelihood"] = pivot_categories["acquisition_or_ipo"].map(lambda x: colors(x,cut))
    pivot_categories = pivot_categories[['acquisition_or_ipo',"likelihood"]]

    c = alt.Chart(pivot_categories.reset_index()).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    y=alt.Y("acquisition_or_ipo:Q", title="Chances of IPO or Acquisition [%]"),
    x=alt.X(f"{column}:O", title=title),
    color= alt.condition(alt.datum.likelihood=="average", alt.value("#4C78A8"), alt.value("#E45756"))
    #"likelihood"
    ).properties(width=630, height=400)
    st.write(c)

def create_chart_num(column,title):
    #pivot_table
    pivot_categories = pd.pivot_table(df, values='id', index=[column],
                columns=['target'], aggfunc= "count", margins = True)
    pivot_categories = pivot_categories.fillna(0).rename(columns={0:'no_exit', 1:'acquisition_or_ipo'})
    #pivot_categories[0] = pivot_categories[0]/pivot_categories.All
    pivot_categories['acquisition_or_ipo'] = pivot_categories['acquisition_or_ipo']/pivot_categories.All
    cut = pivot_categories.loc["All","acquisition_or_ipo"]
    #pivot_categories["likelihood"] = pivot_categories["acquisition_or_ipo"].map(lambda x: colors(x,cut))
    pivot_categories = pivot_categories[['acquisition_or_ipo'
    #,"likelihood"
    ]]

    c = alt.Chart(pivot_categories.reset_index()).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    y=alt.Y("acquisition_or_ipo:Q", title="Chances of IPO or Acquisition [%]"),
    x=alt.X(f"{column}:O", title=title),
    color= alt.condition(alt.datum[column] =="All", alt.value("#314e6e"), alt.value("#4C78A8"))
    #"likelihood"
    #314e6e
    ).properties(width=630, height=400)
    st.write(c)

def create_chart_cat_num(column,title):
    #pivot_table
    pivot_categories = pd.pivot_table(df, values='id', index=[column],
                columns=['target'], aggfunc= "count", margins = True)
    pivot_categories = pivot_categories.fillna(0).rename(columns={0:'no_exit', 1:'acquisition_or_ipo'})#.sort_values(by="All")
    #pivot_categories[0] = pivot_categories[0]/pivot_categories.All
    pivot_categories['acquisition_or_ipo'] = pivot_categories['acquisition_or_ipo']/pivot_categories.All
    cut = pivot_categories.loc["All","acquisition_or_ipo"]
    pivot_categories["likelihood"] = pivot_categories["acquisition_or_ipo"].map(lambda x: colors(x,cut))
    pivot_categories = pivot_categories[['acquisition_or_ipo',"likelihood"]]

    #oder
    list_1 = [pivot_categories.index[-1],pivot_categories.index[-2]]
    list_2 = list(pivot_categories.index[:-5])
    list_1.extend(list_2)

    c = alt.Chart(pivot_categories.drop(["2009","2010","2011","2012","2013"]).reset_index()).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    y=alt.Y("acquisition_or_ipo:Q", title="Chances of IPO or Acquisition [%]"),
    x=alt.X(f"{column}:O", title=title, sort=list_1),
    color= alt.condition(alt.datum[column] =="All", alt.value("#314e6e"), "likelihood")
    #"likelihood"
    ).properties(width=700, height=400)
    st.write(c)


def create_chart_bin_neg(column,title):
    #pivot_table
    pivot_categories = pd.pivot_table(df, values='id', index=[column],
                columns=['target'], aggfunc= "count", margins = True)
    pivot_categories = pivot_categories.fillna(0).rename(columns={0:'no_exit', 1:'acquisition_or_ipo'})
    #pivot_categories[0] = pivot_categories[0]/pivot_categories.All
    pivot_categories['acquisition_or_ipo'] = pivot_categories['acquisition_or_ipo']/pivot_categories.All
    cut = pivot_categories.loc["All","acquisition_or_ipo"]
    pivot_categories["likelihood"] = pivot_categories["acquisition_or_ipo"].map(lambda x: colors(x,cut))
    pivot_categories = pivot_categories[['acquisition_or_ipo'
    ,"likelihood"
    ]]

    c = alt.Chart(pivot_categories.drop("All").reset_index()).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    y=alt.Y("acquisition_or_ipo:Q", title="Chances of IPO or Acquisition [%]"),
    x=alt.X(f"{column}:O", title=title),
    color= alt.condition(alt.datum.likelihood=="average", alt.value("#4C78A8"), alt.value("#E45756"))
    #"likelihood"
    ).properties(width=630, height=400)
    st.write(c)

def create_chart_bin_pos(column,title):
    #pivot_table
    pivot_categories = pd.pivot_table(df, values='id', index=[column],
                columns=['target'], aggfunc= "count", margins = True)
    pivot_categories = pivot_categories.fillna(0).rename(columns={0:'no_exit', 1:'acquisition_or_ipo'})
    #pivot_categories[0] = pivot_categories[0]/pivot_categories.All
    pivot_categories['acquisition_or_ipo'] = pivot_categories['acquisition_or_ipo']/pivot_categories.All
    cut = pivot_categories.loc["All","acquisition_or_ipo"]
    pivot_categories["likelihood"] = pivot_categories["acquisition_or_ipo"].map(lambda x: colors(x,cut))
    pivot_categories = pivot_categories[['acquisition_or_ipo'
    ,"likelihood"
    ]]

    c = alt.Chart(pivot_categories.reset_index()).mark_bar(
    cornerRadiusTopLeft=3,
    cornerRadiusTopRight=3).encode(
    y=alt.Y("acquisition_or_ipo:Q", title="Chances of IPO or Acquisition [%]"),
    x=alt.X(f"{column}:O", title=title),
    color= alt.condition(alt.datum[column] =="All", alt.value("#314e6e"), "likelihood")
    #alt.condition(alt.datum.likelihood=="average", alt.value("#4C78A8"), alt.value("#F1853B"))

    #"likelihood"
    ).properties(width=630, height=400)
    st.write(c)


def main(reference="a"):
    analysis = st.sidebar.selectbox("choose activity", ["prediction", "data visualization"])
    if analysis == "data visualization":
        st.header("Insights Data Analysis based on Crunchbase Dataset")
        #st.markdown("**Have fun immplementing your own Taxifare Dataviz**")
        choice = st.selectbox("Insights", ["General","Location","Funding","Founding team composition", "Founding team education",
            "Founding team experience"], index=0)
        if choice == "General":
            #Category
            st.subheader("Exits according to category")
            st.markdown("- **network_hosting, web and public relations** are more likely to have an exit")
            st.markdown("- **medical and analytics** are less likely to have an exit")
            #st.bar_chart(category_chart['acquisition_or_ipo'])
            create_chart_cat('category_code',"Categories")
            st.subheader("Exits according to founded year")
            create_chart_cat_num("founded_at", "Founded Year")
        if choice == "Location":
            #Country
            st.subheader("Exits according to Country")
            create_chart_cat('country_code',"Countries")
            #State
            st.subheader("Exits according to US State")
            create_chart_cat('state_code',"States")

        if choice == "Founding team composition":
            #Founders
            st.subheader("Exits according to number of founders in the founding team")
            create_chart_cat('founder_count', "# Founders")
            #Female ratio
            st.subheader("Exits according to presence of women in the founding team")
            create_chart_cat('female_ratio', "Presence of women in founding team")

        if choice == "Founding team education":
            #Number of degrees
            st.subheader("Exits according to average number of diplomas per founder")
            create_chart_bin_pos('degree_count', "average university diplomas obtained per founder")

            # top 20 school
            st.subheader("Exits according to studies at top 20 universities by founders (QS ranking 2018)")
            create_chart_bin_pos('top_20_bool', "studies at top 20 university in the founding team")
            #CS ratio
            st.subheader("Exits according to study of Computer Science by founders")
            create_chart_bin_pos('cs_bool', "CS studies in the founding team")
            #MBA bool
            st.subheader("Exits according to MBAs completed by founders")
            create_chart_bin_pos('MBA_bool', "MBAs in the founding team")

            #Phd bool
            st.subheader("Exits according to PhDs completed by founders")
            create_chart_bin_pos('phd_bool', "PhDs in the founding team")

            #Graduate bool
            st.subheader("Exits according to graduate degrees completed by founders")
            create_chart_bin_pos('graduate', "Graduate degrees in the founding team")

            #Undergraduate bool
            st.subheader("Exits according to undergraduate degrees completed by founders")
            create_chart_bin_pos('undergrad', "Undergraduate degrees in the founding team")

            #Professional bool
            st.subheader("Exits according to professional degrees completed by founders")
            create_chart_bin_pos('professional', "Professional degrees in the founding team")




        if choice == "Founding team experience":
            # top 20 school
            st.subheader("Exits according to work at previous companies")
            create_chart_bin_pos('mean_comp_worked_before', "mean # company founders worked at before")
            # top 20 school
            st.subheader("Exits according to founding of previous company")
            create_chart_bin_pos('mean_comp_founded_before', "company founded before by founders (Yes/No)")

        if choice == "Funding":
            # rounds before a
            st.subheader("Exits according to number of funding rounds before Series A")
            create_chart_bin_pos('rounds_before_a', "# rounds before Series A")
            # participants at series a
            st.subheader("Exits according to number of investors at Series A")
            create_chart_bin_pos(f'participants_{reference}', "# investors at Series A")
            # amount raised at series a
            st.subheader("Exits according to amount raised at Series A")
            create_chart_bin_pos(f'raised_amount_usd_{reference}', "Amount raised at series A")


















    if analysis == "prediction":
        st.markdown("**This model will help you decide whether you should invest in the Series A of a company of your choice.**")

        pipeline = joblib.load('xgboost_2014_a.joblib')
        print("loaded model")

        st.header("Please tell us more about your investment opportunity")

        # COLS = ['category_code', 'country_code', 'state_code', 'founded_at', f'timediff_founded_series_{reference}',
        #          f'time_diff_series_{reference}_now', f'participants_{reference}', f'raised_amount_usd_{reference}',
        #          'rounds_before_a', 'mean_comp_worked_before', 'founder_count', 'graduate', 'MBA_bool', 'cs_bool', 'top_20_bool', 'mean_comp_founded_before',
        #          'female_ratio',
        #          ]
        COLS = ['category_code', 'country_code','state_code', 'founded_at',f'timediff_founded_series_{reference}',
                 f'time_diff_series_{reference}_now',f'participants_{reference}', f'raised_amount_usd_{reference}',
                 'rounds_before_a', 'mean_comp_worked_before', 'founder_count', 'degree_count','graduate',
                 'undergrad','professional', 'MBA_bool',
                                        'cs_bool', 'phd_bool', 'top_20_bool', 'mean_comp_founded_before', 'female_ratio']


        categories = ['advertising',
                         'analytics',
                         'biotech',
                         'cleantech',
                         'consulting',
                         'ecommerce',
                         'enterprise',
                         'finance',
                         'hardware',
                         'medical',
                         'mobile',
                         'music',
                         'network_hosting',
                         'public_relations',
                         'search',
                         'security',
                         'semiconductor',
                         'social',
                         'software',
                         'web',"other_category"]

        #founded_ats = ['pre_1996', '1996', '1997', '1998', '1999', '2000','2001', '2002','2003', '2004', '2005', '2006', '2007', '2008', '2009',
       #'2010', '2011', '2012', '2013', "2014","2015","2016","2017","2018", "2019", "2020", "2021"]

        countries = ['CAN',
                     'CHN',
                     'DEU',
                     'FRA',
                     'GBR',
                     'IND',
                     'ISR',
                     'SWE',
                     'USA',
                     'other_country']
        states = ['not_US', 'CA',
                 'CO',
                 'FL',
                 'IL',
                 'MA',
                 'MD',
                 'NC',
                 'NJ',
                 'NY',
                 'PA',
                 'TX',
                 'VA',
                 'WA',
                 'other_state']

        universities = ['Massachusetts Institute Of Technology (MIT) ',
                         'Stanford University',
                         'Harvard University',
                         'California Institute Of Technology (CALTECH)',
                         'University Of Cambridge',
                         'University Of Oxford',
                         'UCL (University College London)',
                         'Imperial College London',
                         'University Of Chicago',
                         'Eth Zurich (Swiss Federal Institute Of Technology)',
                         'Nanyang Technological University (Ntu)',
                         'Ecole Polytechnique Fédérale De Lausanne (EPFL)',
                         'Princeton University',
                         'Cornell University',
                         'National University Of Singapore (NUS)',
                         'Yale University',
                         'Johns Hopkins University',
                         'Columbia University',
                         'University Of Pennsylvania',
                         'Australian National University (ANU)']




        # inputs from user
        st.subheader("1. Personal behaviour")
        time_diff_series_now =  st.number_input("How long are you ready to wait for returns?",min_value=1,max_value=10 ,value=5)

        st.subheader("2. General Information")
        category_code = st.selectbox("Category", categories, index=0)
        founded_at = st.number_input("Founding year",min_value=1900,value=2018)
        country_code = st.selectbox("Country", countries ,index=8)
        state_code = st.selectbox("State", states , index=1)

        st.subheader("3. Investment rounds information")
        participants = st.number_input("# investors in current round (0 if amount is not know yet)",min_value=0,value=0)
        raised_amount_usd = st.number_input("Raised amount in current round (0 if amount is not know yet)",min_value=0,value=0)
        rounds_before_a = st.number_input("# funding rounds before current",min_value=0,value=0)

        st.subheader("4. Founding team information")
        founder_count = st.number_input("# founders",min_value=0,value=0)
        female_ratio = st.number_input("# female founders",min_value=0,value=0)
        degree_count = st.number_input("Average # university diplomas obtained by founder",min_value=0.0,value=1.0, max_value=10.0, step=0.01)
        undergrad = st.radio(label="Any founder with an undergraduate degree?", options=["Yes", "No"])
        graduate = st.radio(label="Any founder with a graduate degree?", options=["Yes", "No"])
        professional = st.radio(label="Any founder with a professional degree?", options=["Yes", "No"])
        MBA_bool = st.radio(label="Any founder with an MBA?", options=["Yes", "No"])
        cs_bool = st.radio(label="Any founder with a Computer Science degree?", options=["Yes", "No"])
        phd_bool = st.radio(label="Any founder with a PhD?", options=["Yes", "No"])
        top_20_bool = st.radio(label="Any founder studied in any of the universities below?", options=["Yes", "No"])
        st.write(universities)
        mean_comp_founded_before = st.radio(label="Any founder a founded company before this one?", options=["Yes", "No"])
        mean_comp_worked_before = st.number_input("Average # companies founders worked at before",min_value=0.0,value=1.0, max_value=10.0, step=0.01)


        ## formatting

        investment_datetime = datetime.utcnow().year
        timediff_founded_series = round(investment_datetime - int(founded_at))


        formated_input = {
                #"normalized_name":input["normalized_name"].lower()
                "category_code": clean_cat(category_code),
                "country_code": clean_country(country_code),
                "state_code": clean_state(state_code),
                "founded_at": clean_date(int(founded_at)),
                f"participants_{reference}": clean_participants(int(participants)),
                f'raised_amount_usd_{reference}': clean_amount(int(raised_amount_usd)),
                f'timediff_founded_series_{reference}': timediff_founded_series,
                'rounds_before_a': clean_rounds(int(rounds_before_a)),
                "degree_count": clean_degree_count(int(degree_count)),
                "graduate" : convert_bool(graduate),
                "undergrad" : convert_bool(undergrad),
                "professional" : convert_bool(professional),
                "MBA_bool" : convert_bool(MBA_bool),
                "cs_bool" : convert_bool(cs_bool),
                "phd_bool" : convert_bool(phd_bool),
                "founder_count" : clean_founder(int(founder_count)),
                "top_20_bool" : convert_bool(top_20_bool),
                "female_ratio" : clean_female(int(female_ratio)),
                "mean_comp_founded_before" : convert_bool(mean_comp_founded_before),
                "mean_comp_worked_before" : clean_worked_before(mean_comp_worked_before),
                f"time_diff_series_{reference}_now" : clean_series(time_diff_series_now)
                }



# inputs =        [{
#     "normalized_name":"Facebook",
#     "category_code": "medical",
#     "founded_at": "2012-12-03 13:10:00",
#     "description": "lorem ipsum",
#     "country_code": "USA",
#     "state_code":  "CA",
#     f"participants_{reference}": 1,
#     f"raised_amount_usd_{reference}": 0,
#     f'rounds_before_a': 1,
#     "graduate": 0,
#     "undergrad": 0,
#     "professional": 0,
#     "MBA_bool" : 0,
#     "cs_bool": 1,
#     "phd_bool": 0,
#     "founder_count":4,
#     "top_20_bool": 0,
#     "female_ratio": 1,
#     "mean_comp_founded_before":1,
#     "mean_comp_worked_before": 2,
#     f"time_diff_series_{reference}_now" : 5
#     }]



        X = pd.DataFrame([formated_input])
        print(X.T)
        X = X[COLS]
        if type(reference) != str:
            X = X.drop(columns="rounds_before_a")
        results = pipeline.predict_proba(X)
        print(results)

        st.header('**Our Prediction**')

        st.write(f"**The probability for this company to exit within {time_diff_series_now} years is**", int(round(results[0][1]*100)), "%")
        st.write(f"<h1 style='text-align: center; color: red;'> {int(round(results[0][1]*100))} Some title</h1>", unsafe_allow_html=True)
        if results[0][1] >= 0.8:
            st.write(f"You might have found the next 🦄")
        if results[0][1] < 0.5:
            st.write(f"We're not sure this is the right opportunity for you at the moment 👀")
        if 0.5 <= results[0][1] < 0.8:
            st.write(f"This opportunity has potential, you should further look into it ⭐️")

        # st.map(data=data)


# print(colored(proc.sf_query, "blue"))
# proc.test_execute()
if __name__ == "__main__":
    #df = read_data()
    main()
