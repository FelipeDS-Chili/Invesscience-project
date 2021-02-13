from datetime import datetime
import os
import joblib
import pandas as pd
import numpy as np
import pytz
from flask import Flask
from flask import request
from flask_cors import CORS
from termcolor import colored
from invesscience.joanna_clean_data import clean_cat, clean_country, clean_state, clean_date, clean_series\
, clean_participants, clean_amount, clean_rounds, clean_study, clean_school, clean_female,\
clean_founded_before, clean_worked_before, clean_founder

app = Flask(__name__)
CORS(app)

PATH_TO_MODEL = os.path.join("monday_model.joblib")

def format_input(input, reference='a'):
    mytz = pytz.timezone('Europe/Paris')
    #investment_datetime = pytz.utc.localize(datetime.utcnow(), is_dst=None).astimezone(mytz)
    investment_datetime = datetime.utcnow()
    input[f'timediff_founded_series_{reference}'] = round((investment_datetime - pd.to_datetime(input["founded_at"]))/np.timedelta64(12, 'M'))
    input["founded_at"] = pd.to_datetime(input["founded_at"]).year #.dt.strftime('%Y')
    input["founded_at"] = clean_date(input["founded_at"])


    default_params = {
        "rounds_before_a": 0,
        #"id": "new"
        #"pickup_longitude": NYC_DEFAULT_LNG,
        #"pickup_datetime": str(pickup_datetime),
        #"key": str(pickup_datetime)
        }
    for k, v in default_params.items():
        input.setdefault(k, v)

    formated_input = {
        #"normalized_name":input["normalized_name"].lower()
        "category_code": clean_cat(input["category_code"].lower()),
        "country_code": clean_country(input["country_code"].upper()),
        "state_code": clean_state(input["state_code"].upper()),
        "founded_at": input["founded_at"],
        f"participants_{reference}": clean_participants(float(input[f"participants_{reference}"])),
        f'raised_amount_usd_{reference}': clean_amount(int(input[f"raised_amount_usd_{reference}"])),
        f'timediff_founded_series_{reference}': input[f'timediff_founded_series_{reference}'],
        'rounds_before_a': clean_rounds(float(input[f'rounds_before_a'])),
        "graduate" : clean_study(float(input['graduate'])),
        "undergrad" : clean_study(float(input['undergrad'])),
        "professional" : clean_study(float(input['professional'])),
        "MBA_bool" : int(input["MBA_bool"]),
        "cs_bool" : int(input["cs_bool"]),
        "phd_bool" : int(input["phd_bool"]),
        "founder_count" : clean_founder(float(input['founder_count'])),
        "top_20_bool" : clean_school(float(input["top_20_bool"])),
        "female_ratio" : clean_female(float(input["female_ratio"])),
        "mean_comp_founded_before" : clean_founded_before(float(input["mean_comp_founded_before"])),
        "mean_comp_worked_before" : clean_worked_before(float(input['mean_comp_worked_before'])),
        f"time_diff_series_{reference}_now" : clean_series(int(input[f"time_diff_series_{reference}_now"]))
        }


    return formated_input


pipeline_def = {'pipeline': joblib.load(PATH_TO_MODEL),
              'from_gcp': False}


@app.route('/')
def index():
    return 'OK'


@app.route('/predict_exit', methods=['GET', 'POST'])
def predict_exit(reference='a'):
    # get the current time
    # need to diplay list of top universities
    COLS = ['category_code', 'country_code', 'state_code', 'founded_at', f'timediff_founded_series_{reference}',
                 f'time_diff_series_{reference}_now', f'participants_{reference}', f'raised_amount_usd_{reference}',
                 'rounds_before_a', 'mean_comp_worked_before', 'founder_count', 'graduate', 'MBA_bool', 'cs_bool', 'top_20_bool', 'mean_comp_founded_before',
                 'female_ratio',
                 ]

    """
    Expected input
        {
        "normalized_name":"Facebook"
        "category_code": "social",
        "founded_at": datetime,
        "description": "lorem ipsum",
        "country_code": "USA",
        "state_code":  "CA",
        f"participants{reference}": 2,
        f"raised_amount_usd_{reference}" 100000,
        f'rounds_before_a": 3,
        "graduate": 1,
        "undergrad": 1,
        "professional": 1,
        "MBA_bool" : 0,
        "cs_bool": 0,
        "phd_bool": 0,
        "founder_count":4,
        "top_20_bool": 1,
        "female_ratio": 1,
        "mean_comp_founded_before":1,
        "mean_comp_worked_before": 2,
        "time_diff_series_{reference}_now" : 5
        }
    :return: {"predictions": [1]}

    """
    inputs = request.get_json()
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs = [format_input(point,reference=reference) for point in inputs]
    # Here wee need to convert inputs to dataframe to feed as input to our pipeline
    # Indeed our pipeline expects a dataframe as input
    X = pd.DataFrame(inputs)
    # Here we specify the right column order
    X = X[COLS]
    if type(reference) != str:
        X = X.drop(columns="rounds_before_a")
    print(X.T)
    pipeline = pipeline_def["pipeline"]
    results = pipeline.predict(X)
    results = [round(float(r), 3) for r in results]
    return {"predictions": results}


# @app.route('/set_model', methods=['GET', 'POST'])
# def set_model():
#     inputs = request.get_json()
#     model_dir = inputs["model_directory"]
#     pipeline_def["pipeline"] = download_model(model_directory=model_dir, rm=True)
#     pipeline_def["from_gcp"] = True
#     return {"reponse": f"correctly got model from {model_dir} directory on GCP"}


if __name__ == '__main__':
    #reference="a"
    #print(predict_exit())
    #print(format_input(inputs))
    app.run(host='127.0.0.1', port=8080, debug=True)
