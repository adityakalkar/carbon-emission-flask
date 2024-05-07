from flask import Flask, render_template, request
import joblib
import pandas as pd
from collections import defaultdict
# from gevent.pywsgi import WSGIServer


def convertBodyWeight(x):
    d = defaultdict(lambda: 0)
    d["underweight"] = 1
    d["normal"] = 2
    d["overweight"] = 3
    d["obese"] = 4
    return d[x]

def convertGender(x):
    d = defaultdict(lambda: 0)
    d["male"] = 1
    d["female"] = 2
    return d[x]

def convertDiet(x):
    d = defaultdict(lambda: 0)
    d["vegan"] = 1
    d["vegetarian"] = 2
    d["pescatarian"] = 3
    d["omnivore"] = 4
    return d[x]

def convertEnergy(x):
    d = defaultdict(lambda: 0)
    d["natural gas"] = 1
    d["electricity"] = 2
    d["coal"] = 3
    d["wood"] = 4
    return d[x]

def convertShower(x):
    d = defaultdict(lambda:0)
    d["less frequently"] = 1
    d["daily"] = 2
    d["more frequently"] = 2
    d["twice a day"] = 4
    return d[x]

def convertTransport(x):
    d = defaultdict(lambda:0)
    d["walk/bicycle"] = 1
    d["public"] = 2
    d["private"] = 3
    return d[x]

def convertVehicleEnergy(x):
    d = defaultdict(lambda:0)
    d["electic"] = 1
    d["hybrid"] = 2
    d["petrol"] = 3
    d["lpg"] = 4
    d["diesel"] = 5
    return d[x]

def convertSocial(x):
    d = defaultdict(lambda:0)
    d["never"] = 1
    d["sometimes"] = 2
    d["often"] = 3
    return d[x]

def convertAir(x):
    d = defaultdict(lambda:0)
    d["never"] = 1
    d["rarely"] = 2
    d["frequently"] = 3
    d["very frequently"] = 4
    return d[x]

def convertTrash(x):
    d = defaultdict(lambda:0)
    d["small"] = 1
    d["medium"] = 2
    d["large"] = 3
    d["extra large"] = 4
    return d[x]

def convertEnergyEfficient(x):
    d = defaultdict(lambda:0)
    d["No"] = 1
    d["Sometimes"] = 2
    d["Yes"] = 3
    return d[x]

def convertRecycling(x):
    d = defaultdict(lambda:[0])
    d["Metal"] = 1
    d["Paper"] = 2
    d["Glass"] = 3
    d["Plastic"] = 4
    ans = []
    for item in x:
        ans.append(d[item])
    return len(ans)

def convertCooking(x):
    d = defaultdict(lambda:[0])
    d["Stove"] = 1
    d["Oven"] = 2
    d["Microwave"] = 3
    d["Grill"] = 4
    d["Airfryer"] = 5
    ans = []
    for item in x:
        ans.append(d[item])
    return len(ans)

# TEMPLATE = "/Users/adityakalkar/HTML/320 final project/templates"
# STATIC = "/Users/adityakalkar/HTML/320 final project/static"

# app = Flask(__name__,template_folder=TEMPLATE,
#             static_folder=STATIC)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df['Body Type'] = df['Body Type'].apply(lambda x: convertBodyWeight(x))
    df['Sex'] = df['Sex'].apply(lambda x: convertGender(x))
    df['Diet'] = df['Diet'].apply(lambda x: convertDiet(x))
    df['How Often Shower'] = df['How Often Shower'].apply(lambda x: convertShower(x))
    df['Heating Energy Source'] = df['Heating Energy Source'].apply(lambda x: convertEnergy(x))
    df['Transport'] = df['Transport'].apply(lambda x: convertTransport(x))
    df['Vehicle Type'] = df['Vehicle Type'].apply(lambda x: convertVehicleEnergy(x))
    df['Social Activity'] = df['Social Activity'].apply(lambda x: convertSocial(x))
    df['Frequency of Traveling by Air'] = df['Frequency of Traveling by Air'].apply(lambda x: convertAir(x))
    df['Waste Bag Size'] = df['Waste Bag Size'].apply(lambda x: convertTrash(x))
    df['Energy efficiency'] = df['Energy efficiency'].apply(lambda x: convertEnergyEfficient(x))
    df['Recycling'] = df['Recycling'].apply(lambda x: convertRecycling(x))
    df['Cooking_With'] = df['Cooking_With'].apply(lambda x: convertCooking(x))

    model = joblib.load('gradient_boosting_model.pkl')

    # Process the user data and make predictions using your machine learning model
    prediction = int(model.predict(df))

    # Render the prediction on the HTML page
    return render_template('result.html', prediction=prediction)

# if __name__ == "__main__":
#         http_server = WSGIServer(('0.0.0.0', 5000), app)
#         http_server.serve_forever()

app.run()
  
