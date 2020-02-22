import json
from flask import Flask, render_template, request, redirect, flash
import re
import sys
from importlib import reload  # Python 3.4+ only.
from threading import Thread
app = Flask(__name__)
finished = False
th = Thread()
app.config["TEMPLATES_AUTO_RELOAD"] = True

# boundaries = re.search('var boundaries')
# if boundaries:
#     print(boundaries)

sortedArray = ""
name = ""


@app.route('/')
def index():
    """
     intialize landing page
    """
    if sortedArray:
        return render_template("/html/home.html", crime_data=sortedArray, name=state)
    return render_template("/html/home.html")


@app.route('/Check_result', methods=['GET', 'POST'])
def check():
    """
      Display Crime Prediction
    """
    global selected
    global sortedArray
    global state
    state = request.args.get('state')
    xmin = float(request.args.get('xmin'))
    ymin = float(request.args.get('ymin'))
    xmax = float(request.args.get('xmax'))
    ymax = float(request.args.get('ymax'))
    import Live_Tweets
    import FinalModel
    import Stemming_Preprocessing
    reload(Live_Tweets)
    reload(FinalModel)
    reload(Stemming_Preprocessing)
    sortedArray = {}

    print(">>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
    print("Boundaries For", state, xmin, ymin, xmax, ymax)
    print(">>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")

    sortedArray = Live_Tweets.startFetching(
        state, xmin, ymin, xmax, ymax)
    print(">>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
    print("Crime Ranking:", "State:", state)
    print(">>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")
    print(sortedArray[0]["name"], sortedArray[0]["value"], "%")
    print(sortedArray[1]["name"], sortedArray[1]["value"], "%")
    print(sortedArray[2]["name"], sortedArray[2]["value"], "%")
    print(sortedArray[3]["name"], sortedArray[3]["value"], "%")
    print(sortedArray[4]["name"], sortedArray[4]["value"], "%")
    print(">>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<")

    return render_template("/html/home.html", crime_data=sortedArray, name=state)


# @app.route("/pass_val", methods=['GET', 'POST'])
# def update():
#     if sortedArray:
#         return render_template("/html/home.html", crime_data=sortedArray, name=state)
#     return render_template("/html/home.html")


if __name__ == '__main__':
    app.run(debug=True)
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True, host='127.0.0.1')
