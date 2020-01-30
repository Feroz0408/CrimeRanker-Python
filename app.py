import json
from flask import Flask, render_template, request, redirect, flash
import re
import sys
from importlib import reload  # Python 3.4+ only.
app = Flask(__name__)

# boundaries = re.search('var boundaries')
# if boundaries:
#     print(boundaries)


@app.route('/', methods=['GET', 'POST'])
def index():
    """
     Loads the homepage 
    """
    if request.args.get('state'):
        state = request.args.get('state')
        xmin = float(request.args.get('xmin'))
        ymin = float(request.args.get('ymin'))
        xmax = float(request.args.get('xmax'))
        ymax = float(request.args.get('ymax'))
        import Live_Tweets
        import CrimeModels
        import Stemming_Preprocessing
        reload(Live_Tweets)
        reload(CrimeModels)
        reload(Stemming_Preprocessing)
        sortedArray = Live_Tweets.startFetching(
            xmin, ymin, xmax, ymax)
        # os.remove("twitDB.json")
        return render_template("/html/home.html", crime_data=sortedArray, name=state)

    print(">>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<")
    print("reererhfdfdvsv")
    return render_template("/html/home.html")


@app.route('/Check_result', methods=['POST'])
def check():
    """
      Returns table data
    """
    global selected

    return render_template("/html/home.html", crime_data="")
    # return render_template("/html/home.html")


@app.route("/pass_val", methods=['POST'])
def new():
    import Live_Tweets
    # import CrimeModels
    # state = request.args.get('state')
    # xmin = float(request.args.get('xmin'))
    # ymin = float(request.args.get('ymin'))
    # xmax = float(request.args.get('xmax'))
    # ymax = float(request.args.get('ymax'))

    # print(xmin, ">>>>>>>>>>>>> Hererere")
    # print(">>>>>>>>>>>>>>>>>>>>")
    # newArray = Live_Tweets.startFetching(xmin, ymin, xmax, ymax)
    # print(">>>>>>.", state, xmin, ymin, xmax,
    #       ymax, ">>>>>???????????/", newArray)

    return render_template("/html/home.html")


if __name__ == '__main__':
    app.run(debug=True)
