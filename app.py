from flask import Flask, render_template, request, redirect, flash
# from testPrediction import PredictData

app = Flask(__name__)


@app.route('/')
def index():
    """
     Loads the homepage
    """
    return render_template("/html/home.html")


@app.route('/Check_result', methods=['GET', 'POST'])
def check():
    """
      Returns table data
    """

    # return render_template("/html/home.html", pred_Lists=pred_Lists, feature_Lists=feature_Lists, pos_score=pos_score,
    #                        neg_score=neg_score, neut_score=neut_score)
    return render_template("/html/home.html")


if __name__ == '__main__':
    app.run(debug=True)
