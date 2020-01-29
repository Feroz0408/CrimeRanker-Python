from flask import Flask, render_template, request, redirect, flash

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
    boundaries = request.form['boundaries']
    if boundaries:
        import Live_Tweets
    # boundaries = ""
    crime_data = Live_Tweets.crime_obj

    return render_template("/html/home.html", crime_data=crime_data)
    # return render_template("/html/home.html")


if __name__ == '__main__':
    app.run(debug=True)
