from flask import Flask
from flask import render_template

app = Flask(__name__)

@app.route("/")
def Index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=False)

"python version = 3.8.8"