from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    name = "Karthik"
    return render_template("index.html", name=name)


@app.route("/about")
def about():
    return "This is a simple Flask app."


@app.route("/submit", methods=["POST"])
def submit():
    user_name = request.form.get("user_name")
    return f"Hello, {user_name}!"


if __name__ == "__main__":
    app.run(debug=True)
