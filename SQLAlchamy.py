from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime
import logging

app = Flask(__name__)

# Configuration
app.config["SECRET_KEY"] = "mysecretkey"

# Logging setup
logging.basicConfig(level=logging.INFO)

# Home Route
@app.route("/")
def index():
    app.logger.info("Home page accessed")
    return render_template("index.html")


# About Route
@app.route("/about")
def about():
    company = "ShambuAI"
    year = datetime.now().year
    return render_template(
        "about.html",
        company=company,
        year=year
    )


# Contact Route
@app.route("/contact", methods=["GET", "POST"])
def contact():

    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        app.logger.info(f"Contact Form Submitted by {name}")

        return render_template(
            "success.html",
            name=name
        )

    return render_template("contact.html")


# Dynamic Route
@app.route("/user/<username>")
def user_profile(username):
    return f"Welcome {username}"


# API Route
@app.route("/api/data")
def api_data():

    data = {
        "name": "Karthik",
        "company": "ShambuAI",
        "python_version": "3.12.8",
        "status": "Running"
    }

    return jsonify(data)


# Redirect Example
@app.route("/home")
def home():
    return redirect(url_for("index"))


# Error Handling
@app.errorhandler(404)
def page_not_found(error):
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_server_error(error):
    return render_template("500.html"), 500


# Main Entry
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
