from flask import Flask
from flask import request
from run_placesCNN_basic import scene_detection
from urllib.parse import unquote

app = Flask(__name__)


@app.route("/classify_image")
def main():
    url = request.args.get("url")
    decoded_url = unquote(url)
    print("Recieved URL: ", decoded_url)
    print("\n...")
    scene = scene_detection(decoded_url)
    return scene


if __name__ == "__main__":
    app.run()
