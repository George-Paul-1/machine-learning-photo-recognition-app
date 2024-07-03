from taipy.gui import Gui
from taipy.gui import Html
from keras import models
from PIL import Image
import tensorflow
from keras import datasets, layers, models, losses, optimizers
import numpy as np


class_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}


model = models.load_model("/Users/georgepaul/Documents/personal/AI-photo-app/ml-photo-app/frontend/model/cifar-10-batches-py-model.keras")


def predict_image(model, path_to_img):
    img = Image.open(path_to_img)
    img = img.convert("RGB")
    img = img.resize((32, 32))
    data = np.asarray(img)
    data = data / 255
    logits = model.predict(np.array([data])[:1])
    probs = tensorflow.nn.softmax(logits).numpy()

    top_prob = probs.max()
    top_pred = class_names[np.argmax(probs)]

    return top_prob, top_pred


opt = optimizers.Adam(learning_rate=0.1)


content = ""
img_path = "https://placehold.co/600?text=No+Image+Available&font=roboto"
prob = 0
pred = ""


html_page = Html("""
<div class="container">
<h1>Machine Learning Photo App</h1>
<p>There is a prediction indication that ranges from 0 to 100. The greater the value, the more certain the machine model is that the prediction is true. This all depends on the Neural Network Builder model that we generated. The longer you train the model, the smarter it will get. If it does not have enough training time, it will make incorrect predictions.</p>
<div class="attachment">
<div><taipy:file_selector extensions=".png">{content}</taipy:file_selector></div>
<div><p>Choose an image from your computer to upload</p></div>
</div>
<div>
<taipy:image>{img_path}</taipy:image>
<taipy:indicator min="0" max="100" width="25vw" height="25vh" orientation="vertical" value="{prob}">{prob}</taipy:indicator>
</div>
<div class="prediction">
<p>Prediction:</p><div><taipy:text>{pred}</taipy:text></div>
</div>
</div>
""")

def on_change(state, var_name, var_val):
    if var_name == "content":
        top_prob, top_pred = predict_image(model, var_val)
        state.prob = round(top_prob * 100)
        state.pred = "Its a " + top_pred
        state.img_path = var_val


app = Gui(page=html_page)

if __name__ == "__main__":
    app.run(use_reloader=True, port=8000)