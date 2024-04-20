from tkinter import filedialog
import numpy as np
from PIL import Image
from CNN import get_model

model = get_model()

file_path = filedialog.askopenfilename()
image = Image.open(file_path)
resized_image = image.resize((32, 32))
resized_image.save("resized_image.jpg")

img = np.array(resized_image)
img = np.expand_dims(img, axis=0)
img = img / 255

predict = model.predict(img)
result = np.argmax(predict)

results = {0:"самолет", 1:"автомобиль", 2:"птица", 3:"кот", 4:"олень", 5:"собака", 6:"лягушка", 7:"лошадь", 8:"корабль", 9:"грузовик"}
print(predict[0][result])
print(results.get(result))