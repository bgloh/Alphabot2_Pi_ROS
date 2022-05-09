import numpy as np
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms

model = "Test.onnx"
img = Image.open("./TestImage.jpg")
resize = transforms.Resize([480, 640])
img = np.array(resize(img))
to_tensor = transforms.ToTensor()
img = to_tensor(img)
# print(img.shape)
# order = (0, 3, 1, 2)
input = np.expand_dims(img, axis=0)
# input = np.transpose(exResize, order)
# print(input.shape)
sess = onnxruntime.InferenceSession(model)
output = sess.run(["discrete_actions"], {"obs_0": input, "action_masks": np.array([[1., 1., 1., 1., 1.]]).astype(np.float32)})
print(output)
#
# sess.get_modelmeta()