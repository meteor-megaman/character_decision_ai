import numpy as np
import matplotlib.pyplot as plt

# mnist_data.npzからデータを読み込む
with np.load('out/mnist_data.npz') as data:
    x_test = data['x_test']

# x_testの最初の要素を選択
test_image = x_test[0]
test_image = test_image.reshape(28, 28)

# 画像として保存、余白を削除
plt.imshow(test_image, cmap='gray')
plt.axis('off')  # 軸表記を除去
plt.savefig('out/test_image.png', bbox_inches='tight', pad_inches=0)

print(28*28)