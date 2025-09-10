import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.manifold import TSNE

(x_train,_),(x_test,y_test)=mnist.load_data()
x_train,x_test=x_train.reshape(-1,784)/255.,x_test.reshape(-1,784)/255.

inp=Input((784,))
enc=Dense(32,activation='relu')(inp)
dec=Dense(784,activation='sigmoid')(enc)
auto=Model(inp,dec);enc_model=Model(inp,enc)
auto.compile('adam','binary_crossentropy')
auto.fit(x_train,x_train,epochs=10,batch_size=256,verbose=0)

enc_test=enc_model.predict(x_test)
X2d=TSNE(2,random_state=0).fit_transform(enc_test)
plt.scatter(X2d[:,0],X2d[:,1],c=y_test,cmap="tab10",s=5);plt.show()

dec_test=auto.predict(x_test[:10])
for i in range(10):
    plt.imshow(x_test[i].reshape(28,28),cmap="gray");plt.title("Original");plt.axis('off');plt.show()
    plt.imshow(dec_test[i].reshape(28,28),cmap="gray");plt.title("Reconstructed");plt.axis('off');plt.show()
