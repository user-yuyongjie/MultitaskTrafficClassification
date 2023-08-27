"""在优化深度学习模型的过程中，有很多种方法可以尝试，包括调整网络架构、修改损失函数、改变优化器的参数、使用预训练模型、改变训练数据的处理方式等等。以下提供一些可能的优化建议：

使用更复杂的网络架构：更复杂的网络架构可能能够捕捉到更多的数据特征，从而提高模型的性能。例如，可以考虑使用更多的卷积层或者全连接层，或者使用不同类型的层，如循环神经网络（RNN）层
或者自注意力（Self-Attention）层。

修改损失函数或优化器：可以尝试使用不同的损失函数或者优化器，或者改变优化器的参数，如学习率、动量等。

使用预训练模型：如果有可用的预训练模型，可以使用迁移学习的方式，将预训练模型的部分层作为新模型的一部分，只训练剩余的层。

改变数据处理方式：可以尝试对数据进行不同的预处理操作，如归一化、标准化等。此外，也可以使用数据增强的方式增加训练数据的多样性。

使用早停（Early Stopping）和模型检查点（Model Checkpoint）：早停可以防止模型过拟合，而模型检查点可以在训练过程中保存性能最好的模型。"""
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.layers import Dense
from keras.layers import multiply
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Activation
from keras.optimizers import Adam

timestep = 60
np.random.seed(10)

num_class = 5
train_sample_per_class = 20
lambda_value = 1


trainData = np.load("trainData.npy")
trainlabel = np.load("trainLabel.npy")
# trainData = trainData[:, -timestep*2:]
# trainlabel = trainlabel[:, -timestep*2:]
trainData = trainData[:, :timestep*2]
trainlabel = trainlabel[:, :timestep*2]
trainlabel = trainlabel.astype(int)

trainmask = np.zeros((trainlabel.shape[0],256))

class_counter = np.zeros((num_class))
train_size = trainlabel.shape[0]
j = 0
for i in range(train_size):
    class_id = trainlabel[i,2] - 1
    if class_counter[class_id] < train_sample_per_class:
        trainmask[i, :] = 1
        j += 1
        class_counter[class_id] += 1
print("unmasked samples: ", str(np.sum(trainmask==1)/256))


valData = np.load("valData.npy")
valLabel = np.load("valLabel.npy")
# testData = testData[:, -timestep*2:]
# testLabel = testLabel[:, -timestep*2:]
valData = valData[:, :timestep*2]
valLabel = valLabel[:, :timestep*2]

valLabel = valLabel.astype(int)
valmask = np.ones((valLabel.shape[0], 256))
valmask[:,:]=1


testData = np.load("testData.npy")
testLabel = np.load("testLabel.npy")
# testData = testData[:, -timestep*2:]
# testLabel = testLabel[:, -timestep*2:]
testData = testData[:, :timestep*2]
testLabel = testLabel[:, :timestep*2]

testLabel = testLabel.astype(int)
testmask = np.ones((testLabel.shape[0], 256))
testmask[:,:]=1

for i in range(trainlabel.shape[0]):
    #Categorizing Bandwidth
    if trainlabel[i, 0] < 10000:
        trainlabel[i, 0] = 1
    elif trainlabel[i, 0] < 50000:
        trainlabel[i, 0] = 2
    elif trainlabel[i, 0] < 100000:
        trainlabel[i, 0] = 3
    elif trainlabel[i, 0] < 1000000:
        trainlabel[i, 0] = 4
    else:
        trainlabel[i, 0] = 5
    #Categorizing Duration
    if trainlabel[i, 1] < 10:
        trainlabel[i, 1] = 1
    elif trainlabel[i, 1] < 30:
        trainlabel[i, 1] = 2
    elif trainlabel[i, 1] < 60:
        trainlabel[i, 1] = 3
    else:
        trainlabel[i, 1] = 4

for i in range(valLabel.shape[0]):
    #Categorizing Bandwidth
    if valLabel[i, 0] < 10000:
        valLabel[i, 0] = 1
    elif valLabel[i, 0] < 50000:
        valLabel[i, 0] = 2
    elif valLabel[i, 0] < 100000:
        valLabel[i, 0] = 3
    elif valLabel[i, 0] < 1000000:
        valLabel[i, 0] = 4
    else:
        valLabel[i, 0] = 5
    #Categorizing Duration
    if valLabel[i, 1] < 10:
        valLabel[i, 1] = 1
    elif valLabel[i, 1] < 30:
        valLabel[i, 1] = 2
    elif valLabel[i, 1] < 60:
        valLabel[i, 1] = 3
    else:
        valLabel[i, 1] = 4


for i in range(testLabel.shape[0]):
    #Categorizing Bandwidth
    if testLabel[i, 0] < 10000:
        testLabel[i, 0] = 1
    elif testLabel[i, 0] < 50000:
        testLabel[i, 0] = 2
    elif testLabel[i, 0] < 100000:
        testLabel[i, 0] = 3
    elif testLabel[i, 0] < 1000000:
        testLabel[i, 0] = 4
    else:
        testLabel[i, 0] = 5
    #Categorizing Duration
    if testLabel[i, 1] < 10:
        testLabel[i, 1] = 1
    elif testLabel[i, 1] < 30:
        testLabel[i, 1] = 2
    elif testLabel[i, 1] < 60:
        testLabel[i, 1] = 3
    else:
        testLabel[i, 1] = 4


train_size = trainlabel.shape[0]
Y_train1 = np.zeros((train_size,5))
Y_train1[np.arange(train_size),trainlabel[:,0]-1] = 1
Y_train2 = np.zeros((train_size,4))
Y_train2[np.arange(train_size),trainlabel[:,1]-1] = 1
Y_train3 = np.zeros((train_size,5))
Y_train3[np.arange(train_size),trainlabel[:,2]-1] = 1

val_size = valLabel.shape[0]
Y_val1 = np.zeros((val_size,5))
Y_val1[np.arange(val_size),valLabel[:,0]-1] = 1
Y_val2 = np.zeros((val_size,4))
Y_val2[np.arange(val_size),valLabel[:,1]-1] = 1
Y_val3 = np.zeros((val_size,5))
Y_val3[np.arange(val_size),valLabel[:,2]-1] = 1

test_size = testLabel.shape[0]
Y_test1 = np.zeros((test_size,5))
Y_test1[np.arange(test_size),testLabel[:,0]-1] = 1
Y_test2 = np.zeros((test_size,4))
Y_test2[np.arange(test_size),testLabel[:,1]-1] = 1
Y_test3 = np.zeros((test_size,5))
Y_test3[np.arange(test_size),testLabel[:,2]-1] = 1

# trainData = np.expand_dims(trainData, axis=-1)
# testData = np.expand_dims(testData, axis=-1)
trainData = trainData.reshape((trainData.shape[0], timestep, 2))
testData = testData.reshape((testData.shape[0], timestep, 2))
valData = valData.reshape((valData.shape[0], timestep, 2))


def improved_model():

    model_input = Input(shape=(timestep,2))
    mask_input = Input(shape=(256,))

    x = Conv1D(32, 3, activation='relu')(model_input)
    x = Conv1D(32, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(64, 3, activation='relu')(x)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Conv1D(128, 3, activation='relu')(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=(2))(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)

    output1 = Dense(5, activation='softmax', name='Bandwidth')(x)

    output2 = Dense(4, activation='softmax', name='Duration')(x)

    x3 = multiply([x,mask_input])
    output3 = Dense(5, activation='softmax', name='Class')(x3)

    model = Model(inputs=[model_input,mask_input], outputs=[output1, output2, output3])
    opt = Adam(clipnorm = 1.)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], loss_weights=[1,1,lambda_value], optimizer=opt, metrics=['accuracy'])

    return model

model = improved_model()

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

model.fit([trainData,trainmask], [Y_train1, Y_train2, Y_train3],
          validation_data = ([valData, valmask], [Y_val1, Y_val2, Y_val3]),
          batch_size = 64, epochs = 20, verbose = True, shuffle = True,
          callbacks=[early_stopping, model_checkpoint])

result = model.evaluate([testData, testmask], [Y_test1, Y_test2, Y_test3])
print(result)