import numpy as np
from keras.layers import Flatten
from tensorflow.keras.layers import Input, Dense, Multiply, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

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

def base_model():
    input_shape = (timestep, 2)
    mask_shape = (256,)

    model_input = Input(shape=input_shape)
    mask_input = Input(shape=mask_shape)

    x = Flatten()(model_input)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(32, activation='relu')(x)

    x = Dense(64, activation='relu')(encoded)
    x = Dense(128, activation='relu')(x)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = Reshape(input_shape)(decoded)

    autoencoder = Model(model_input, decoded)

    encoder = Model(model_input, encoded)

    decoder_input = Input(shape=(32,))
    x = Dense(64, activation='relu')(decoder_input)
    x = Dense(128, activation='relu')(x)
    decoded = Dense(np.prod(input_shape), activation='sigmoid')(x)
    decoded = Reshape(input_shape)(decoded)
    decoder = Model(decoder_input, decoded)




    x = Flatten()(encoded)

    x = Dense(256, activation='relu')(x)
    x = Dense(256, activation='relu')(x)

    output1 = Dense(5, activation='softmax', name='Bandwidth')(x)
    output2 = Dense(4, activation='softmax', name='Duration')(x)

    x3 = Multiply()([x, mask_input])
    output3 = Dense(5, activation='softmax', name='Class')(x3)

    model = Model(inputs=[model_input, mask_input], outputs=[output1, output2, output3])
    
    opt = Adam(clipnorm = 1.)
    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'], 
                  loss_weights=[1, 1, lambda_value], optimizer=opt, metrics=['accuracy'])
    return model

model = base_model()

model.fit([trainData, trainmask], [Y_train1, Y_train2, Y_train3],
          validation_data=([valData, valmask], [Y_val1, Y_val2, Y_val3]),
          batch_size=64, epochs=20, verbose=True, shuffle=True)

result = model.evaluate([testData, testmask], [Y_test1, Y_test2, Y_test3])
print(result)