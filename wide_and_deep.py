import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Dropout, SpatialDropout1D, Activation, concatenate
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import ReLU, PReLU, LeakyReLU, ELU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "gender", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income_bracket"
]

LABEL_COLUMN = 'label'

CATEGORICAL_COLUMNS = [
    "workclass", "education", "marital_status", "occupation", "relationship",
    "race", "gender", "native_country"
]

CONTINUOUS_COLUMNS = [
    "age", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]

def preprocessing():
    # 数据加载
    train_data = pd.read_csv(r'./dataset/adult.data', names=COLUMNS)
    # 缺失值删除
    train_data.dropna()
    test_data = pd.read_csv(r'./dataset/adult.test', names=COLUMNS, skiprows=1)
    test_data.dropna()

    all_data = pd.concat([train_data, test_data], axis=0)
    all_data[LABEL_COLUMN] = all_data['income_bracket'].apply(lambda x: '>50K' in x).astype(int)
    all_data.drop(columns = ['income_bracket'], inplace = True)

    # 标签y
    y = all_data[LABEL_COLUMN].values

    # 数据x
    all_data.pop(LABEL_COLUMN)

    # 类别型的label encoding
    for c in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        all_data[c] = le.fit_transform(all_data[c])

    # 分别取出train和test的特征和标签
    train_size = len(train_data)
    x_train = all_data.iloc[:train_size]
    y_train = y[:train_size]
    x_test = all_data.iloc[train_size:]
    y_test = y[train_size:]

    # 类别型的列
    x_train_categ = x_train[CATEGORICAL_COLUMNS].values
    x_test_categ = x_test[CATEGORICAL_COLUMNS].values

    # 连续型的列
    x_train_conti = x_train[CONTINUOUS_COLUMNS].values.astype(np.float64)
    x_test_conti = x_test[CONTINUOUS_COLUMNS].values.astype(np.float64)

    # 对连续值的列做幅度缩放
    scaler = StandardScaler()
    x_train_conti = scaler.fit_transform(x_train_conti)
    x_test_conti = scaler.fit_transform(x_test_conti)

    return [x_train, y_train, x_test, y_test, x_train_categ, x_test_categ, x_train_conti, x_test_conti, all_data]

class Wide_and_Deep:
    def __init__(self, mode='wide and deep'):
        self.mode = mode
        x_train, y_train, x_test, y_test, x_train_categ, x_test_categ, x_train_conti, x_test_conti, all_data \
            = preprocessing()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_categ = x_train_categ
        self.x_test_categ = x_test_categ
        self.x_train_conti = x_train_conti
        self.x_test_conti = x_test_conti
        self.all_data = all_data
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        self.x_train_categ_poly = self.poly.fit_transform(x_train_categ)
        self.x_test_categ_poly = self.poly.transform(x_test_categ)
        self.categ_inputs = None
        self.conti_input = None
        self.deep_component_outlayer = None
        self.logistic_input = None
        self.model = None

    def deep_component(self):
        categ_inputs = []
        categ_embeds = []
        # 对类别型的列做embedding
        for i in range(len(CATEGORICAL_COLUMNS)):
            # 预计输入是1个维度的数据，即1列
            input_i = Input(shape=(1,), dtype='int32') # shape(1,) 等价于 shape(1), 表示输入1维的向量
            dim = len(np.unique(self.all_data[CATEGORICAL_COLUMNS[i]])) # 表示输入词的总数,此时的类别特征已经被LabelEncoder离散化过了
            embed_dim = int(np.ceil(dim**0.25)) # 一个词映射成几个浮点数,取2次开方，表示经过Embedding后输出词的维度
            embed_i = Embedding(input_dim=dim, output_dim=embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            categ_inputs.append(input_i)
            categ_embeds.append(flatten_i)

        # 连续值的列
        conti_input = Input(shape=(len(CONTINUOUS_COLUMNS),)) # 输入len(CONTINUOUS_COLUMNS)维的向量
        conti_dense = Dense(256, use_bias=False)(conti_input)
        # 拼接类别型的embedding特征和连续值特征
        concat_embeds = concatenate([conti_dense]+categ_embeds)
        # 激活层与BN层(批标准化)
        concat_embeds = Activation('relu')(concat_embeds)
        bn_concat = BatchNormalization()(concat_embeds)
        # 全连接+激活层+BN层
        fc1 = Dense(512, use_bias=False)(bn_concat)
        ac1 = ReLU()(fc1)
        bn1 = BatchNormalization()(ac1)
        fc2 = Dense(256, use_bias=False)(bn1)
        ac2 = ReLU()(fc2)
        bn2 = BatchNormalization()(ac2)
        fc3 = Dense(128)(bn2)
        ac3 = ReLU()(fc3)

        self.categ_inputs = categ_inputs
        self.conti_input = conti_input
        self.deep_component_outlayer = ac3

    def wide_component(self):
        # wide部分的组件
        dim = self.x_train_categ_poly.shape[1]
        self.logistic_input = Input(shape=(dim,))

    def create_model(self):
        # wide+deep
        self.deep_component()
        self.wide_component()
        if self.mode == 'wide and deep':
            out_layer = concatenate([self.deep_component_outlayer, self.logistic_input])
            inputs = [self.conti_input] + self.categ_inputs + [self.logistic_input]
        elif self.mode == 'deep':
            out_layer = self.deep_component_outlayer
            inputs = [self.conti_input] + self.categ_inputs
        else:
            print('wrong mode')
            return

        output = Dense(1, activation='sigmoid')(out_layer)
        self.model = Model(inputs=inputs, outputs=output)

    # 训练
    def train_model(self, epochs=15, optimizer='adam', batch_size=128):
        # 不同结构的训练

        # 没有model的情况
        if not self.model:
            print('You have to create model first')
            return

        # 使用wide&deep的情况
        if self.mode == 'wide and deep':
            input_data = [self.x_train_conti] + \
                         [self.x_train_categ[:, i] for i in range(self.x_train_categ.shape[1])] + \
                         [self.x_train_categ_poly]
        # 只使用deep的情况
        elif self.mode == 'deep':
            input_data = [self.x_train_conti] + \
                         [self.x_train_categ[:, i] for i in range(self.x_train_categ.shape[1])]
        else:
            print('wrong mode')
            return

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(input_data, self.y_train, epochs=epochs, batch_size=batch_size)

    # 评估
    def evaluate_model(self):
        if not self.model:
            print('You have to create model first')
            return

        if self.mode == 'wide and deep':
            input_data = [self.x_test_conti] + \
                         [self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])] + \
                         [self.x_test_categ_poly]
        elif self.mode == 'deep':
            input_data = [self.x_test_conti] + \
                         [self.x_test_categ[:, i] for i in range(self.x_test_categ.shape[1])]
        else:
            print('wrong mode')
            return

        loss, acc = self.model.evaluate(input_data, self.y_test)
        print(f'test_loss: {loss} - test_acc: {acc}')

    def save_model(self, filename='wide_and_deep.h5'):
        self.model.save(filename)

if __name__ == '__main__':
    wide_deep_net = Wide_and_Deep()
    wide_deep_net.create_model()
    wide_deep_net.train_model()
    wide_deep_net.evaluate_model()
    wide_deep_net.save_model()