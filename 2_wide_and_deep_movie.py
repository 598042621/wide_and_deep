import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Activation, concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.models import Model

all_df_path = r'./all_df.pkl'

CATEGORICAL_COLUMNS = [
    'userID', 'movieType', 'movieID','area','director','characteristic'
]

DICTVECTOR_COLUMNS = []

CONTINUES_COLUMNS = [
    'rating_douban'
]

MODEL_CONFIG = {
    'light_data': False
}

# Split Dataframe text
def dictverctorizering(df_input):
    df = df_input.copy()
    list_actor = []
    for i in df.index:
        dict_actor = {}
        for item in df.loc[i,'主演'].split('|'):
            dict_actor[item] = 1
        list_actor.append(dict_actor)

    df['actor_dict'] = pd.Series(list_actor)

    return df

def preprocessing():
    if os.path.exists(all_df_path):
        all_df = pickle.load(open(all_df_path,'rb'))
    else:
        # 数据加载
        user_df = pd.read_csv(r'./dataset/dataset1/user.csv')
        movie_df = pd.read_csv(r'./dataset/dataset1/movie.csv')

        # 特征重命名
        user_df.rename(columns={'评分':'rating', '用户ID':'userID', '类型':'movieType', '电影名':'movieID'}, inplace=True)
        user_df.drop(columns=['用户名','评论时间'], inplace=True)
        df = dictverctorizering(movie_df[['主演']])
        movie_df = pd.merge(movie_df, df, on='主演')
        movie_df.rename(columns={'主演':'actor','导演':'director','地区':'area','特色':'characteristic','评分':'rating_douban','电影名':'movieID'},inplace=True)
        movie_df.drop(columns=['类型'], inplace=True)
        movie_df.drop_duplicates(subset='movieID',inplace=True)

        # 特征拼接
        all_df = pd.merge(user_df, movie_df[['actor','area','director','characteristic','rating_douban','movieID','actor_dict']], on=['movieID'])

        # 创建标签列, 评分阈值为5
        all_df['label'] = all_df['rating'].apply(lambda x: 1 if x > 5 else 0)

        # 轻量化数据
        if(MODEL_CONFIG['light_data']==True):
            all_df = all_df.iloc[:int(all_df.shape[0]*0.3)]

        # 类别型特征LabelEncoder
        enc = LabelEncoder()
        for c in CATEGORICAL_COLUMNS:
            all_df[c] = enc.fit_transform(all_df[c])

        pickle.dump(all_df, open(all_df_path, 'wb'))

    # 字典型特征
    # dic = DictVectorizer()
    # for d in DICTVECTOR_COLUMNS:
    #     dictvec_data = dic.fit_transform(all_df[d]).toarray()
    #     dictvec_df = pd.DataFrame(dictvec_data)
    # #all_df = pd.concat([all_df,dictvec_df], axis=1)
    # # 词频型特征
    # bow = CountVectorizer()  # 词袋模型
    # bow_array = bow.fit_transform(all_df['actor'].values).toarray()
    # bow_df = pd.DataFrame(bow_array).add_prefix('word_')
    # all_df = pd.concat([all_df, bow_df], axis=1)
    # DICTVECTOR_COLUMNS.extend(bow_df.columns)


    # 划分数据集
    train, test = train_test_split(all_df, test_size=0.2)
    y_train = train.pop('label').values
    x_train = train.values
    y_test = test.pop('label').values
    x_test = test.values

    # 类别型的列
    x_train_categ = train[CATEGORICAL_COLUMNS].values
    x_test_categ = test[CATEGORICAL_COLUMNS].values
    # 连续型的列
    x_train_conti = train[CONTINUES_COLUMNS].values
    x_test_conti = test[CONTINUES_COLUMNS].values
    # 词袋向量列
    x_train_vec = train[DICTVECTOR_COLUMNS].values
    x_test_vec = test[DICTVECTOR_COLUMNS].values

    # 对连续值的列做幅度缩放
    scaler = StandardScaler()
    x_train_conti = scaler.fit_transform(x_train_conti)
    x_test_conti = scaler.fit_transform(x_test_conti)

    return x_train, y_train, x_test, y_test, x_train_categ, x_test_categ, x_train_conti, x_test_conti, all_df



class Wide_and_Deep():
    def __init__(self, mode = 'wide and deep'):
        self.mode = mode
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_train_categ = x_train_categ
        self.x_test_categ = x_test_categ
        self.x_train_conti = x_train_conti
        self.x_test_conti = x_test_conti
        self.all_data = all_df
        self.poly = PolynomialFeatures(degree=2, interaction_only=True)
        self.x_train_categ_poly = self.poly.fit_transform(x_train_categ)
        self.x_test_categ_poly = self.poly.transform(x_test_categ)
        self.categ_inputs = None
        self.conti_input = None
        self.dictvec_inputs = None
        self.deep_component_outlayer = None
        self.logistic_input = None
        self.model = None
        self.all_inputs = None

    def deep_component(self):
        categ_inputs = []
        categ_embeds = []
        dictvec_inputs = []
        dictvec_embeds = []
        # 对类别型的列做embedding
        for i in range(len(CATEGORICAL_COLUMNS)):
            # 预计输入是1个维度的数据，即1列
            input_i = Input(shape=(1,), dtype='int32')  # shape(1,) 等价于 shape(1), 表示输入1维的向量
            dim = len(np.unique(self.all_data[CATEGORICAL_COLUMNS[i]]))  # 表示输入词的总数,此时的类别特征已经被LabelEncoder离散化过了
            embed_dim = int(np.ceil(dim ** 0.25))  # 一个词映射成几个浮点数,取2次开方，表示经过Embedding后输出词的维度
            embed_i = Embedding(input_dim=dim, output_dim=embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            categ_inputs.append(input_i)
            categ_embeds.append(flatten_i)
        # 对字典型的列做embedding
        for i in range(len(DICTVECTOR_COLUMNS)):
            input_i = Input(shape=(1,), dtype='int32')
            dim = 22948  # 表示输入词的总数,此时的类别特征已经被LabelEncoder离散化过了
            embed_dim = int(np.ceil(dim ** 0.25))  # 一个词映射成几个浮点数,取2次开方，表示经过Embedding后输出词的维度
            embed_i = Embedding(input_dim=dim, output_dim=embed_dim, input_length=1)(input_i)
            flatten_i = Flatten()(embed_i)
            dictvec_inputs.append(input_i)
            dictvec_embeds.append(flatten_i)

        # 连续值的列
        conti_input = Input(shape=(len(CONTINUES_COLUMNS),)) # 输入len(CONTINUOUS_COLUMNS)维的向量
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
        self.dictvec_inputs = dictvec_inputs
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
            self.all_inputs = inputs
        elif self.mode == 'deep':
            out_layer = self.deep_component_outlayer
            inputs = [self.conti_input] + self.categ_inputs
        else:
            print('wrong mode')
            return
        # 若二分类任务，一般是sigmoid函数，损失函数为binary_crossentropy
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
    x_train, y_train, x_test, y_test, x_train_categ, x_test_categ, x_train_conti, x_test_conti, all_df = preprocessing()
    wide_deep_net = Wide_and_Deep()
    wide_deep_net.create_model()
    wide_deep_net.train_model()
    wide_deep_net.evaluate_model()
    wide_deep_net.save_model()

