from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, Permute
from keras.layers.embeddings import Embedding
from keras.layers.merge import add, concatenate, dot
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import collections
import nltk
import numpy as np
import os

DATA_DIR = "../data"
TRAIN_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_train.txt")
TEST_FILE = os.path.join(DATA_DIR, "qa1_single-supporting-fact_test.txt")

EMBEDDING_SIZE = 64
LATENT_SIZE = 32
BATCH_SIZE = 32
NUM_EPOCHS = 50
PAD_VALUE = 'pad'


class DataProcessor:

    def get_data(self, train_file_path):
        stories, questions, answers = [], [], []
        story_text = ''
        with open(train_file_path, "rb") as train_file:
            for line in train_file:
                line = line.decode("utf-8").strip()
                line_number, text = line.split(" ", 1)
                if "\t" in text:
                    question, answer, _ = text.split("\t")
                    stories.append(story_text)
                    questions.append(question)
                    answers.append(answer)
                    story_text = ''
                else:
                    story_text = story_text + text + ' '
        return stories, questions, answers

    def build_vocab(self, train_data, test_data):
        data = [PAD_VALUE]
        counter = collections.Counter()
        for stories, questions, answers in [train_data, test_data]:
            data.extend(stories)
            data.extend(questions)
            data.extend(answers)
        data = np.hstack(data)

        for sentence in data:
            for word in nltk.word_tokenize(sentence):
                counter[word.lower()] += 1

        word2idx = {w: i for i, (w, _) in enumerate(counter.most_common())}
        idx2word = {v: k for k, v in word2idx.items()}
        return word2idx, idx2word

    def get_maxlens(self, train_data, test_data):
        story_maxlen, question_maxlen = 0, 0
        for stories, questions, _ in [train_data, test_data]:
            for story in stories:
                story_words = nltk.word_tokenize(story)
                story_len = len(story_words)
                if story_len > story_maxlen:
                    story_maxlen = story_len
            for question in questions:
                question_len = len(nltk.word_tokenize(question))
                if question_len > question_maxlen:
                    question_maxlen = question_len
        return story_maxlen, question_maxlen

    def vectorize(self, data, word2idx, story_maxlen, question_maxlen):
        X_stories, X_questions, Y = [], [], []
        stories, questions, answers = data
        for story, question, answer in zip(stories, questions, answers):
            vectorized_story = [word2idx[w.lower()] for w in
                                nltk.word_tokenize(story)]
            vectorized_question = [word2idx[w.lower()] for w in
                                   nltk.word_tokenize(question)]
            X_stories.append(vectorized_story)
            X_questions.append(vectorized_question)
            Y.append(word2idx[answer.lower()])
        return pad_sequences(X_stories, maxlen=story_maxlen, value=word2idx[PAD_VALUE]),\
               pad_sequences(X_questions, maxlen=question_maxlen, value=word2idx[PAD_VALUE]),\
               np_utils.to_categorical(Y, num_classes=len(word2idx))


class MemoryNetwork:

    def __init__(self, model=None):
        self.model = model
        self.history = None

    def init_network(self, story_maxlen, question_maxlen, vocab_size):

        # inputs
        story_input = Input(shape=(story_maxlen,))
        question_input = Input(shape=(question_maxlen,))

        # story encoder memory
        story_encoder = Embedding(input_dim=vocab_size,
                                  output_dim=EMBEDDING_SIZE,
                                  input_length=story_maxlen)(story_input)
        story_encoder = Dropout(0.3)(story_encoder)

        # question encoder
        question_encoder = Embedding(input_dim=vocab_size,
                                     output_dim=EMBEDDING_SIZE,
                                     input_length=question_maxlen)(question_input)
        question_encoder = Dropout(0.3)(question_encoder)

        # match between story and question
        match = dot([story_encoder, question_encoder], axes=[2, 2])

        # encode story into vector space of question
        story_encoder_c = Embedding(input_dim=vocab_size,
                                    output_dim=question_maxlen,
                                    input_length=story_maxlen)(story_input)
        story_encoder_c = Dropout(0.3)(story_encoder_c)

        # combine match and story vectors
        response = add([match, story_encoder_c])
        response = Permute((2, 1))(response)

        # combine response and question vectors
        answer = concatenate([response, question_encoder], axis=-1)
        answer = LSTM(LATENT_SIZE)(answer)
        answer = Dropout(0.3)(answer)
        answer = Dense(vocab_size)(answer)
        output = Activation("softmax")(answer)

        model = Model(inputs=[story_input, question_input], outputs=output)
        model.compile(optimizer="rmsprop",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        self.model = model

    def train(self, train_data, validation_data):
        history = self.model.fit([train_data[0], train_data[1]], [train_data[2]],
                                 batch_size=BATCH_SIZE,
                                 epochs=NUM_EPOCHS,
                                 validation_data=([validation_data[0],
                                                   validation_data[1]],
                                                  [validation_data[2]]))
        self.history = history

    def predict(self, story_text, question_text):
        return self.model.predict([story_text, question_text])


data_processor = DataProcessor()
data_train = data_processor.get_data(TRAIN_FILE)
data_test = data_processor.get_data(TEST_FILE)
word2idx, idx2word = data_processor.build_vocab(data_train, data_test)
vocab_size = len(word2idx)
story_maxlen, question_maxlen = data_processor.get_maxlens(data_train, data_test)
Xstrain, Xqtrain, Ytrain = data_processor.vectorize(data_train, word2idx, story_maxlen,
                                                    question_maxlen)
Xstest, Xqtest, Ytest = data_processor.vectorize(data_test, word2idx, story_maxlen,
                                                 question_maxlen)


network = MemoryNetwork()
network.init_network(story_maxlen, question_maxlen, vocab_size)
network.train((Xstrain, Xqtrain, Ytrain), (Xstest, Xqtest, Ytest))
ytest = np.argmax(Ytest, axis=1)
Ytest_ = network.predict(Xstest, Xqtest)
ytest_ = np.argmax(Ytest_, axis=1)

NUM_DISPLAY = 10
for i in range(NUM_DISPLAY):
    story = " ".join([idx2word[x] for x in Xstest[i].tolist() if x != word2idx[PAD_VALUE]])
    question = " ".join([idx2word[x] for x in Xqtest[i].tolist()])
    label = idx2word[ytest[i]]
    prediction = idx2word[ytest_[i]]
    print(story, question, label, prediction)
