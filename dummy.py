import os
import shutil
import random as rnd
import json
import trax
import trax.fastmath.numpy as np
from trax import layers as tl
from trax import shapes
from sklearn.model_selection import train_test_split
from utils import load_tweets, process_tweet, Layer
from trax.supervised import training
import json

# import numpy as np
import random as rnd
import subprocess


class TweetSentimentAnalysis:
    def __init__(self):
        self.Vocab = None

    def load_tweets(self):
        all_positive_tweets, all_negative_tweets = load_tweets()
        print(f"The number of positive tweets: {len(all_positive_tweets)}")
        print(f"The number of negative tweets: {len(all_negative_tweets)}")

        val_pos_tweeter = all_positive_tweets[4500:]
        train_pos_tweeter = all_positive_tweets[:4500]
        val_neg_tweeter = all_negative_tweets[4500:]
        train_neg_tweeter = all_negative_tweets[:4500]

        X_train_tweeter = train_pos_tweeter + train_neg_tweeter
        X_val_tweeter = val_pos_tweeter + val_neg_tweeter
        y_train_tweeter = np.append(
            np.ones(len(train_pos_tweeter)), np.zeros(len(train_neg_tweeter))
        )
        y_val_tweeter = np.append(
            np.ones(len(val_pos_tweeter)), np.zeros(len(val_neg_tweeter))
        )

        self.train_pos = train_pos_tweeter
        self.train_neg = train_neg_tweeter
        self.train_x = X_train_tweeter
        self.train_y = np.concatenate([y_train_tweeter])
        self.val_pos = val_pos_tweeter
        self.val_neg = val_neg_tweeter
        self.val_x = X_val_tweeter
        self.val_y = np.concatenate([y_val_tweeter])

    def save_vocab(self, filename="Vocabulary.json"):
        with open(filename, "w") as json_file:
            json.dump(self.Vocab, json_file)

    def load_vocab(self, filename="Vocabulary.json"):
        with open(filename, "r") as json_file:
            self.Vocab = json.load(json_file)

    def process_tweet(self, tweet):
        return process_tweet(tweet)

    def get_vocab(self, train_x, min_occurrence=2):
        Vocab = {"__PAD__": 0, "__</e>__": 1, "__UNK__": 2}
        word_counts = {}
        for tweet in train_x:
            processed_tweet = self.process_tweet(tweet)
            for word in processed_tweet:
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count > min_occurrence and word not in Vocab:
                Vocab[word] = len(Vocab)

        self.Vocab = Vocab

    def tweet_to_tensor(self, tweet, unk_token="__UNK__", verbose=False):
        word_l = self.process_tweet(tweet)
        if verbose:
            print("List of words from the processed tweet:")
            print(word_l)
        unk_ID = self.Vocab[unk_token]
        if verbose:
            print(f"The unique integer ID for the unk_token is {unk_ID}")
        tensor_l = [
            self.Vocab[word] if word in self.Vocab else unk_ID for word in word_l
        ]
        return tensor_l

    def data_generator(self, data_pos, data_neg, batch_size, loop, shuffle=False):
        assert batch_size % 2 == 0
        n_to_take = batch_size // 2
        pos_index = 0
        neg_index = 0
        len_data_pos = len(data_pos)
        len_data_neg = len(data_neg)
        pos_index_lines = list(range(len_data_pos))
        neg_index_lines = list(range(len_data_neg))
        if shuffle:
            rnd.shuffle(pos_index_lines)
            rnd.shuffle(neg_index_lines)
        stop = False

        while not stop:
            batch = []

            for i in range(n_to_take):
                if pos_index >= len_data_pos:
                    if not loop:
                        stop = True
                        break
                    pos_index = 0
                    if shuffle:
                        rnd.shuffle(pos_index_lines)

                tweet = data_pos[pos_index_lines[pos_index]]
                tensor = self.tweet_to_tensor(tweet)
                batch.append(tensor)
                pos_index = pos_index + 1

            for i in range(n_to_take):
                if neg_index >= len_data_neg:
                    if not loop:
                        stop = True
                        break
                    neg_index = 0
                    if shuffle:
                        rnd.shuffle(neg_index_lines)

                tweet = data_neg[neg_index_lines[neg_index]]
                tensor = self.tweet_to_tensor(tweet)
                batch.append(tensor)
                neg_index = neg_index + 1

            if stop:
                break

            max_len = max([len(t) for t in batch])
            tensor_pad_l = []

            for tensor in batch:
                n_pad = max_len - len(tensor)
                pad_l = n_pad * [0]
                tensor_pad = tensor + pad_l
                tensor_pad_l.append(tensor_pad)

            inputs = np.array(tensor_pad_l)
            target_pos = n_to_take * [1]
            target_neg = n_to_take * [0]
            target_l = target_pos + target_neg
            targets = np.array(target_l)
            example_weights = np.ones_like(targets).astype(int)

            yield inputs, targets, example_weights

    def train_generator(self, batch_size, loop=True, shuffle=False):
        return self.data_generator(
            self.train_pos, self.train_neg, batch_size, loop, shuffle
        )

    def val_generator(self, batch_size, loop=True, shuffle=False):
        return self.data_generator(
            self.val_pos, self.val_neg, batch_size, loop, shuffle
        )

    def test_generator(self, batch_size, loop=False, shuffle=False):
        return self.data_generator(
            self.val_pos, self.val_neg, batch_size, loop, shuffle
        )


tweet_sentiment_analysis = TweetSentimentAnalysis()
tweet_sentiment_analysis.load_tweets()
min_occurrence = 0
tweet_sentiment_analysis.get_vocab(tweet_sentiment_analysis.train_x, min_occurrence)
tweet_sentiment_analysis.save_vocab()
inputs, targets, example_weights = next(
    tweet_sentiment_analysis.train_generator(4, loop=True, shuffle=True)
)


class SentimentAnalysisModel:
    def __init__(self, vocab_size=118675, d_model=256, n_layers=2, mode="train"):
        self.model = self.build_model(vocab_size, d_model, n_layers, mode)

    def tweet_to_tensor(tweet, vocab_dict, unk_token="__UNK__", verbose=False):
        word_l = process_tweet(tweet)
        if verbose:
            print("List of words from the processed tweet:")
            print(word_l)
        unk_ID = vocab_dict[unk_token]
        if verbose:
            print(f"The unique integer ID for the unk_token is {unk_ID}")
        tensor_l = [
            vocab_dict[word] if word in vocab_dict else unk_ID for word in word_l
        ]
        return tensor_l

    def build_model(self, vocab_size, d_model, n_layers, mode):
        model = tl.Serial(
            tl.Embedding(vocab_size, d_model),
            [tl.GRU(n_units=d_model, mode=mode) for _ in range(n_layers)],
            tl.Select([0]),  # Select the output of the last GRU layer
            tl.Mean(axis=1),
            tl.Dense(n_units=2),
            tl.LogSoftmax(),
        )
        return model

    def load_model(self, model_file="./new_model/model.pkl.gz"):
        self.model.init_from_file(
            file_name=model_file,
            weights_only=True,
            input_signature=shapes.signature(
                (
                    shapes.ShapeDtype((1, 1), np.int32),
                    shapes.ShapeDtype((1,), np.int32),
                    shapes.ShapeDtype((1,), np.int32),
                )
            ),
        )

    def train(
        self,
        train_generator,
        eval_generator,
        output_dir="./new_model/",
        n_steps=10,
        random_seed=31,
        batch_size=32,
    ):
        directory_to_remove = output_dir
        try:
            subprocess.run(["rm", "-rf", directory_to_remove], check=True)
            print(f"Successfully removed {directory_to_remove}")
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")

        train_task = training.TrainTask(
            labeled_data=train_generator,
            loss_layer=tl.CrossEntropyLoss(),
            optimizer=trax.optimizers.Adam(0.001),
            n_steps_per_checkpoint=1,
        )

        eval_task = training.EvalTask(
            labeled_data=eval_generator,
            metrics=[tl.CrossEntropyLoss(), tl.Accuracy()],
        )

        training_loop = training.Loop(
            self.model,
            train_task,
            eval_tasks=[eval_task],
            output_dir=output_dir,
            random_seed=31,
        )
        training_loop.run(n_steps=n_steps)

    def predict(self, sentence, vocab_dict):
        inputs = np.array(self.tweet_to_tensor(sentence, vocab_dict=vocab_dict))
        inputs = inputs[None, :]
        preds_probs = self.model(inputs)
        preds = int(preds_probs[0, 1] > preds_probs[0, 0])
        sentiment = "negative" if preds == 0 else "positive"
        return preds, sentiment

    def compute_accuracy(self, preds, y, y_weights):
        is_pos = preds[:, 1] > preds[:, 0]
        is_pos_int = is_pos.astype(np.int32)
        correct = y == is_pos_int
        sum_weights = np.sum(y_weights)
        correct_float = correct.astype(np.int32)
        weighted_correct_float = np.multiply(correct_float, y_weights)
        weighted_num_correct = np.sum(weighted_correct_float)
        accuracy = weighted_num_correct / sum_weights
        return accuracy, weighted_num_correct, sum_weights

    def test_model(self, generator):
        accuracy = 0.0
        total_num_correct = 0
        total_num_pred = 0

        for batch in generator:
            inputs, targets, example_weight = batch
            pred = self.model(inputs)
            batch_accuracy, batch_num_correct, batch_num_pred = self.compute_accuracy(
                preds=pred, y=targets, y_weights=example_weight
            )
            total_num_correct += batch_num_correct
            total_num_pred += batch_num_pred

        accuracy = total_num_correct / total_num_pred
        return accuracy


def model_train(sa_model):
    try:
        shutil.rmtree(output_dir)
    except OSError as e:
        pass
    sa_model.train(
        tweet_sentiment_analysis.train_generator(batch_size, loop=True, shuffle=True),
        tweet_sentiment_analysis.val_generator(batch_size, loop=True, shuffle=True),
        output_dir="./new_model/",
        n_steps=20,
        random_seed=31,
    )
    return sa_model


sa_model = SentimentAnalysisModel(vocab_size=len(tweet_sentiment_analysis.Vocab))
batch_size = 32
output_dir = "./new_model/"

# sa_model = model_train(sa_model)
sa_model.load_model()

accuracy = sa_model.test_model(
    tweet_sentiment_analysis.test_generator(batch_size, loop=False, shuffle=False)
)
print(f"The accuracy of your model on the validation set is {accuracy:.4f}")
