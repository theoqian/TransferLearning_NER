import pickle
import os
from data_utils import *
from model import TransformModel
import tensorflow as tf
from conlleval import return_report

flags = tf.app.flags
flags.DEFINE_integer("char_dim", 100, "Embedding size for characters.")
flags.DEFINE_integer("lstm_dim", 100, "Number of hidden units in LSTM")
flags.DEFINE_integer("batch_size", 20, "Batch size")
flags.DEFINE_string("word_vectors_file", os.path.join("data", "wiki_100.utf8"), "Path for word vectors")
flags.DEFINE_string("ckpt_path", "ckpt", "Path to save model")
flags.DEFINE_string("map_file", "maps.pkl", "File for char2id, id2char, tag2id, id2tag")
FLAGS = tf.app.flags.FLAGS


def joint_train():
    train_sentences = load_sentence("data\\train_data")
    test_sentences = load_sentence("data\\test_data")
    boson_train_sentences = load_sentence("data\\BosonNLP_NER_train")
    boson_test_sentences = load_sentence("data\\BosonNLP_NER_test")
    if not os.path.isfile(FLAGS.map_file):
        id2char, char2id = char_mapping(train_sentences + boson_train_sentences)
        id2tag, tag2id = tag_mapping()
        with open(FLAGS.map_file, "wb") as f:
            pickle.dump([char2id, id2char, tag2id, id2tag], f)
    else:
        with open(FLAGS.map_file, "rb") as f:
            char2id, id2char, tag2id, id2tag = pickle.load(f)
    boson_id2tag, boson_tag2id = boson_tag_mapping()
    with open("boson_maps.pkl", "wb") as f:
        pickle.dump([char2id, id2char, boson_tag2id, boson_id2tag], f)
    train_data = preprocess_data(train_sentences, char2id, tag2id)
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    test_data = preprocess_data(test_sentences, char2id, tag2id)
    test_manager = BatchManager(test_data, FLAGS.batch_size)
    boson_train_data = preprocess_data(boson_train_sentences, char2id, boson_tag2id)
    boson_train_manager = BatchManager(boson_train_data, FLAGS.batch_size)
    boson_test_data = preprocess_data(boson_test_sentences, char2id, boson_tag2id)
    boson_test_manager = BatchManager(boson_test_data, FLAGS.batch_size)
    with tf.Session() as sess:
        model = TransformModel(num_chars=len(id2char),
                               num_target=len(id2tag),
                               char_dim=FLAGS.char_dim,
                               lstm_dim=FLAGS.lstm_dim,
                               name="normal_ner")
        boson_model = TransformModel(num_chars=len(id2char),
                                     num_target=len(boson_id2tag),
                                     char_dim=FLAGS.char_dim,
                                     lstm_dim=FLAGS.lstm_dim,
                                     name="boson_ner")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state("joint_ckpt")
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with random parameters")
            sess.run(tf.global_variables_initializer())
            embeddings = sess.run(model.char_lookup().read_value())
            embeddings = load_wordvec(FLAGS.word_vectors_file, id2char, FLAGS.char_dim, embeddings)
            sess.run(model.char_lookup().assign(embeddings))
        print("========== Start training ==========")
        for i in range(40):
            loss = []
            boson_loss = []
            for batch, boson_batch in zip(train_manager.iter_batch(), boson_train_manager.iter_batch()):
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % 1000 == 0:
                    print("Step: %d Loss: %f" % (step, batch_loss))
                boson_step, boson_batch_loss = boson_model.run_step(sess, True, boson_batch)
                boson_loss.append(boson_batch_loss)
                if boson_step % 1000 == 0:
                    print("Step: %d Loss: %f" % (boson_step, boson_batch_loss))
            print("Epoch: {} Loss: {:>9.6f}".format(i, np.mean(loss)))
            results = model.evaluate(sess, test_manager, id2tag)
            for line in test_ner(results, "data\\test_result"):
                print(line)
            print("Epoch: {} Boson Loss: {:>9.6f}".format(i, np.mean(boson_loss)))
            results = boson_model.evaluate(sess, boson_test_manager, boson_id2tag)
            for line in test_ner(results, "data\\boson_test_result"):
                print(line)
            ckpt_file = os.path.join("joint_ckpt", str(i) + "ner.ckpt")
            saver.save(sess, ckpt_file)
        print("========== Finish training ==========")


def train(train_file, test_file, map_file, model_name, ckpt_path, result_file, max_epoch=30):
    train_sentences = load_sentence(train_file)
    test_sentences = load_sentence(test_file)
    if not os.path.isfile(map_file):
        id2char, char2id = char_mapping(train_sentences)
        id2tag, tag2id = tag_mapping()
        with open(map_file, "wb") as f:
            pickle.dump([char2id, id2char, tag2id, id2tag], f)
    else:
        with open(map_file, "rb") as f:
            char2id, id2char, tag2id, id2tag = pickle.load(f)
    train_data = preprocess_data(train_sentences, char2id, tag2id)
    train_manager = BatchManager(train_data, FLAGS.batch_size)
    test_data = preprocess_data(test_sentences, char2id, tag2id)
    test_manager = BatchManager(test_data, FLAGS.batch_size)
    with tf.Session() as sess:
        model = TransformModel(num_chars=len(id2char),
                               num_target=len(id2tag),
                               char_dim=FLAGS.char_dim,
                               lstm_dim=FLAGS.lstm_dim,
                               name=model_name)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with random parameters")
            sess.run(tf.global_variables_initializer())
            embeddings = sess.run(model.char_lookup().read_value())
            embeddings = load_wordvec(FLAGS.word_vectors_file, id2char, FLAGS.char_dim, embeddings)
            sess.run(model.char_lookup().assign(embeddings))
        print("========== Start training ==========")
        for i in range(max_epoch):
            loss = []
            for batch in train_manager.iter_batch():
                step, batch_loss = model.run_step(sess, True, batch)
                loss.append(batch_loss)
                if step % 1000 == 0:
                    print("Step: %d Loss: %f" % (step, batch_loss))
            print("Epoch: {} Loss: {:>9.6f}".format(i, np.mean(loss)))
            results = model.evaluate(sess, test_manager, id2tag)
            for line in test_ner(results, result_file):
                print(line)
            ckpt_file = os.path.join(FLAGS.ckpt_path, str(i) + "ner.ckpt")
            model.saver.save(sess, ckpt_file)
        print("========== Finish training ==========")


def test_ner(results, path):
    output_file = os.path.join(path)
    with open(output_file, "w", encoding='utf-8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def test(test_file, map_file, model_name, ckpt_path, result_file):
    test_sentences = load_sentence(test_file)
    with open(map_file, "rb") as f:
        char2id, id2char, tag2id, id2tag = pickle.load(f)
    test_data = preprocess_data(test_sentences, char2id, tag2id)
    test_manager = BatchManager(test_data, FLAGS.batch_size)
    with tf.Session() as sess:
        model = TransformModel(num_chars=len(id2char),
                               num_target=len(id2tag),
                               char_dim=FLAGS.char_dim,
                               lstm_dim=FLAGS.lstm_dim,
                               name=model_name)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        results = model.evaluate(sess, test_manager, id2tag)
        for line in test_ner(results, result_file):
            print(line)


def evaluate_line(map_file, model_name, ckpt_path, line):
    with open(map_file, "rb") as f:
        char2id, id2char, tag2id, id2tag = pickle.load(f)
    with tf.Session() as sess:
        model = TransformModel(num_chars=len(id2char),
                               num_target=len(id2tag),
                               char_dim=FLAGS.char_dim,
                               lstm_dim=FLAGS.lstm_dim,
                               name=model_name)
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        result = model.evaluate_line(sess,
                                     input_from_line(line, char2id),
                                     id2tag)
        for c, t in zip(line, result):
            print("%s - %s" % (c, t))


def main(_):
    train("data\\BosonNLP_NER_train",
          "data\\BosonNLP_NER_test",
          "boson_maps.pkl",
          "boson_ner",
          "boson_ckpt",
          "data\\boson_test_result")
    test("data\\BosonNLP_NER_test",
         "boson_maps.pkl",
         "boson_ner",
         "joint_ckpt",
         "data\\boson_test_result")
    joint_train()
    evaluate_line("boson_maps.pkl",
                  "boson_ner",
                  "joint_ckpt",
                  "乐鑫信息科技推出的新款ESP32受到中国科技委主席黄秋生的好评。")


if __name__ == '__main__':
    tf.app.run(main)
