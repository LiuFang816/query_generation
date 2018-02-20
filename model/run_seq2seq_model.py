# -*- coding: utf-8 -*-
from data.data_reader import *
from model.seq2seq import *
import time
import nltk
from datetime import timedelta
import numpy as np
flags=tf.flags
flags.DEFINE_string('data_path','../data','data path')
flags.DEFINE_string('save_path','../data/checkpoints/seq2seq/atis/best_validation','best val save path')
flags.DEFINE_string('save_dir','../data/checkpoints/seq2seq/atis','save dir')
flags.DEFINE_string('tensorboard_dir','../data/tensorboard/seq2seq/atis','tensorboard path')
flags=flags.FLAGS

class Config:
    init_scale=0.05
    batch_size=32
    learning_rate=1e-3
    input_num_steps=None
    output_num_steps=None
    encoder_vocab_size=None
    decoder_vocab_size=None
    hidden_size=100
    embedding_size=100
    num_epochs=50
    START_ID=435
    num_layers=2
    use_attention=True
    mode='test'
    save_per_batch=50
    print_per_batch=100

def feed_data(x_batch,y_batch):
    feed_dict={
        model.inputs:x_batch,
        model.outputs:y_batch,
    }
    return feed_dict

def get_time_dif(start_time):
    end_time=time.time()
    time_dif=end_time-start_time
    return timedelta(seconds=int(round(time_dif)))

#
# def evaluate(sess,x_,y_):
#     data_len=len(x_)
#     eval_batch=batch_iter(x_,y_,config.batch_size)
#     total_loss=0.0
#     total_acc=0.0
#     for x_batch,y_batch in eval_batch:
#         batch_len=len(x_batch)
#         feed_dict=feed_data(x_batch,y_batch)
#         loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
#         # loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
#         total_loss+=loss*batch_len
#         total_acc+=acc*batch_len
#     # return total_loss/data_len
#     return total_loss/data_len,total_acc/data_len

def evaluate(sess,x_,y_,id_to_word):
    data_len=len(x_)
    eval_batch=batch_iter(x_,y_,config.batch_size)
    total_loss=0.0
    total_acc=0.0
    predictions=[]
    targets=[]
    for x_batch,y_batch in eval_batch:
        batch_len=len(x_batch)
        feed_dict=feed_data(x_batch,y_batch)
        # print(x_batch)
        loss,acc,prediction=sess.run([model.loss,model.acc,model.decoder_prediction],feed_dict=feed_dict)
        # loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
        total_loss+=loss*batch_len
        total_acc+=acc*batch_len
        predictions.extend(prediction.tolist())
        targets.extend(y_batch.tolist())
    # return total_loss/data_len,total_acc/data_len
    return total_loss/data_len,total_acc/data_len,predictions,targets

def train(id_to_word):
    tensorboard_dir=flags.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)
    tf.summary.scalar('loss',model.loss)
    # tf.summary.scalar('accuracy',model.acc)
    merged_summary=tf.summary.merge_all()
    writer=tf.summary.FileWriter(tensorboard_dir)

    saver=tf.train.Saver()
    if not os.path.exists(flags.save_dir):
        os.makedirs(flags.save_dir)
    Config = tf.ConfigProto()
    Config.gpu_options.allow_growth = True
    with tf.Session(config=Config) as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        print('training and evaluating...')
        start_time=time.time()
        total_batch=0
        best_eval_acc=0.0
        last_improved=0
        required_improvement=1000 #超过1000轮未提升则提前结束训练
        flag=False
        for epoch in range(config.num_epochs):
            print('Epoch:',epoch+1)
            train_batch=batch_iter(train_inputs,train_outputs,config.batch_size)
            for x_batch,y_batch in train_batch:
                feed_dict=feed_data(x_batch,y_batch)
                if total_batch%config.save_per_batch==0:
                    s=sess.run(merged_summary,feed_dict=feed_dict)
                    writer.add_summary(s,total_batch)
                if total_batch%config.print_per_batch==0:
                    loss,prediction,acc,temp,correct=sess.run([model.loss,model.decoder_prediction,model.acc,model.temp,model.correct],feed_dict=feed_dict)
                    # loss,acc,prediction=sess.run([model.loss,model.acc,model.predict_class],feed_dict=feed_dict)
                    loss_val,acc_val,_,_=evaluate(sess,val_inputs,val_outputs,id_to_word)
                    saver.save(sess,flags.save_path)
                    improved_str=''
                    if acc_val>best_eval_acc:
                        best_eval_acc=acc_val
                        last_improved=total_batch
                        saver.save(sess,flags.save_path)
                        improved_str='*'
                    else:
                        improved_str=''
                    time_dif=get_time_dif(start_time)
                    # print(temp)
                    # print(correct)
                    print('train loss: %.3f, train acc: %.3f, val loss: %.3f, val acc: %.3f, %s'%(loss,acc,loss_val,acc_val,improved_str))
                    targets=[]
                    for i in y_batch[0]:
                        targets.append(id_to_word[i])
                    predict=[]
                    for i in prediction[0]:
                        predict.append(id_to_word[i])
                    print(targets)
                    print(predict)
                    print('---------------------------------------------------------\n\n')

                    # for i in range(2):#print 2 个example （≤batch_size）
                    #     input_=''
                    #     target_=''
                    #     predic_=''
                    #     for j in range(config.num_steps-1):
                    #         input_+=id_to_word[x_batch[i][j]]+' '
                    #         target_+=id_to_word[y_batch[i][j]]+' '
                    #         predic_+=id_to_word[prediction[i][j]]+' '
                    #     print('input:   %s\n\nTarget:   %s\n\nPrediction: %s \n\n'%(input_,target_,predic_))

                sess.run(model.train_op,feed_dict=feed_dict)
                total_batch+=1



import json
def test(id_to_word):
    Config = tf.ConfigProto()
    Config.gpu_options.allow_growth = True
    with tf.Session(config=Config) as sess:
        sess.run(tf.global_variables_initializer())
        saver=tf.train.Saver()
        saver.restore(sess,flags.save_path)
        test_loss,test_acc,predictions,targets=evaluate(sess,test_inputs,test_outputs,id_to_word)
        msg='Test Loss:{0:>6.2}, Test Acc:{1:>7.2%}'
        print(msg.format(test_loss,test_acc))
        # print(predictions[0])
        # print(targets[0])
        with open('pre_seq2seq_atis.txt','w') as f:
            # print(predictions)
            f.write(json.dumps(predictions))
        with open('tar_seq2seq_atis.txt','w') as f:
            f.write(json.dumps(targets))



# if __name__ == '__main__':
#     config=Config()
#     print('start fetching data...')
#     encoder_word_to_id=read_vocab('../data/atis/vocab.en.txt')
#     decoder_word_to_id=read_vocab('../data/atis/vocab.de.txt')
#     decoder_word_to_id['START']=len(decoder_word_to_id)
#     train_inputs,train_outputs,val_inputs,val_outputs,test_inputs,test_outputs,left_id, right_id, EN_PAD_ID,DE_PAD_ID,encoder_numsteps,decoder_numsteps=raw_data('../data/atis',encoder_word_to_id,decoder_word_to_id)
#     config.input_num_steps=encoder_numsteps
#     config.output_num_steps=decoder_numsteps
#     config.encoder_vocab_size=len(encoder_word_to_id)
#     config.decoder_vocab_size=len(decoder_word_to_id)
#     id_to_word=reverseDic(decoder_word_to_id)
#
#     model=Seq2seqModel(config)
#
#     if config.mode=='train':
#         train(id_to_word)
#     else:
#         test(id_to_word)

def get_blue(target_file,pre_file):
    with open(target_file,'r') as f:
        target=json.load(f)
    with open(pre_file,'r') as f:
        prediction=json.load(f)
    clean_target=[]
    clean_prediction=[]
    for i in range(len(prediction)):
        if 433 in prediction[i]:
            index=prediction[i].index(433)
            prediction[i]=prediction[i][:index]
    for i in range(len(target)):
        if 433 in target[i]:
            index=target[i].index(433)
            target[i]=target[i][:index]
    for i in range(len(prediction)):
        if (len(prediction[i])>=4) & (len(target[i])>=4):
            clean_prediction.append(prediction[i])
            clean_target.append(target[i])
    BLEU=[]
    for i in range(len(clean_prediction)):
        BLEU.append(nltk.translate.bleu_score.sentence_bleu([clean_target[i]], clean_prediction[i]))
    print(sum(BLEU)/len(BLEU))
get_blue('tar_seq2seq_atis.txt','pre_seq2seq_atis.txt')