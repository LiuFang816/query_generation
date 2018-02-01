# -*- coding: utf-8 -*-
from data.data_reader import *
from model.seq2seq import *
import time
from datetime import timedelta
import numpy as np
flags=tf.flags
flags.DEFINE_string('data_path','../data','data path')
flags.DEFINE_string('save_path','../data/checkpoints/seq2seq/best_validation','best val save path')
flags.DEFINE_string('save_dir','../data/checkpoints/seq2seq','save dir')
flags.DEFINE_string('tensorboard_dir','../data/tensorboard/seq2seq','tensorboard path')
flags=flags.FLAGS

class Config:
    init_scale=0.05
    batch_size=5
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
    mode='train'
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


def evaluate(sess,x_,y_):
    data_len=len(x_)
    eval_batch=batch_iter(x_,y_,config.batch_size)
    total_loss=0.0
    total_acc=0.0
    for x_batch,y_batch in eval_batch:
        batch_len=len(x_batch)
        feed_dict=feed_data(x_batch,y_batch)
        loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
        # loss,acc=sess.run([model.loss,model.acc],feed_dict=feed_dict)
        total_loss+=loss*batch_len
        total_acc+=acc*batch_len
    # return total_loss/data_len
    return total_loss/data_len,total_acc/data_len



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
                    loss_val,acc_val=evaluate(sess,val_inputs,val_outouts)
                    # if acc_val>best_eval_acc:
                    #     best_eval_acc=acc_val
                    #     last_improved=total_batch
                    #     saver.save(sess,flags.save_path)
                    #     improved_str='*'
                    # else:
                    #     improved_str=''
                    time_dif=get_time_dif(start_time)
                    print(temp)
                    print(correct)
                    print('train loss: %.3f, train acc: %.3f, val loss: %.3f, train acc: %.3f'%(loss,acc,loss_val,acc_val))
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
# def test(save_path,N):
#     Config = tf.ConfigProto()
#     Config.gpu_options.allow_growth = True
#     with tf.Session(config=Config) as sess:
#         sess.run(tf.global_variables_initializer())
#         saver=tf.train.Saver()
#         saver.restore(sess,flags.save_path)
#         test_loss,test_acc=evaluate(sess,test_inputs,test_labels)
#         msg='Test Loss:{0:>6.2}, Test Acc:{1:>7.2%}'
#         print(msg.format(test_loss,test_acc))
#         test_batch=data_reader.batch_iter(test_inputs,test_labels,config.batch_size)
#         with open(save_path,'w') as f:
#             for x_batch,y_batch in test_batch:
#                 feed_dict=feed_data(x_batch,y_batch,config.dropout_keep_prob)
#                 loss,acc,logits=sess.run([model.loss,model.acc,model.logits],feed_dict=feed_dict)
#                 for i in range(config.batch_size):
#                     target=[]
#                     prediction=[]
#                     for j in range(config.num_steps):
#                         prediction.append([])
#                         target.append(logits[i][j])
#                         tmp = list(logits[i][j])
#                         for m in range(N):
#                             index=tmp.index(max(tmp))
#                             prediction[j].append(index)
#                             tmp[index]=-100
#
#                             # target=json.dumps(target)
#                             # # print(target)
#                             # prediction=json.dumps(prediction)
#                             # # print(prediction)
#                             # f.write(target+'\n')
#                             # f.write(prediction+'\n')



if __name__ == '__main__':
    config=Config()
    print('start fetching data...')
    encoder_word_to_id=read_vocab('../data/atis/vocab.en.txt')
    decoder_word_to_id=read_vocab('../data/atis/vocab.de.txt')
    decoder_word_to_id['START']=len(decoder_word_to_id)
    train_inputs,train_outputs,val_inputs,val_outouts,test_inputs,test_outputs,left_id, right_id, EN_PAD_ID,DE_PAD_ID,encoder_numsteps,decoder_numsteps=raw_data('../data/atis',encoder_word_to_id,decoder_word_to_id)
    config.input_num_steps=encoder_numsteps
    config.output_num_steps=decoder_numsteps
    config.encoder_vocab_size=len(encoder_word_to_id)
    config.decoder_vocab_size=len(decoder_word_to_id)
    id_to_word=reverseDic(decoder_word_to_id)

    model=Seq2seqModel(config,left_id,right_id)

    if config.mode=='train':
        train(id_to_word)
    # else:
    #     test('test.txt',1)
