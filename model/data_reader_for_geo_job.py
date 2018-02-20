# -*- coding: utf-8 -*-
import os
import collections
import numpy as np

def batch_iter(x,y,batch_size):
    data_len=len(x)
    num_batch=int((data_len-1)/batch_size)+1

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        if end_id-start_id<batch_size:
            break
        yield x[start_id:end_id],y[start_id:end_id]


def read_data(file):
    with open(file,'r',encoding='utf-8',errors='ignore') as f:
        inputs, outputs=[],[]
        lines=f.readlines()
        input_len=[]
        output_len=[]
        for line in lines:
            out,inp=line.split('\t')
            out=out.split(' ')
            inp=inp.strip('\n').split(' ')
            input_len.append(len(inp))
            output_len.append(len(out))
            inputs.append(inp)
            outputs.append(out)
        return inputs,outputs,max(input_len),max(output_len)

# inputs,outputs,input_len,output_len=read_data('geo/test.txt')
# print(outputs)
# counter = collections.Counter(output_len)
# print(counter)

def read_vocab(filename):
    word_to_id={}
    count=0
    with open(filename,'r') as f:
        words=f.readlines()
        for word in words:
            word=word.split('\t')[0]
            word_to_id[word]=count
            count+=1
    word_to_id['PAD']=count
    word_to_id['UNK']=count+1
    return word_to_id

# encoder_word_to_id=read_vocab('geo/vocab.en.txt')
# print(encoder_word_to_id)

def reverseDic(curDic):
    newmaplist = {}
    for key, value in curDic.items():
        newmaplist[value] = key
    return newmaplist

def file_to_id(word_to_id,data,num_steps):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j]=word_to_id[data[i][j]] if data[i][j] in word_to_id else word_to_id['UNK']
        if num_steps:
            if len(data[i])>num_steps:
                data[i]=data[i][:num_steps]
            else:
                for _ in range(num_steps - len(data[i])):
                    data[i].append(word_to_id['PAD'])
                # print(len(data[i]))
    return data

# input_data=file_to_id(encoder_word_to_id,inputs,input_len)
# print(input_data[1])

def raw_data(data_path=None, encoder_word_to_id=None,decoder_word_to_id=None):
    train_path = os.path.join(data_path,"train.txt")
    test_path = os.path.join(data_path, "test.txt")

    train_input,train_output,encoder_numsteps,decoder_numsteps=read_data(train_path)
    test_input,test_output,_,_=read_data(test_path)

    train_input=np.asarray(file_to_id(encoder_word_to_id,train_input,encoder_numsteps))
    train_output=np.asarray(file_to_id(decoder_word_to_id,train_output,decoder_numsteps))

    test_input=np.asarray(file_to_id(encoder_word_to_id,test_input,encoder_numsteps))
    test_output=np.asarray(file_to_id(decoder_word_to_id,test_output,decoder_numsteps))

    left_id = encoder_word_to_id['(']
    right_id = encoder_word_to_id[')']
    EN_PAD_ID = encoder_word_to_id['PAD']
    DE_PAD_ID = decoder_word_to_id['PAD']
    return train_input,train_output,test_input,test_output,left_id, right_id, EN_PAD_ID,DE_PAD_ID,encoder_numsteps,decoder_numsteps

# encoder_word_to_id=read_vocab('geo/vocab.en.txt')
# decoder_word_to_id=read_vocab('geo/vocab.de.txt')
# decoder_word_to_id['START']=len(decoder_word_to_id)
# train_input,train_output,test_input,test_output,left_id, right_id, EN_PAD_ID,DE_PAD_ID,encoder_numsteps,decoder_numsteps=raw_data('geo',encoder_word_to_id,decoder_word_to_id)
#
# print(train_input)
# test_batch=batch_iter(test_input,test_output,5)
# for x,y in test_batch:
#     print(x)