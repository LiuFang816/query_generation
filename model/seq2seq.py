# -*- coding: utf-8 -*-
import tensorflow as tf

class Seq2seqModel:
    def __init__(self,config):
        self.config=config
        self.inputs=tf.placeholder(tf.int32,[None,self.config.input_num_steps],name='inputs')
        self.outputs=tf.placeholder(tf.int32,[None,self.config.output_num_steps],name='outputs')

        self.encoder_embeding=tf.get_variable('encoder_embedding',[self.config.encoder_vocab_size,self.config.embedding_size],dtype=tf.float32)
        self.embed_inputs=tf.nn.embedding_lookup(self.encoder_embeding,self.inputs)

        self.decoder_embedding=tf.get_variable('decoder_embedding',[self.config.decoder_vocab_size,self.config.embedding_size],dtype=tf.float32)
        self.embed_outputs=tf.nn.embedding_lookup(self.decoder_embedding,self.outputs)

        self.EMBED_START_ID=tf.nn.embedding_lookup(self.decoder_embedding,tf.constant(self.config.START_ID,dtype=tf.int32,shape=[self.config.batch_size],name='START_ID'))


    def add_encoder(self,encoder_inputs):
        with tf.variable_scope('encoder'):
            def create_cell(hidden_size):
                def get_cell(hidden_size):
                    cell=tf.contrib.rnn.LSTMCell(hidden_size,state_is_tuple=True)
                    return cell
                cell=tf.nn.rnn_cell.MultiRNNCell([get_cell(hidden_size) for _ in range(self.config.num_layer)])
                return cell
        encoder_cell=create_cell(self.config.hidden_size)
        encoder_outputs,encoder_final_state=tf.nn.dynamic_rnn(encoder_cell,encoder_inputs,
                                                              sequence_length=tf.constant(self.config.input_num_steps,shape=[self.config.batch_size],dtype=tf.int32),dtype=tf.float32)
        return encoder_outputs,encoder_final_state

    def add_decoder(self,encoder_final_state,encoder_outputs,decoder_inputs):
        with tf.variable_scope('decoder'):
            def build_decoder_cell():
                def create_cell(hidden_size):
                    def get_cell(hidden_size):
                        cell=tf.contrib.rnn.LSTMCell(hidden_size,state_is_tuple=True)
                        return cell
                    cell=tf.nn.rnn_cell.MultiRNNCell([get_cell(hidden_size) for _ in range(self.config.num_layer)])
                    return cell
                decoder_cell=create_cell(self.config.hidden_size)
                decoder_init_state=encoder_final_state
                if self.config.use_attention:
                    attention_mechanism=tf.contrib.seq2seq.LuongAttention(
                        self.config.hidden_size,encoder_outputs,memory_sequence_length=tf.constant(self.config.input_num_steps,shape=[self.config.batch_size],dtype=tf.int32)
                    )
                    decoder_cell=tf.contrib.seq2seq.AttentionWrapper(decoder_cell,attention_mechanism,attention_layer_size=self.config.hidden_size,name='attention')
                    attention_state=decoder_cell.zero_state(self.config.batch_size,tf.float32).clone(cell_state=encoder_final_state)
                    decoder_init_state=attention_state
                return decoder_cell,decoder_init_state

