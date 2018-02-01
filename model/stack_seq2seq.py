# -*- coding: utf-8 -*-
import tensorflow as tf

class Stack_Seq2seqModel:
    def __init__(self,config,left_id, right_id):
        self.config=config
        self.inputs=tf.placeholder(tf.int64,[config.batch_size,self.config.input_num_steps],name='inputs')
        self.outputs=tf.placeholder(tf.int64,[config.batch_size,self.config.output_num_steps],name='outputs')

        self.encoder_embeding=tf.get_variable('encoder_embedding',[self.config.encoder_vocab_size,self.config.embedding_size],dtype=tf.float32)
        self.embed_inputs=tf.nn.embedding_lookup(self.encoder_embeding,self.inputs)

        self.decoder_embedding=tf.get_variable('decoder_embedding',[self.config.decoder_vocab_size,self.config.embedding_size],dtype=tf.float32)
        self.embed_outputs=tf.nn.embedding_lookup(self.decoder_embedding,self.outputs)

        self.EMBED_START_ID=tf.nn.embedding_lookup(self.decoder_embedding,tf.constant(self.config.START_ID,dtype=tf.int32,shape=[self.config.batch_size],name='START_ID'))

        self.encoder_outputs,self.encoder_final_state=self.add_encoder(self.embed_inputs)
        self.decoder_logits,self.decoder_prediction,self.temp,self.correct,self.acc=self.add_decoder(self.encoder_final_state,self.encoder_outputs,self.embed_outputs)
        self.loss,self.train_op=self.train(self.outputs,self.decoder_logits)
        self.start=left_id
        self.end=right_id
    def add_encoder(self,encoder_inputs):
        with tf.variable_scope('encoder'):
            def create_cell(hidden_size):
                def get_cell(hidden_size):
                    cell=tf.contrib.rnn.LSTMCell(hidden_size,state_is_tuple=True)
                    return cell
                cell=tf.nn.rnn_cell.MultiRNNCell([get_cell(hidden_size) for _ in range(self.config.num_layers)])
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
                    cell=tf.nn.rnn_cell.MultiRNNCell([get_cell(hidden_size) for _ in range(self.config.num_layers)])
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
            decoder_cell,decoder_init_state=build_decoder_cell()

            weight=tf.get_variable('weight',[self.config.hidden_size,self.config.decoder_vocab_size],dtype=tf.float32)
            bias=tf.get_variable('bias',[self.config.decoder_vocab_size],dtype=tf.float32)

            if self.config.mode=='train':
                outputs=[]
                state=decoder_init_state
                with tf.variable_scope('RNN'):
                    for step in range(self.config.output_num_steps-1):
                        if step>0:
                            tf.get_variable_scope().reuse_variables()
                        elif step==0:
                            (cell_output,state)=decoder_cell(self.EMBED_START_ID,state)
                            outputs.append(cell_output)
                        (cell_output,state)=decoder_cell(decoder_inputs[:,step,:],state)
                        outputs.append(cell_output)
            else:
                outputs=[]
                state=decoder_init_state
                input=self.EMBED_START_ID
                with tf.variable_scope('RNN'):
                    for step in range(self.config.output_num_steps):
                        if step>0:
                            tf.get_variable_scope().reuse_variables()
                        (cell_output,state)=decoder_cell(input,state)
                        outputs.append(cell_output)
                        logits=tf.matmul(cell_output,weight)+bias
                        input=tf.nn.embedding_lookup(self.decoder_embedding,tf.argmax(logits,axis=-1))
            # print(outputs)
            output=tf.reshape(tf.concat(outputs,1),[-1,self.config.hidden_size])
            # print(output)
            decoder_logits=tf.matmul(output,weight)+bias
            # print(decoder_logits)
            decoder_logits=tf.reshape(decoder_logits,[self.config.batch_size,self.config.output_num_steps,self.config.decoder_vocab_size])
            decoder_prediction=tf.argmax(tf.nn.softmax(decoder_logits),-1)
            temp=tf.cast(tf.equal(self.outputs,decoder_prediction),tf.int32)
            correct_pre=tf.reduce_min(temp,1)
            acc=tf.reduce_mean(tf.cast(correct_pre,tf.float32))
            return decoder_logits,decoder_prediction,temp,correct_pre,acc

    def train(self,decoder_targets,decoder_logits):
        # print(decoder_targets)
        # print(decoder_logits)
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_targets,logits=decoder_logits)
        loss=tf.reduce_mean(cross_entropy)
        train_op=tf.train.AdamOptimizer(self.config.learning_rate).minimize(loss)
        return loss,train_op


    def _build_rnn_graph_lstm(self, inputs, config):
        """Build the inference graph using canonical LSTM cells."""
        def make_cell():
            cell = tf.contrib.rnn.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.dropout_keep_prob)
            return cell

        cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, tf.float32)
        state = self._initial_state

        outputs = []

        def func_push(state):
            self.state_stack.push(state)
            return state[0][0],state[0][1],state[1][0],state[1][1]


        #-----------------特殊情况需要保留名称------------------
        # def updateState(state,time_step):
        #     # state = ((state[0][0], state[0][1]), (state[1][0],state[1][1]))
        #
        #     (out, newstate) = cell(inputs[:, time_step, :], state)
        #     # print('------------------------------hhhhhh----------------------------')
        #     tf.get_variable_scope().reuse_variables()
        #     return newstate
        # nameSet=[word_to_id['Import'],word_to_id['ClassDef'],word_to_id['FunctionDef'],word_to_id['Assign'],word_to_id['AsyncFunctionDef'],word_to_id['Attribute']]
        # def f_default(state):
        #     return state,state
        #
        # def func_push(state, time_step):
        #     #add特殊情况需要保留名称
        #     state,newState = tf.cond(tf.logical_or(
        #         tf.logical_or(tf.equal(self._input_data[0][time_step-1], nameSet[0]), tf.equal(self._input_data[0][time_step-1], nameSet[1])),
        #         tf.logical_or(tf.equal(self._input_data[0][time_step-1], nameSet[2]), tf.equal(self._input_data[0][time_step-1], nameSet[3])),
        #     ),lambda: updateState(state, time_step), lambda: f_default(state))
        #
        #     self.state_stack.push(newState)
        #     return state[0][0], state[0][1], state[1][0], state[1][1]
        # #-------------------------------------------------------------

        def func_pop(time_step,state):
            (cell_output,state)=cell(inputs[:,time_step,:],state)
            state=self.state_stack.pop()
            (cell_output,state)=cell(cell_output,state)
            return state[0][0],state[0][1],state[1][0],state[1][1]
        #def func_pop(state):
        #    #------增加一层---------------#
        #    w = tf.get_variable(
        #        "state_w", [2*config.hidden_size, config.hidden_size], dtype=tf.float32)
        #    # b = tf.get_variable("state_b", [1,config.hidden_size], dtype=data_type())
        #    old_state=self.state_stack.pop()
        #    concat_state=[[],[]]
        #    new_state=[[],[]]
        #    # concat_state=tf.concat([old_state,state],1)
        #    # concat_state=tf.reshape(concat_state,[1,-1])
        #    for i in range(2):
        #        for j in range(2):
        #            concat_state[i].append(tf.concat([old_state[i][j],state[i][j]],1))
        #            new_state[i].append(tf.matmul(concat_state[i][j], w))
        #    return new_state[0][0], new_state[0][1], new_state[1][0], new_state[1][1]

        def func_default(state):
            return state[0][0],state[0][1],state[1][0],state[1][1]

        with tf.variable_scope("RNN"):
            for time_step in range(self.config.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                # print(self.input)
                new_state=tf.cond(tf.equal(self.embed_inputs[0][time_step],self.start),
                                  lambda:func_push(state),lambda:func_default(state))
                new_state=tf.cond(tf.equal(self.embed_inputs[0][time_step],self.end),
                                  lambda:func_pop(time_step,state),lambda:func_default(state))
                state=((new_state[0],new_state[1]),(new_state[2],new_state[3]))

                (outputs, state) = cell(inputs[:, time_step, :], state)
                #outputs.append(cell_output)
        # output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return outputs, state
