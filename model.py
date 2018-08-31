import tensorflow as tf
import numpy as np
from compute_acc import acc_compute
from creat_data import creat_cnf
dim = 128
lstmnum = 4
lstmround = 8

CHECKFILE = './checkpoint/model.ckpt'

def memory(stat, input1, last, scope, is_train):
    with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
        ft = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.sigmoid,
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               scope='3lun' + 'memoryft1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm,
            scope='3lun' + 'memoryft2' + scope)
        it = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.sigmoid,
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               scope='3lun' + 'memoryit1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm,
            scope='3lun' + 'memoryit2' + scope)
        ct = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.tanh,
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               scope='3lun' + 'memoryct1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.tanh, normalizer_fn=tf.contrib.layers.batch_norm,
            scope='3lun' + 'memoryct2' + scope)
        ctnew = tf.multiply(ft, last) + tf.multiply(it, ct)
        ot = tf.contrib.layers.fully_connected(stat, dim, activation_fn=tf.nn.sigmoid,
                                               normalizer_fn=tf.contrib.layers.batch_norm,
                                               scope='3lun' + 'memoryot1' + scope) + tf.contrib.layers.fully_connected(
            input1, dim, activation_fn=tf.nn.sigmoid, normalizer_fn=tf.contrib.layers.batch_norm,
            scope='3lun' + 'memoryot2' + scope)
        htnew = tf.multiply(ot, tf.nn.tanh(ctnew))

        with tf.name_scope('lstm_mlp') as scope_:
            fc11 = dropout(tf.contrib.layers.fully_connected(htnew, dim * 2, activation_fn=tf.nn.elu,
                                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                                             scope=scope_ + scope + '1'), is_train)

            fc12 = dropout(tf.contrib.layers.fully_connected(ctnew, dim * 2, activation_fn=tf.nn.elu,
                                                             normalizer_fn=tf.contrib.layers.batch_norm,
                                                             scope=scope_ + scope + '2'), is_train)

            fc13 = dropout(
                tf.contrib.layers.fully_connected(tf.concat([fc11, htnew], 1), dim * 4, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '3'), is_train)

            fc14 = dropout(
                tf.contrib.layers.fully_connected(tf.concat([fc12, ctnew], 1), dim * 4, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '4'), is_train)

            fc15 = dropout(
                tf.contrib.layers.fully_connected(tf.concat([fc13, fc11], 1), dim * 4, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '5'), is_train)

            fc16 = dropout(
                tf.contrib.layers.fully_connected(tf.concat([fc14, fc12], 1), dim * 4, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '6'), is_train)

            fc17 = dropout(
                tf.contrib.layers.fully_connected(tf.concat([fc15, fc13], 1), dim * 4, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '7'), is_train)

            fc18 = dropout(
                tf.contrib.layers.fully_connected(tf.concat([fc16, fc14], 1), dim * 4, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '8'), is_train)

            htnew = dropout(
                tf.contrib.layers.fully_connected(tf.concat([htnew, fc17, fc15], 1), dim, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '9'),
                is_train)

            ctnew = dropout(
                tf.contrib.layers.fully_connected(tf.concat([ctnew, fc18, fc16], 1), dim, activation_fn=tf.nn.elu,
                                                  normalizer_fn=tf.contrib.layers.batch_norm,
                                                  scope=scope_ + scope + '10'),
                is_train)

        return htnew, ctnew


def dropout(tensor, is_train=0):
    if is_train == 1:
        keep_prob = 1.0
    else:
        keep_prob = 1.0
    return tf.nn.dropout(tensor, keep_prob)


def usememory(stat, input1, last, scope, is_train):
    ht, ct = memory(stat, input1, last, scope, is_train)
    return ht, ct


def lstm(stat, input1, last, scope, is_train):
    for i in range(lstmnum):
        stat[i], last = usememory(stat[i], input1, last, scope + str(i), is_train)
    return stat, last


def model(inputx, k_sat, is_train):  # normalizer_fn=tf.contrib.layers.batch_norm,
    with tf.name_scope('SAT') as scope:
        ht = list(np.zeros(lstmnum))
        fc1 = dropout(tf.contrib.layers.fully_connected(inputx, dim , activation_fn=tf.nn.elu,
                                                        normalizer_fn=tf.contrib.layers.batch_norm,
                                                        scope=scope + '1'),
                      is_train)

        input1 = inputx
        for i in range(lstmnum):
            ht[i] = fc1
        last = fc1
        ht, ct = lstm(ht, input1, last, 'lstm', is_train)
        for i in range(lstmround - 1):
            ht, ct = lstm(ht, input1, ct, 'lstm', is_train)

        fc35 = dropout(
            tf.contrib.layers.fully_connected(tf.concat([inputx, ht[lstmnum - 1]], 1), dim, activation_fn=tf.nn.elu,
                                              normalizer_fn=tf.contrib.layers.batch_norm,
                                              scope=scope + '35'),
            is_train)

        fc36 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx, fc35], 1), dim, activation_fn=tf.nn.elu,
                                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '36'),
                       is_train)

        fc37 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx, fc36], 1), dim, activation_fn=tf.nn.elu,
                                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '37'),
                       is_train)

        fc38 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx, fc37], 1), dim, activation_fn=tf.nn.elu,
                                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '38'),
                       is_train)

        fc39 = dropout(tf.contrib.layers.fully_connected(tf.concat([inputx, fc38], 1), dim, activation_fn=tf.nn.elu,
                                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                                         scope=scope + '39'),
                       is_train)

        fc40 = tf.contrib.layers.fully_connected(tf.concat([inputx, ht[lstmnum - 1], fc39], 1), 2*k_sat,
                                                 activation_fn=tf.nn.elu,
                                                 normalizer_fn=tf.contrib.layers.batch_norm,
                                                 scope=scope + '40')

        return tf.reshape(tf.nn.softmax(fc40),(k_sat,2)),tf.reshape((fc40),(k_sat,2))


def make_action(size_of_X,size_of_C,k_sat,try_step,n_epoch,is_Training=1):
    inputx = tf.placeholder(tf.float32, shape=[None,2 * k_sat], name="cnf")
    pre_state=tf.placeholder(tf.int32,shape=[None,k_sat],name='pre_state')
    one_hot = tf.one_hot(pre_state, 2)

    rewards=tf.placeholder(tf.float32,shape=[1],name='rewards')
    action,log = model(inputx, k_sat, is_Training)
    var_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=log,labels=one_hot))/10+1e-4

    train_step = tf.train.RMSPropOptimizer(1e-5).minimize(rewards-var_loss)
    global lr, global_step
    with tf.Session() as sess:
        saver = tf.train.Saver(max_to_keep=2)
        ckpt = tf.train.get_checkpoint_state('./checkpoint/')
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore the model from checkpoint %s' % ckpt.model_checkpoint_path)
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            start_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        else:
            sess.run(tf.global_variables_initializer())
            start_step = 0
            print('start training from new state')

        for n in range(n_epoch):
            print('Now finding the ans of SAT No.', n+1)
            X, C_np = creat_cnf(size_of_X, size_of_C, k_sat)
            reward_list = []
            for j in range(try_step):

                choose_set=np.arange(0,size_of_C)
                np.random.shuffle(choose_set)
                choose_list=list(choose_set)
                #print(choose_list)
                for i in choose_list:
                    count=0
                    ci_np = C_np[i]
                    input_raw = np.copy(ci_np)

                    acc_old = compute_reward(size_of_C, k_sat, C_np, X)
                    for k in range(k_sat):
                        input_raw[k] = X[int(ci_np[k] - 1)]
                    pre=np.copy(input_raw[:k_sat]).astype(np.int32)
                    pre.shape=(1,k_sat)
                    input_raw.shape=(1,2*k_sat)
                    act=sess.run(action,feed_dict={inputx:input_raw,pre_state:pre})

                    for k in range(k_sat):
                        if act[k][0] > act[k][1]:
                            X[int(ci_np[k] - 1)] = 0.0
                        elif act[k][0] < act[k][1]:
                            X[int(ci_np[k] - 1)] = 1.0

                    acc_new = compute_reward(size_of_C, k_sat, C_np, X)
                    if acc_new == 1.0:
                        print('Success! find an ans')
                        print(X)
                        saver.save(sess, CHECKFILE, global_step=n)
                        break
                    else:
                        reward = (acc_new - acc_old) * 10 + 1e-2
                    if j==0 and count<=(size_of_C//5):
                        reward_list.append(reward)

                #print(reward)
                        reward=np.array([1e-3])
                        reward.shape=(1)
                        sess.run(train_step,feed_dict={inputx:input_raw,rewards:reward,pre_state:pre})
                        count+=1
                        print('No. ',n+1,'SAT ','in try step ',j+1,'  after cnf ',i+1,'   ','acc from',acc_old,'  to  ',acc_new)
                    else:
                        reward_list.append(reward)
                        reward=compute_add_reward(reward_list,size_of_C,j)
                        sess.run(train_step, feed_dict={inputx: input_raw, rewards: reward, pre_state: pre})
                        count+=1
                        print('No. ',n+1,'SAT ','in try step ', j + 1, '  after cnf ', i + 1, '   ', 'acc from', acc_old, '  to  ',
                              acc_new)
                if acc_new==1.0:
                    break
            if acc_new!=1.0:
                #saver.save(sess, CHECKFILE, global_step=n)
                print('Fail! Maybe SAT dont have an ans')
                print('restore pred checkpoint')
		ckpt = tf.train.get_checkpoint_state('./checkpoint/')
                saver.restore(sess, ckpt.model_checkpoint_path)
        return
def compute_reward(size_of_C,k_sat,C_np,X):
    acc=acc_compute(size_of_C,k_sat,C_np,X)
    return acc

def compute_add_reward(reward_list,size_of_C,j):
    e_greedy=0.9
    reward=0.0
    reward_=0.0
    count=0
    for p in range(len(reward_list)):
        reward+=reward_list[p]
        count+=1
        if count == size_of_C:
            reward*=10^-j
            j-=1
            count=0
            reward_+=reward
            reward=0.0
    reward_=np.array(reward_)
    reward_.shape=(1)
    return e_greedy*reward_