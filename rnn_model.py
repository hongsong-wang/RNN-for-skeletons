import numpy as np
import random
import math
import h5py
from scipy.io import savemat
import theano
from theano import tensor as T
import lasagne
import lasagne.layers as layers
from lasagne.nonlinearities import softmax, sigmoid, rectify

class construct_model(object):
    def __init__(self, param, dim_point=3, num_joints=25, num_class=60):
        self._param = param
        self._dim_point = dim_point
        self._num_joints = num_joints
        self._num_class = num_class

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=True):
        assert len(inputs) == len(targets)
        indices = np.arange(len(inputs))
        if shuffle:
            np.random.shuffle(indices)
        if 0:
            for start_idx in range(0, len(inputs) - len(inputs) % batchsize, batchsize):
                excerpt = indices[start_idx:start_idx + batchsize]
                y = [targets[s] for s in excerpt]
                x = np.asarray([inputs[s] for s in excerpt])
                yield x, y
        start_idx = 0
        num_batch = (len(inputs) + batchsize - 1) / batchsize
        for batch_idx in range(0, num_batch):
            if batch_idx == num_batch-1:
                excerpt = indices[start_idx:]
            else:
                excerpt = indices[start_idx:start_idx + batchsize]
            start_idx = start_idx + batchsize
            y = [targets[s] for s in excerpt]
            x = np.asarray([inputs[s] for s in excerpt])
            yield x, y

    def smooth_skeleton(self, skeleton):
        assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        filt = np.array([-3,12,17,12,-3])/35.0
        skt = np.concatenate((skeleton[0:2], skeleton, skeleton[-2:]), axis=0)
        for idx in xrange(2, skt.shape[0]-2):
            skeleton[idx-2] = np.swapaxes(np.dot(np.swapaxes(skt[idx-2:idx+3], 0, -1), filt), 0, -1)
        return skeleton

    def calculate_height(self, skeleton):
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        w1 = skeleton[:,23,:] - center1
        w2 = skeleton[:,22,:] - center1
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
        h0 = skeleton[:,3,:] - center2
        h1 = skeleton[:,19,:] - center2
        h2 = skeleton[:,15,:] - center2
        width = np.max([np.max(np.abs(w1[:,0])), np.max(np.abs(w2[:,0]))])
        heigh = np.max([np.max(np.abs(h1[:,1])), np.max(np.abs(h2[:,1])), np.max(h0[:,1])])
        return width, heigh

    def subtract_mean(self, skeleton, smooth=False, scale=True):
        if smooth:
            skeleton = self.smooth_skeleton(skeleton)
        # substract mean values
        # notice: use two different mean values to normalize skeleton data
        if 0:
            center = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
            # center = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
            for idx in xrange(skeleton.shape[1]):
                skeleton[:, idx] = skeleton[:, idx] - center
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4

        for idx in [24,25,12,11,10,9, 5,6,7,8,23,22]:
            skeleton[:, idx-1] = skeleton[:, idx-1] - center1
        for idx in (set(range(1, 1+skeleton.shape[1]))-set([24,25,12,11,10,9,  5,6,7,8,23,22])):
            skeleton[:, idx-1] = skeleton[:, idx-1] - center2

        if scale:
            width, heigh = self.calculate_height(skeleton)
            scale_factor1, scale_factor2 = 0.36026082, 0.61363413
            skeleton[:,:,0] = scale_factor1*skeleton[:,:,0]/width
            skeleton[:,:,1] = scale_factor2*skeleton[:,:,1]/heigh
        return skeleton

    def load_sample_step_list(self, h5_file, list_file, num_seq, step=1, start_zero=True, sub_mean=False, scale=False, smooth=False):
        name_list = [line.strip() for line in open(list_file, 'r').readlines()]
        label_list = [(int(name[17:20])-1) for name in name_list]
        X = []
        label = []
        with h5py.File(h5_file,'r') as hf:
            for idx, name in enumerate(name_list):
                skeleton = np.asarray(hf.get(name))
                if sub_mean:
                    skeleton = self.subtract_mean(skeleton, smooth=smooth, scale=scale)
                for start in range(0, 1 if start_zero else step):
                    skt = skeleton[start:skeleton.shape[0]:step]
                    if skt.shape[0] > num_seq:
                        # process sequences longer than num_seq, sample two sequences, if start_zero=True, only sample once from 0
                        for sidx in ([np.arange(num_seq)] if start_zero else [np.arange(num_seq), np.arange(skt.shape[0]-num_seq, skt.shape[0])]):
                            X.append(skt[sidx])
                            label.append(label_list[idx])
                    else:
                        if skt.shape[0] < 0.05*num_seq: # skip very small sequences
                            continue
                        skt = np.concatenate((np.zeros((num_seq-skt.shape[0], skt.shape[1], skt.shape[2])), skt), axis=0)
                        X.append(skt)
                        label.append(label_list[idx])

        X = np.asarray(X)
        label = (np.asarray(label)).astype(np.int32)
        # rearrange skeleton data by part
        X = self.divide_skeleton_part(X)
        # X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]*X.shape[3]))
        X = X.astype(np.float32)
        return X, label

    def divide_skeleton_part(self, X, num_joints=25):
        # two arms, two legs and one trunk, index from left to right, top to bottom
        # arms: [24,25,12,11,10,9]  [5,6,7,8,23,22]
        # legs: [20,19,18,17]  [13,14,15,16]
        # trunk: [4, 3, 21, 2, 1]
        assert(X.shape[2] == num_joints), ' skeleton must has %d joints'%num_joints
        sidx_list = [np.asarray([24,25,12,11,10,9]), np.asarray([5,6,7,8,23,22]),
                    np.asarray([20,19,18,17]), np.asarray([13,14,15,16]), np.asarray([4, 3, 21, 2, 1])]

        slic_idx = [it*X.shape[3] for it in [0, 6, 6, 4, 4, 5] ]
        slic_idx = np.cumsum(slic_idx )

        X_new = np.zeros((X.shape[0], X.shape[1], slic_idx[-1]))
        for idx, sidx in enumerate(sidx_list):
            sidx = sidx - 1 # index starts from 0
            X_temp = X[:,:,sidx,:]
            X_new[:,:,slic_idx[idx]:slic_idx[idx+1]] = np.reshape(X_temp, (X_temp.shape[0], X_temp.shape[1], X_temp.shape[2]*X_temp.shape[3]))
        return X_new

    def rand_view_transform(self, X, angle1=-10, angle2=10, s1=0.9, s2=1.1):
        # skeleton data X, tensor3
        # genearte rand matrix
        random.random()
        agx = random.randint(angle1, angle2)
        agy = random.randint(angle1, angle2)
        s = random.uniform(s1, s2)
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        # X0 = np.reshape(X,(-1,3))*Ry*Rx*Ss
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        X = X.astype(np.float32)
        return X

    def train_lstm_model(self):
        def bi_direction_lstm(input, hid, only_return_final=False):
            l_forward = lasagne.layers.LSTMLayer(input, hid, nonlinearity=lasagne.nonlinearities.tanh,
                               backwards=False, only_return_final=only_return_final)
            l_backward = lasagne.layers.LSTMLayer(input, hid, nonlinearity=lasagne.nonlinearities.tanh,
                               backwards=True, only_return_final=only_return_final)
            return lasagne.layers.ConcatLayer([l_forward, l_backward], axis=-1)

        input_var = T.tensor3('X')
        target_var = T.ivector('y')

        ## define recurrent neural networks structure
        print 'construct theano network ...'
        if self._param['model_type'] == 'whole':
            network = lasagne.layers.InputLayer((None, None, self._dim_point*self._num_joints), input_var)
            network = bi_direction_lstm(network, 512, only_return_final=False)
            network = bi_direction_lstm(network, 512, only_return_final=False)
            network = lasagne.layers.ExpressionLayer(network, lambda X: X.max(1), output_shape='auto')
            network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5), self._num_class, nonlinearity=softmax)

        if self._param['model_type']=='part':
            new_slic_idx = [it*self._dim_point for it in [0, 6, 6, 4, 4, 5] ]
            new_slic_idx = np.cumsum(new_slic_idx )
            net_list = []
            input_data = lasagne.layers.InputLayer((None, None, new_slic_idx[-1]), input_var)
            for slc_id in xrange(0, len(new_slic_idx)-1):
                data_per = lasagne.layers.SliceLayer(input_data, indices=slice(new_slic_idx[slc_id], new_slic_idx[slc_id+1]), axis=-1)
                rnn_per = bi_direction_lstm(data_per, 256, only_return_final=False)
                # rnn_per = bi_direction_lstm(data_per, 256, only_return_final=False)
                net_list.append(rnn_per)
            network = lasagne.layers.ConcatLayer(net_list, axis=-1)
            network = bi_direction_lstm(network, 512, only_return_final=False)
            network = lasagne.layers.ExpressionLayer(network, lambda X: X.max(1), output_shape='auto')
            network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5), self._num_class, nonlinearity=softmax) # default 0.5

        if self._param['model_type'] == 'part_hier':
            new_slic_idx = [it*self._dim_point for it in [0, 6, 6, 4, 4, 5] ]
            new_slic_idx = np.cumsum(new_slic_idx )
            net_list1 = []
            input_data = lasagne.layers.InputLayer((None, None, new_slic_idx[-1]), input_var)
            for slc_id in xrange(0, len(new_slic_idx)-1):
                data_per = lasagne.layers.SliceLayer(input_data, indices=slice(new_slic_idx[slc_id], new_slic_idx[slc_id+1]), axis=-1)
                rnn_per = bi_direction_lstm(data_per, 64, only_return_final=False)
                net_list1.append(rnn_per)
            comb_part_list1 = [[0, 4], [1, 4], [2, 4], [3, 4]]
            net_list2 = []
            for comb_part in comb_part_list1:
                data_per = lasagne.layers.ConcatLayer([net_list1[comb_part[0]], net_list1[comb_part[1]]], axis=-1)
                rnn_per = bi_direction_lstm(data_per, 128, only_return_final=False)
                net_list2.append(rnn_per)
            comb_part_list2 = [[0, 1], [2, 3]]
            net_list3 = []
            for comb_part in comb_part_list2:
                data_per = lasagne.layers.ConcatLayer([net_list2[comb_part[0]], net_list2[comb_part[1]]], axis=-1)
                rnn_per = bi_direction_lstm(data_per, 256, only_return_final=False)
                net_list3.append(rnn_per)
            network = lasagne.layers.ConcatLayer(net_list3, axis=-1)
            network = bi_direction_lstm(network, 512, only_return_final=False)
            network = lasagne.layers.ExpressionLayer(network, lambda X: X.max(1), output_shape='auto')
            network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5), self._num_class, nonlinearity=softmax)

        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean() + self._param['weight_regular'] * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

        params = lasagne.layers.get_all_params(network, trainable=True)
        learning_rate = theano.shared(np.float32( self._param['base_learn_rate'] ))
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=learning_rate, momentum=0.9)
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        validation_fn = theano.function([input_var, target_var], loss)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        predict_fn = theano.function([input_var], test_prediction)

        trainX, trainY = self.load_sample_step_list(self._param['trn_arr_file'], self._param['trn_lst_file'], self._param['num_seq'],
                step=self._param['step'], start_zero=False, scale=self._param['scale'],
                sub_mean=self._param['sub_mean'], smooth=self._param['smooth'])
        valX, valY = self.load_sample_step_list(self._param['tst_arr_file'], self._param['tst_lst_file'], self._param['num_seq'],
                step=self._param['step'], start_zero=True, scale=self._param['scale'],
                sub_mean=self._param['sub_mean'], smooth=self._param['smooth'])

        for epoch in range( self._param['max_iter'] ):
            if self._param['initial_file'] != None and epoch==0:
                with h5py.File(self._param['initial_file'],'r') as f:
                    param_values = []
                    for i in range(0, len(f.keys())):
                        param_values.append( np.asarray(f.get('arr_%d' % i)) )
                    lasagne.layers.set_all_param_values(network, param_values)
                    if self._param['test_phase']:
                        val_predictions = np.zeros((0, self._num_class))
                        for x, y in self.iterate_minibatches(valX, valY, self._param['batchsize'], shuffle=False):
                            val_predictions = np.concatenate((val_predictions, predict_fn(x)), axis=0)
                        save_name='data/result.mat'
                        savemat(save_name, mdict={'predictions':val_predictions, 'gt_label':valY} )
                    return

            if self._param['resume_epoch'] and epoch <= self._param['resume_epoch'] and self._param['initial_file'] == None:
                if (epoch > 0 and epoch % self._param['step_inter'] == 0):
                    learning_rate.set_value(np.float32(learning_rate.get_value() * self._param['lr_gamma']))

                if epoch== self._param['resume_epoch']:
                    print 'load parameters and initialize network ...'
                    para_file = self._param['save_path'] + '_epoch%d.h5'%epoch
                    with h5py.File(para_file, 'r') as f:
                        param_values = []
                        for i in range(0, len(f.keys())):
                            param_values.append( np.asarray(f.get('arr_%d' % i)) )
                        lasagne.layers.set_all_param_values(network, param_values)
                continue

            train_loss = 0
            train_batches = 0
            for x, y in self.iterate_minibatches(trainX, trainY, self._param['batchsize']):
                if self._param['rand_start'] and random.random()>0.5:
                    start = random.randint(0, int(self._param['max_start_rate']*x.shape[1]))
                    x = x[:, start:]
                if self._param['rand_view'] and random.random()>0.5:
                    x = self.rand_view_transform(x)
                train_loss = train_loss + train_fn(x, y)
                train_batches += 1
            train_loss = train_loss/train_batches

            if epoch % 2 == 0:
                val_loss = 0
                val_batches = 0
                for x, y in self.iterate_minibatches(valX, valY, self._param['batchsize']):
                    val_loss = val_loss + validation_fn(x, y)
                    val_batches += 1
                val_loss = val_loss/val_batches

                print "Epoch %d: train_loss = %f, val_loss = %f, lr = %f" % (epoch, train_loss, val_loss, learning_rate.get_value())

            if (epoch > 0 and epoch % self._param['step_inter'] == 0):
                learning_rate.set_value(np.float32(learning_rate.get_value() * self._param['lr_gamma']))

            if epoch % 4 == 0 and epoch > 0:
                ## evaluate results
                val_predictions = np.zeros((0, self._num_class))
                for x, y in self.iterate_minibatches(valX, valY, self._param['batchsize'], shuffle=False):
                    val_predictions = np.concatenate((val_predictions, predict_fn(x)), axis=0)

                pred_val = np.argmax(val_predictions, axis=1)
                print ('evluation epoch=%d/%d, accuracy=%f' % (epoch, self._param['max_iter'],
                        (sum( int(pred_val[i]) == valY[i] for i in xrange(len(pred_val))) / float(len(pred_val)) )) )

                ## save parameters to file
                save_file = self._param['save_path'] + '_epoch%d.h5'%epoch
                # np.savez(save_file, *lasagne.layers.get_all_param_values(network))
                para_lst = lasagne.layers.get_all_param_values(network)
                with h5py.File(save_file, 'w') as fid:
                    for i in range(len(para_lst)):
                        fid['arr_%d' % i] = para_lst[i]

def run_model():
    param = {}
    param['max_iter'] = 2000
    param['step_inter'] = 40 # initial 60
    param['base_learn_rate'] = 0.02
    param['lr_gamma'] = 0.7
    param['weight_regular'] = 0
    param['batchsize'] = 256 # 256

    param['trn_arr_file'] = 'data/view_seq/array_list_train.h5'
    param['trn_lst_file'] = 'data/view_seq/file_list_train.txt'
    param['tst_arr_file'] = 'data/view_seq/array_list_test.h5'
    param['tst_lst_file'] = 'data/view_seq/file_list_test.txt'

    # 'part', part based lstm model
    # 'part_hier', hierarchical part model
    # 'whole'
    param['model_type'] = 'part'
    param['save_path'] = 'data/view_seq/save_param/part' # save prefix, part_hier
    param['initial_file'] = None
    param['resume_epoch'] = 0
    param['test_phase'] = False

    param['num_seq'] = 100
    param['step'] = 1
    param['rand_start'] = True
    param['max_start_rate'] = 0.3 # 0.3*length
    param['rand_view'] = False

    param['sub_mean']=True
    param['scale']=False
    param['smooth']=False

    model = construct_model(param)
    model.train_lstm_model()

if __name__ == '__main__':
    run_model()
