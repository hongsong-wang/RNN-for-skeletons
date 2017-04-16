import os
import numpy as np
import random
import h5py
# import cv2

class ntu_rgbd(object):
    def __init__(self, data_path):
        self._data_path = data_path

    def skeleton_miss_list(self):
        lines = open('data/samples_with_missing_skeletons.txt', 'r').readlines()
        return [line.strip()+'.skeleton' for line in lines]

    def get_multi_subject_list(self):
        lines = open('data/samples_with_multi_subjects.txt', 'r').readlines()
        return [line.strip() for line in lines]

    def filter_list(self, file_list):
        miss_list = self.skeleton_miss_list()
        return list(set(file_list)-set(miss_list))

    def check_list_by_frame_num(self):
        all_list = os.listdir(self._data_path)
        all_list = self.filter_list(all_list)
        for filename in all_list:
            lines = open(os.path.join(self._data_path, filename), 'r').readlines()
            step1 = int(lines[0].strip())
            step2 = lines.count('25\r\n')
            if step2 != step1 and step2 != 2*step1 and step2 != 3*step1:
                print filename, step1, step2

    def cross_subject_split(self):
        print 'cross subject evaluation ...'
        trn_sub = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
        all_list = os.listdir(self._data_path)
        trn_list = [file for file in all_list if int(file[9:12]) in trn_sub]
        tst_list = list(set(all_list) - set(trn_list))
        # filter file list with missing skeleton
        trn_list = self.filter_list(trn_list)
        tst_list = self.filter_list(tst_list)
        return trn_list, tst_list

    def cross_view_split(self):
        print 'cross view evaluation ...'
        trn_view = [2, 3]
        all_list = os.listdir(self._data_path)
        trn_list = [file for file in all_list if int(file[5:8]) in trn_view]
        tst_list = list(set(all_list) - set(trn_list))
        # filter file list with missing skeleton
        trn_list = self.filter_list(trn_list)
        tst_list = self.filter_list(tst_list)
        return trn_list, tst_list

    def get_all_data(self):
        all_list = os.listdir(self._data_path)
        return self.filter_list(all_list)

    def smooth_skeleton(self, skeleton):
        assert(skeleton.shape[2] == 3), ' input must be skeleton array'
        filt = np.array([-3,12,17,12,-3])/35.0
        skt = np.concatenate((skeleton[0:2], skeleton, skeleton[-2:]), axis=0)
        for idx in xrange(2, skt.shape[0]-2):
            skeleton[idx-2] = np.swapaxes(np.dot(np.swapaxes(skt[idx-2:idx+3], 0, -1), filt), 0, -1)
        return skeleton

    def subtract_mean(skeleton, smooth=False):
        if smooth:
            skeleton = self.smooth_skeleton(skeleton)
        # substract mean values
        center = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        for idx in xrange(skeleton.shape[1]):
            skeleton[:, idx] = skeleton[:, idx] - center
        return skeleton

    def load_skeleton_file_multi_subject(self, filename, sub_idx=1, num_joints=25):
        # sub_idx, subject index, 1, 2
        # return ndarray, n_step*n_joint*7 (3 postion, 4 angle)
        lines = open(os.path.join(self._data_path, filename), 'r').readlines()
        # notice: determine the number of step, not sure which is better
        step = int(lines[0].strip())
        skeleton = np.zeros((step, num_joints, 7))
        start = 1
        sidx = [0,1,2,7,8,9,10]
        idx = 0
        while start < len(lines): # and idx < step
            if sub_idx==1:
                if lines[start].strip() in ['1', '2', '3']:
                    skeleton[idx] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                                for line_per in lines[start+1+2:start+26+2]])
                    idx = idx + 1
                    start = start + 26 + 2
                else:
                    start = start + 1
            if sub_idx==2:
                if lines[start].strip() in ['2', '3']:
                    skeleton[idx] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                                for line_per in lines[start+1+2+27:start+26+2+27]])
                    idx = idx + 1
                    start = start + 1 + 26 + 2 + 27
                else:
                    start = start + 1
            if sub_idx==3:
                if lines[start].strip() in ['3']:
                    skeleton[idx] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                                for line_per in lines[start+1+2+27+27:start+26+2+27+27]])
                    idx = idx + 1
                    start = start + 1 + 26 + 2 + 27 + 27
                else:
                    start = start + 1
        return skeleton[0:idx]

    def load_skeleton_file(self, filename, num_joints=25):
        # return ndarray, n_step*n_joint*7 (3 postion, 4 angle)
        lines = open(os.path.join(self._data_path, filename), 'r').readlines()
        # notice: determine the number of step, not sure which is better
        if 0:
            step = int(lines[0].strip())
        else:
            step = lines.count('25\r\n')
        skeleton = np.zeros((step, num_joints, 7))
        start = 1
        sidx = [0,1,2,7,8,9,10]
        idx = 0
        while start < len(lines): # and idx < step
            if lines[start].strip()=='25':
                skeleton[idx] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                            for line_per in lines[start+1:start+26]])
                idx = idx + 1
                start = start + 26
            else:
                start = start + 1
        return skeleton

    def load_skeleton_file_list(self, file_list):
        skeleton_list = []
        for file in file_list:
            skeleton = self.load_skeleton_file(file)
            # only use postion or angele
            skeleton = skeleton[:,:, 0:3]
            # skeleton = skeleton[:,:, 3:7]
            skeleton_list.append(skeleton )
        return skeleton_list

    def load_sample_skeleton_file_list(self, file_list, num_seq, start_rand=True, max_start=30):
        skeleton_list = []
        for name in file_list:
            # load and sample skeleton
            skeleton = self.load_skeleton_file(name)
            # only use postion or angele
            skeleton = skeleton[:,:, 0:3]
            # skeleton = skeleton[:,:, 3:7]
            if start_rand:
                start = np.random.randint(0, max_start)
            else:
                start = 0

            if skeleton.shape[0] < start + num_seq:
                # pad zeros in front of skeleton data
                sample = np.concatenate((np.zeros((num_seq-skeleton.shape[0]+start, skeleton.shape[1], skeleton.shape[2])),
                         skeleton[start:skeleton.shape[0]]), axis=0)
            else:
                sidx = np.arange(start, start + num_seq)
                sample = skeleton[sidx]

            skeleton_list.append(sample)

        skeleton_list = np.asarray(skeleton_list, dtype='float32')
        return skeleton_list

    def save_h5_file_skeleton_list(self, save_home, trn_list, split='train', angle=False):
        if 0:
            multi_list = self.get_multi_subject_list()
            one_list = list(set(trn_list) - set(multi_list))
            multi_list = list(set(trn_list) - set(one_list))

        # save file list to txt
        save_name = os.path.join(save_home, 'file_list_' +  split + '.txt')
        with open(save_name, 'w') as fid_txt:  # fid.write(file+'\n')
            # save array list to hdf5
            save_name = os.path.join(save_home, 'array_list_' + split + '.h5')
            with h5py.File(save_name, 'w') as fid_h5:
                for fn in trn_list:
                    skeleton_set, pid_set, std_set = self.person_position_std(fn)
                    # filter skeleton by standard value
                    count = 0
                    for idx2 in xrange(len(pid_set)):
                        if std_set[idx2][0] > 0.1 or std_set[idx2][1] > 0.1:
                            count = count + 1
                            name=fn+pid_set[idx2]
                            if angle:
                                fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                            else:
                                fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                            fid_txt.write(name + '\n')
                    if count == 0:
                        std_sum = [np.sum(it) for it in std_set]
                        idx2 = np.argmax(std_sum)
                        name=fn+pid_set[idx2]
                        if angle:
                            fid_h5[name] = skeleton_set[idx2][:,:, 3:]
                        else:
                            fid_h5[name] = skeleton_set[idx2][:,:, 0:3]
                        fid_txt.write(name + '\n')

    def person_position_std(self, filename, num_joints=25):
        lines = open(os.path.join(self._data_path, filename), 'r').readlines()
        step = int(lines[0].strip())
        pid_set = []
        # idx_set length of sequence
        idx_set = []
        skeleton_set = []
        start = 1
        sidx = [0,1,2,7,8,9,10]
        while start < len(lines): # and idx < step
            if lines[start].strip()=='25':
                pid = lines[start-1].split()[0]
                if pid not in pid_set:
                    idx_set.append(0)
                    pid_set.append(pid)
                    skeleton_set.append(np.zeros((step, num_joints, 7)))
                idx2 = pid_set.index(pid)
                skeleton_set[idx2][idx_set[idx2]] = np.asarray([map(float, np.array(line_per.strip().split())[sidx]) \
                                            for line_per in lines[start+1:start+26]])
                idx_set[idx2] = idx_set[idx2] + 1
                start = start + 26
            else:
                start = start + 1
        std_set=[]
        for idx2 in xrange(len(idx_set)):
            skeleton_set[idx2] = skeleton_set[idx2][0:idx_set[idx2]]
            xm = np.abs(skeleton_set[idx2][1:idx_set[idx2],:,0] - skeleton_set[idx2][0:idx_set[idx2]-1,:,0])
            xm = xm.sum(axis=-1)
            ym = np.abs(skeleton_set[idx2][1:idx_set[idx2],:,1] - skeleton_set[idx2][0:idx_set[idx2]-1,:,1])
            ym = ym.sum(axis=-1)
            std_set.append((np.std(xm), np.std(ym)))
        return skeleton_set, pid_set, std_set

    def save_h5_file_seq(self, save_home, trn_list, trn_label, split='train', num_sample_save = 10000, num_seq=100):
        skeleton_list = []
        label_list = []
        seq_len_list = []
        iter_idx = 0
        save_idx = 0
        for idx, name in enumerate(trn_list):
            # load and sample skeleton
            skeleton = self.load_skeleton_file(name)
            # only use postion or angele
            skeleton = skeleton[:,:, 0:3]
            # skeleton = skeleton[:,:, 3:7]
            if skeleton.shape[0] < num_seq:
                # pad zeros in front of skeleton data
                sample = np.concatenate((np.zeros((num_seq-skeleton.shape[0], skeleton.shape[1], skeleton.shape[2])),
                         skeleton), axis=0)
            else:
                sidx = np.arange(num_seq)
                sample = skeleton[sidx]

            seq_len_list.append(skeleton.shape[0])
            skeleton_list.append(sample)
            label_list.append(trn_label[idx])

            iter_idx = iter_idx + 1
            if iter_idx== num_sample_save:
                # save skeleton
                skeleton_list = np.asarray(skeleton_list, dtype='float32')
                label_list = np.asarray(label_list, dtype='float32')
                seq_len_list = np.asarray(seq_len_list)
                save_name = os.path.join(save_home, 'seq' + str(num_seq) + '_' + split + str(save_idx) + '.h5')
                with h5py.File(save_name, 'w') as f:
                    f['data'] = skeleton_list
                    f['label'] = label_list
                    f['seq_len_list'] = seq_len_list
                save_idx = save_idx + 1
                iter_idx = 0
                skeleton_list = []
                label_list = []
                seq_len_list = []
        if iter_idx > 0:
            skeleton_list = np.asarray(skeleton_list, dtype='float32')
            label_list = np.asarray(label_list, dtype='float32')
            seq_len_list = np.asarray(seq_len_list)
            save_name = os.path.join(save_home, 'seq' + str(num_seq) + '_' + split + str(save_idx) + '.h5')
            with h5py.File(save_name, 'w') as f:
                f['data'] = skeleton_list
                f['label'] = label_list
                f['seq_len_list'] = seq_len_list

    def calculate_height(self, skeleton):
        center1 = (skeleton[:,2,:] + skeleton[:,8,:] + skeleton[:,4,:] + skeleton[:,20,:])/4
        w1 = skeleton[:,23,:] - center1
        w2 = skeleton[:,22,:] - center1
        center2 = (skeleton[:,1,:] + skeleton[:,0,:] + skeleton[:,16,:] + skeleton[:,12,:])/4
        h0 = skeleton[:,3,:] - center2
        h1 = skeleton[:,19,:] - center2
        h2 = skeleton[:,15,:] - center2
        width = np.max([np.max(np.abs(w1[:,0])), np.max(np.abs(w2[:,0]))])
        heigh1 = np.max(h0[:,1])
        heigh2 = np.max([np.max(np.abs(h1[:,1])), np.max(np.abs(h2[:,1]))])
        return np.asarray([width, heigh1, heigh2])

    def caculate_person_height(self, h5_file, list_file):
        # average value of different person: 0.36026082  0.61363413  0.76827  (mean for each person)
        # average value of different person: 1.67054954  0.87844846  1.28303429 (max for each person)
        # average value of different person: 0.0680575   0.19834167  0.21219039 (min for each person)
        name_list = [line.strip() for line in open(list_file, 'r').readlines()]
        pid_list = np.array([int(name[9:12]) for name in name_list])
        with h5py.File(h5_file,'r') as hf:
            wh_set = []
            for pid in set(pid_list):
                sidx = np.where(pid_list==pid)[0]
                wh = np.zeros((len(sidx), 3))
                for i, idx in enumerate(sidx):
                    name = name_list[idx]
                    skeleton = np.asarray(hf.get(name))
                    wh[i] = self.calculate_height(skeleton)
                wh_set.append(wh.max(axis=0)) # notice: mean or max for different position, view points
            wh_set = np.asarray(wh_set)
            print wh_set.mean(axis=0)

if __name__ == '__main__':
    data_path = '/home/wanghongsong/data/NTURGBD/nturgb+d_skeletons/'
    db = ntu_rgbd(data_path)
    db.load_skeleton_file('S011C001P028R001A034.skeleton')
    # db.caculate_person_height('data/seq/array_list_all_data.h5', 'data/seq/file_list_all_data.txt')

    if 0:
        if 1:
            trn_list, tst_list = db.cross_subject_split()
            # trn_list, tst_list = db.cross_view_split()
            db.save_h5_file_skeleton_list('data/subj_seq', trn_list, split='train')
            db.save_h5_file_skeleton_list('data/subj_seq', tst_list, split='test')

        if 0:
            db.save_h5_file_skeleton_list('data/seq', db.get_all_data(), split='angle_all_data', angle=True)
            db2 = ntu_rgbd('/home/wanghongsong/data/NTURGBD/AllSkeletonFiles_remove_nan_nolabel')
            db2.save_h5_file_skeleton_list('data/seq', db2.get_all_data(), split='angle_final_test', angle=True)
