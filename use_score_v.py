import sys
import pickle
import numpy as np
import os
import argparse
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def use(reflist, predlist, batchsize):
    import tensorflow_hub as tfhub

    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    model = tfhub.load(module_url)
    refs = list()
    preds = list()
    count = 0
    for ref, pred in zip(reflist, predlist):
		#print(ref)        
        ref = ' '.join(ref).strip()
        pred = ' '.join(pred).strip()
        if pred == '':
            pred = ' <s> '
			#count+=1
			#continue
        refs.append(ref)
        preds.append(pred)
	#total_csd = np.zeros(count)
    scores = list()
    for i in range(0, len(refs), batchsize):
        ref_emb = model(refs[i:i+batchsize])
        pred_emb = model(preds[i:i+batchsize])
        csm = cosine_similarity_score(ref_emb, pred_emb)
        csd = csm.diagonal()
        total_csd = csd #np.concatenate([total_csd, csd])
        scores = total_csd.tolist()
	#print(count)
    avg_css = np.average(total_csd)

    corpuse = (round(avg_css*100, 2))
    ret = ('for %s functions\n' % (len(predlist)))
    ret+= 'cosine similarity score with universal sentence encoder embedding is %s\n' % corpuse
    return ret

def cosine_similarity_score(x, y):
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_similarity_matrix = cosine_similarity(x, y)
    return cosine_similarity_matrix

def prep_dataset(reffile, predfile, batchsize, delim, tdats, sdats, lim_overlap_sdats):
    preds = dict()
    predicts = open(predfile, 'r')
    for c, line in enumerate(predicts):
        try:
            split_line = line.split('\t')
            (fid, pred) = split_line[0], split_line[-1]
            try:
                fid = int(fid)
            except:
                pass
            pred = pred.split()
            pred = fil(pred)
            preds[fid] = pred
        except:
            continue
    predicts.close()

    refs = list()
    newpreds = list()
    d = 0
    targets = open(reffile, 'r')
    for line in targets:
        (fid, com) = line.split(delim)
        try:
            fid = int(fid)
        except:
            pass
        com = com.split()
        com = fil(com)

        if lim_overlap_sdats > -1:
			# remove the tdats from the sdats
            a_multiset = collections.Counter(sdats[fid])
            b_multiset = collections.Counter(tdats[fid])
			#overlap = list((a_multiset & b_multiset).elements())
            a_remainder = list((a_multiset - b_multiset).elements())
			#b_remainder = list((b_multiset - a_multiset).elements())

            s = set(set(com) & set(a_remainder)) # words in sdats and coms
            t = set(set(com) & set(tdats[fid]))
            o = set(set(a_remainder) - set(tdats[fid]))
            s_o = set(set(o) & set(com)) # words in sdats (but not tdats) and coms

            o_s = len(s_o)
            o_t = len(t)

			#if not(o_s > lim_overlap_sdats):
            if not(o_s == lim_overlap_sdats):
                continue

        if len(com) < 1:
            continue
        try:
            newpreds.append(preds[fid])
        except Exception as ex:
            continue

        refs.append(com)
    print('final status')
    score = use(refs, newpreds, batchsize)
    return score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='./data/jam_cgpt_170k')
    parser.add_argument('--coms-filename', dest='comsfilename', type=str, default='cgptcom.test')
    parser.add_argument('--tdats-filename', dest='tdatsfilename', type=str, default='tdats.test')
    parser.add_argument('--sdats-filename', dest='sdatsfilename', type=str, default='sdats.test')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=50000)
	# parser.add_argument('--data', dest='datapath', type=str, default='/nfs/projects/simmetrics/data/standard/output')
    parser.add_argument('--gpu', dest='gpu', type=str, default='')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--vmem-limit', dest='vmemlimit', type=int, default=0)
    parser.add_argument('--delim', dest='delim', type=str, default='<SEP>')
    parser.add_argument('--lim-overlap-sdats', dest='limoverlapsdats', type=int, default=-1)
    args = parser.parse_args()
    predfile = args.input
    comsfile = args.comsfilename
    tdatsfilename = args.tdatsfilename
    sdatsfilename = args.sdatsfilename
    batchsize = args.batchsize
    dataprep = args.dataprep
    gpu = args.gpu
    challenge = args.challenge
    vmemlimit = args.vmemlimit
    delim = args.delim
    lim_overlap_sdats = args.limoverlapsdats

    comsfile = dataprep + '/' + comsfile

    tdats = dict()
    sdats = dict()

    if lim_overlap_sdats != -1:
        tdatsf = open('%s/%s' % (dataprep, tdatsfilename), 'r')
        for c, line in enumerate(tdatsf):
            (fid, tdat) = line.split(delim)
            try:
                fid = int(fid)
            except ValueError:
                pass
            tdat = tdat.split()
            tdat = fil(tdat)
            tdats[fid] = tdat
        tdatsf.close()

        sdatsf = open('%s/%s' % (dataprep, sdatsfilename), 'r')
        for c, line in enumerate(sdatsf):
            (fid, sdat) = line.split(delim)
            fid = int(fid)
            sdat = sdat.split()
            sdat = fil(sdat)
            sdats[fid] = sdat
        sdatsf.close()

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    if(vmemlimit > 0):
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=vmemlimit)])
            except RuntimeError as e:
                print(e)
	
    score = prep_dataset(comsfile, predfile, batchsize, delim, tdats, sdats, lim_overlap_sdats)
    print(score)
