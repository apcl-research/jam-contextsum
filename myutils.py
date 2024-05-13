import sys
import javalang
from timeit import default_timer as timer
import tensorflow.keras as keras
import tensorflow.keras.utils
import numpy as np
import tensorflow as tf
import networkx as nx
import random
import threading
import sqlite3
import os
#import tokenizer
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
start = 0
end = 0

def prep(msg):
    global start
    statusout(msg)
    start = timer()

def statusout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()

def drop():
    global start
    global end
    end = timer()
    sys.stdout.write('done, %s seconds.\n' % (round(end - start, 2)))
    sys.stdout.flush()

def index2word(tok):
	i2w = {}
	for word, index in tok.w2i.items():
		i2w[index] = word

	return i2w

def seq2sent(seq, tokenizer):
    sent = []
    check = index2word(tokenizer)
    for i in seq:
        sent.append(check[i])

    return(' '.join(sent))

      
class batch_gen_lm(tensorflow.keras.utils.Sequence):
    def __init__(self, seqdata, extradata, tt, config, training=True):
        self.tt = tt
        self.batch_size = config['batch_size']

        self.extradata = extradata

        self.seqdata = dict()
        self.seqdata['dt%s' % tt] = seqdata.get('/dt%s' % tt)
        self.seqdata['ds%s' % tt] = seqdata.get('/ds%s' % tt)

        self.allfidlocs = list(range(0, np.array(self.seqdata['dt%s' % tt]).shape[0]))
        self.config = config
        self.training = training
        
        self.tdatvocabsize = config['tdatvocabsize']
        
        if not self.training:
            tdatstart = np.zeros(self.config['tdatlen'])
            stk = self.extradata['comstok'].w2i['<s>']
            tdatstart[0] = stk
            self.tdatstart = tdatstart
        
        random.shuffle(self.allfidlocs)

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfidlocs = self.allfidlocs[start:end]
        return self.make_batch(batchfidlocs)

    def make_batch(self, batchfidlocs):
        return self.divideseqs(batchfidlocs)

    def __len__(self):
        return int(np.ceil(np.array(self.seqdata['dt%s' % (self.tt)]).shape[0])/self.batch_size)

    def on_epoch_end(self):
        random.shuffle(self.allfidlocs)

    def divideseqs(self, batchfidlocs):
        tdatseqs = list()
        sdatseqs = list()
        
        badfids = list()
        tdatouts = list()

        fiddat = dict()
        
        with_tdats = False
        with_sdats = False

        for c, inp in enumerate(self.config['batch_config'][0]):
            if inp == 'tdat':
                with_tdats = True
            elif inp == 'sdat':
                with_sdats = True

        for fidloc in batchfidlocs:

            fid = self.config['locfid']['c%s' % self.tt][fidloc]

            if with_tdats:
                #wtdatseq = self.get_tdat(fid, self.config['tdatlen'], filedb, filedbcur)
                wtdatseq = self.seqdata['dt%s' % self.tt][fidloc]
                wtdatseq = wtdatseq[:self.config['tdatlen']]

            if with_sdats:
                wsdatseq = self.seqdata['ds%s' % self.tt][fidloc]
                wsdatseq = wsdatseq[:self.config['sdatlen']]
                wsdatseq = np.delete(wsdatseq, slice(self.config['stdatlen'],None), 1)

            if not self.training:
                wcomseq = self.comstart
                inps = list()
                for inp in self.config['batch_config'][0]:
                    if inp == 'tdat':
                        inps.append(wtdatseq)
                    elif inp == 'sdat':
                        inps.append(wsdatseq)

                fiddat[fid] = inps
            else:
                for i in range(0, len(wtdatseq)):
                    #if with_tdats:
                    #    tdatseqs.append(wtdatseq)
                    if with_sdats:
                        sdatseqs.append(wsdatseq)
                        
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    tdatseq = wtdatseq[0:i]
                    tdatout = wtdatseq[i]
                    tdatout = keras.utils.to_categorical(tdatout, num_classes=self.tdatvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wtdatseq)):
                        try:
                            tdatseq[j]
                        except IndexError as ex:
                            tdatseq = np.append(tdatseq, 0)

                    tdatseqs.append(tdatseq)
                    tdatouts.append(np.asarray(tdatout))

        if self.training:
            if with_sdats:
                sdatseqs = np.asarray(sdatseqs)
                
            tdatseqs = np.asarray(tdatseqs)
            tdatouts = np.asarray(tdatouts)
            
            inps = list()
            oups = list()
            
            for inp in self.config['batch_config'][0]:
                if inp == 'tdat':
                    inps.append(tdatseqs)
                elif inp == 'sdat':
                    inps.append(sdatseqs)
        
            for oup in self.config['batch_config'][1]:
                if oup == 'tdatout':
                    oups.append(tdatouts)
                
            if len(oups) == 1:
                oups = oups[0]
        
        if not self.training:
            return (fiddat, badfids)
        else:
            return (inps, oups)

    def idx2tok(self, nodelist, path):
        out = list()
        for idx in path:
            out.append(nodelist[idx])
        return out

            
class batch_gen(tensorflow.keras.utils.Sequence):
    def __init__(self, seqdata, extradata, tt, config, training=True):
        self.comvocabsize = config['comvocabsize']
        self.tt = tt
        self.batch_size = config['batch_size']

        self.extradata = extradata

        #self.filedb = sqlite3.connect(sqlfile)
        #self.filedbcur = self.filedb.cursor()
        #self.sqlfile = sqlfile

        self.seqdata = dict()
        self.seqdata['dt%s' % tt] = seqdata.get('/dt%s' % tt)
        self.seqdata['ds%s' % tt] = seqdata.get('/ds%s' % tt)
        self.seqdata['s%s' % tt] = seqdata.get('/s%s' % tt)
        self.seqdata['c%s' % tt] = seqdata.get('/c%s' % tt)

        self.allfidlocs = list(range(0, np.array(self.seqdata['dt%s' % tt]).shape[0]))
        self.config = config
        self.training = training
        
        if not self.training:
            comstart = np.zeros(self.config['comlen'])
            stk = self.extradata['comstok'].w2i['<s>']
            comstart[0] = stk
            self.comstart = comstart
        
        if 'tdat_sent' in self.config['batch_config'][0]:
            tdatstok = self.extradata['tdatstok']
            self.nltok = tdatstok.w2i['<NL>']
        
        random.shuffle(self.allfidlocs)

    def __getitem__(self, idx):
        start = (idx*self.batch_size)
        end = self.batch_size*(idx+1)
        batchfidlocs = self.allfidlocs[start:end]
        return self.make_batch(batchfidlocs)

    def make_batch(self, batchfidlocs):
        return self.divideseqs(batchfidlocs)

    def __len__(self):
        return int(np.ceil(np.array(self.seqdata['dt%s' % (self.tt)]).shape[0])/self.batch_size)

    def on_epoch_end(self):
        random.shuffle(self.allfidlocs)

    #def get_tdat(self, fid, maxlen, filedb, filedbcur):
    #    tdatstok = self.extradata['tdatstok']
    #    filedbcur.execute('select tdat from fundats where fid={}'.format(fid))
    #    filedb.commit()
    #    for tdat in filedbcur.fetchall():
    #        tdatraw = tdat[0]
    #    tdat = tdatstok.texts_to_sequences(tdatraw, maxlen=maxlen)[0]
    #    return(tdat)

    def divideseqs(self, batchfidlocs):
        tdatseqs = list()
        tdatsentseqs = list()
        sdatseqs = list()
        comseqs = list()
        smlseqs = list()
        smlpaths = list()
        smlnodes = list()
        smledges = list()
        callnodes = list()
        calledges = list()
        bios = list()
        comouts = list()
        badfids = list()

        fiddat = dict()
        
        with_tdats = False
        with_sdats = False
        with_coms = False
        with_smlseqs = False
        with_smlpaths = False
        with_smlnodes = False
        with_smledges = False
        with_callnodes = False
        with_calledges = False
        with_bio = False
        with_tdat_sents = False 
        #filedb = sqlite3.connect(self.sqlfile)
        #filedbcur = filedb.cursor()

        comseqpos = 0
        
        for c, inp in enumerate(self.config['batch_config'][0]):
            if inp == 'tdat':
                with_tdats = True
            elif inp == 'tdat_sent':
                with_tdat_sents = True
            elif inp == 'sdat':
                with_sdats = True
            elif inp == 'com':
                with_coms = True
                comseqpos = c
            elif inp == 'smlseq':
                with_smlseqs = True
            elif inp == 'smlpath':
                with_smlpaths = True
            elif inp == 'smlnode':
                with_smlnodes = True
            elif inp == 'smledge':
                with_smledges = True
            elif inp == 'callnode':
                with_callnodes = True
            elif inp == 'calledge':
                with_calledges = True
            elif inp == 'bio':
                with_bio = True

        for fidloc in batchfidlocs:

            fid = self.config['locfid']['c%s' % self.tt][fidloc]

            wcomseq = self.seqdata['c%s' % self.tt][fidloc]
            wcomseq = wcomseq[:self.config['comlen']]
            #fid = wcomseq[:1][0]
            #wcomseq = wcomseq[1:self.config['comlen']+1]

            if with_tdats or with_tdat_sents:
                #wtdatseq = self.get_tdat(fid, self.config['tdatlen'], filedb, filedbcur)
                wtdatseq = self.seqdata['dt%s' % self.tt][fidloc]
                wtdatseq = wtdatseq[:self.config['tdatlen']]
                if with_tdat_sents:
                    wtdatsentseq = list()
                        # for each row in wtdatseq
                        # split the row into lines by self.nltok
                    sentseq = list()
                    i = 0
                    while i < len(wtdatseq):
                        word = wtdatseq[i]
                        if word == 0:
                                # 0 should be padding,
                                # so we shouldn't need to check the rest of the data
                            break

                        if len(sentseq) < self.config['max_sentence_len'] - 1:
                            sentseq.append(word)
                            if word == self.nltok:
                                sentseq.extend([0] * (self.config['max_sentence_len'] - len(sentseq))) # padding
                                wtdatsentseq.append(sentseq)
                                sentseq = list()

                            # if sentseq has reached the limit, we will omit the words
                        elif len(sentseq) == self.config['max_sentence_len'] - 1:
                            sentseq.append(self.nltok) # make sure the last word in the line is <NL>
                            wtdatsentseq.append(sentseq)
                            sentseq = list()
                            while word != self.nltok and i < len(wtdatseq) - 1:
                                i = i + 1
                                word = wtdatseq[i]

                        i = i + 1


                    if sentseq:
                        sentseq.append(self.nltok) # append a <NL> to the end of the last bits
                        sentseq.extend([0] * (self.config['max_sentence_len'] - len(sentseq))) # padding
                        wtdatsentseq.append(sentseq) # put the last bits to the wtdatsentseq
                        del sentseq

                        # padding
                    wtdatsentseq = wtdatsentseq[:self.config['max_sentence_cnt']]
                    if (len(wtdatsentseq) < self.config['max_sentence_cnt']):
                        padding_sent = [0] * self.config['max_sentence_len']
                        wtdatsentseq.extend([padding_sent] *
                                            (self.config['max_sentence_cnt'] - len(wtdatsentseq)))

                        # double check the size of the wtdatsentseq to make sure it has the correct shape
                        # assert len(wtdatsentseq) == self.config['max_sentence_cnt']
                        # for sentseq in wtdatsentseq:
                        #     assert len(sentseq) == self.config['max_sentence_len']

            if with_sdats:
                wsdatseq = self.seqdata['ds%s' % self.tt][fidloc]
                wsdatseq = wsdatseq[:self.config['sdatlen']]
                wsdatseq = np.delete(wsdatseq, slice(self.config['stdatlen'],None), 1)

            if with_smlseqs:
                wsmlseq = self.seqdata['s%s' % self.tt][fidloc]
                wsmlseq = wsmlseq[:self.config['smllen']]

            if with_smlnodes or with_smlpaths:
                wsmlnodes = self.extradata['s%s_nodes' % (self.tt)][fid]
                wsmlnodeslen = len(wsmlnodes)

                # crop/expand ast sequence
                wsmlnodes = wsmlnodes[:self.config['maxastnodes']]
                tmp = np.zeros(self.config['maxastnodes'], dtype='int32')
                tmp[:wsmlnodes.shape[0]] = wsmlnodes
                wsmlnodes = np.int32(tmp)

            if with_smledges or with_smlpaths:
                wsmledges = self.extradata['s%s_edges' % (self.tt)][fid]
                if (wsmledges.shape[0] > 1000):
                    badfids.append(fid)
                    continue
            
                # crop/expand ast adjacency matrix to dense
                wsmledges = np.asarray(wsmledges.todense())
                wsmledges = wsmledges[:self.config['maxastnodes'], :self.config['maxastnodes']]
                tmp = np.zeros((self.config['maxastnodes'], self.config['maxastnodes']), dtype='int32')
                tmp[:wsmledges.shape[0], :wsmledges.shape[1]] = wsmledges
                wsmledges = np.int32(tmp)

            if with_smlpaths:
                g = nx.from_numpy_matrix(wsmledges)
                astpaths = nx.all_pairs_shortest_path(g, cutoff=self.config['pathlen'])
                wsmlpaths = list()

                for astpath in astpaths:
                    source = astpath[0]
                    
                    if len([n for n in g.neighbors(source)]) > 1:
                        continue
                    
                    for path in astpath[1].values():
                        if len([n for n in g.neighbors(path[-1])]) > 1:
                            continue # ensure only terminals as in Alon et al
                        
                        if len(path) > 1 and len(path) <= self.config['pathlen']:
                            newpath = self.idx2tok(wsmlnodes, path)
                            tmp = [0] * (self.config['pathlen'] - len(newpath))
                            newpath.extend(tmp)
                            wsmlpaths.append(newpath)
                
                random.shuffle(wsmlpaths) # Alon et al stipulate random selection of paths
                wsmlpaths = wsmlpaths[:self.config['maxpaths']] # Alon et al use 200, crop/expand to size
                if len(wsmlpaths) < self.config['maxpaths']:
                    wsmlpaths.extend([[0]*self.config['pathlen']] * (self.config['maxpaths'] - len(wsmlpaths)))
                wsmlpaths = np.asarray(wsmlpaths)

            if with_callnodes:
                wcallnodes = np.asarray(self.extradata['callnodedata'][fid])

                # cropping call chain nodes
                wcallnodes = wcallnodes[:self.config['maxcallnodes'],:self.config['tdatlen']]
                tmp3 = np.zeros((self.config['maxcallnodes'],self.config['tdatlen']), dtype='int32')
                tmp3[:wcallnodes.shape[0],:wcallnodes.shape[1]] = wcallnodes
                wcallnodes = np.int32(tmp3)


            if with_calledges:
                try:
                    wcalledges = self.extradata['calledges'][fid]
                    wcalledges = np.asarray(wcalledges.todense())
                except:
                    wcalledges = np.zeros((self.config['maxcallnodes'],self.config['maxcallnodes']), dtype=int)
                
                # cropping call chain edges
                wcalledges = wcalledges[:self.config['maxcallnodes'], :self.config['maxcallnodes']]
                tmp2 = np.zeros((self.config['maxcallnodes'], self.config['maxcallnodes']), dtype='int32')
                tmp2[:wcalledges.shape[0], :wcalledges.shape[1]] = wcalledges
                wcalledges = np.int32(tmp2)
                
            if with_bio:
                humanattnres = self.extradata['biodats'][fid]
                
                humanattnres = humanattnres[:wsmlnodeslen]
                
                whumanattn = list()
                for s in humanattnres:
                    whumanattn.append((s[0]+1)/2)
                
                whlen = len(whumanattn)
                
                # magnify differences
                whumanattn = [x*10 for x in whumanattn]

                # center values around 1 such that 1 is the average
                whavg = sum(whumanattn) / whlen
                whumanattn = [(x-whavg)+1 for x in whumanattn]
                
                # replace with randomly generated values for testing
                # run n times to get different random sets for comparison
                # this preserves reproducibility assuming seed is constant
                #whumanattn = [random.random()/10 for i in range(whlen)] # random1
                #whumanattn = [random.random()/10 for i in range(whlen)] # random2
                #whumanattn = [random.random()/10 for i in range(whlen)] # random3
                #whumanattn = [random.random()/10 for i in range(whlen)] # random4
                #whumanattn = [random.random()/10 for i in range(whlen)] # random5
                
                for i in range(self.config['maxastnodes'] - wsmlnodeslen):
                    whumanattn.append(1)
                 
                # softmax 
                #whumanattn = np.exp(whumanattn - np.max(whumanattn))
                #whumanattn = whumanattn / whumanattn.sum(axis=0)
                
                #print(whumanattn)
                #quit()
                
                whumanattn = np.asarray(whumanattn, dtype="float32")

            if not self.training:
                wcomseq = self.comstart
                inps = list()
                for inp in self.config['batch_config'][0]:
                    if inp == 'tdat':
                        inps.append(wtdatseq)
                    elif inp == 'tdat_sent':
                        inps.append(wtdatsentseq)
                    elif inp == 'sdat':
                        inps.append(wsdatseq)
                    elif inp == 'com':
                        inps.append(wcomseq)
                    elif inp == 'smlseq':
                        inps.append(wsmlseq)
                    elif inp == 'smlpath':
                        inps.append(wsmlpaths)
                    elif inp == 'smlnode':
                        inps.append(wsmlnodes)
                    elif inp == 'smledge':
                        inps.append(wsmledges)
                    elif inp == 'callnode':
                        inps.append(wcallnodes)
                    elif inp == 'calledge':
                        inps.append(wcalledges)
                    elif inp == 'bio':
                        inps.append(whumanattn)
                comseqs.append(wcomseq)
                fiddat[fid] = inps
            else:
                for i in range(0, len(wcomseq)):
                    if with_tdats:
                        tdatseqs.append(wtdatseq)
                    if with_tdat_sents:
                        tdatsentseqs.append(wtdatsentseq)
                    if with_sdats:
                        sdatseqs.append(wsdatseq)
                    if with_smlseqs:
                        smlseqs.append(wsmlseq)
                    if with_smlpaths:
                        smlpaths.append(wsmlpaths)
                    if with_smlnodes:
                        smlnodes.append(wsmlnodes)
                    if with_smledges:
                        smledges.append(wsmledges)
                    if with_callnodes:
                        callnodes.append(wcallnodes)
                    if with_calledges:
                        calledges.append(wcalledges)
                    if with_bio:
                        bios.append(whumanattn)
                        
                    # slice up whole comseq into seen sequence and current sequence
                    # [a b c d] => [] [a], [a] [b], [a b] [c], [a b c] [d], ...
                    comseq = wcomseq[0:i]
                    comout = wcomseq[i]
                    comout = keras.utils.to_categorical(comout, num_classes=self.comvocabsize)

                    # extend length of comseq to expected sequence size
                    # the model will be expecting all input vectors to have the same size
                    for j in range(0, len(wcomseq)):
                        try:
                            comseq[j]
                        except IndexError as ex:
                            comseq = np.append(comseq, 0)

                    comseqs.append(comseq)
                    comouts.append(np.asarray(comout))

        if self.training:
            if with_tdats:
                tdatseqs = np.asarray(tdatseqs)
            if with_tdat_sents:
                tdatsentseqs = np.asarray(tdatsentseqs)
            if with_sdats:
                sdatseqs = np.asarray(sdatseqs)
            if with_smlseqs:
                smlseqs = np.asarray(smlseqs)
            if with_smlpaths:
                smlpaths = np.asarray(smlpaths)
            if with_smlnodes:
                smlnodes = np.asarray(smlnodes)
            if with_smledges:
                smledges = np.asarray(smledges)
            if with_callnodes:
                callnodes = np.asarray(callnodes)
            if with_calledges:
                calledges = np.asarray(calledges)
            if with_bio:
                bios = np.asarray(bios)
                
            comseqs = np.asarray(comseqs)
            comouts = np.asarray(comouts)
            
            inps = list()
            oups = list()
            
            for inp in self.config['batch_config'][0]:
                if inp == 'tdat':
                    inps.append(tdatseqs)
                elif inp == 'tdat_sent':
                    inps.append(tdatsentseqs)
                elif inp == 'sdat':
                    inps.append(sdatseqs)
                elif inp == 'com':
                    inps.append(comseqs)
                elif inp == 'smlseq':
                    inps.append(smlseqs)
                elif inp == 'smlpath':
                    inps.append(smlpaths)
                elif inp == 'smlnode':
                    inps.append(smlnodes)
                elif inp == 'smledge':
                    inps.append(smledges)
                elif inp == 'callnode':
                    inps.append(callnodes)
                elif inp == 'calledge':
                    inps.append(calledges)
                elif inp == 'bio':
                    inps.append(bios)
        
            for oup in self.config['batch_config'][1]:
                if oup == 'comout':
                    oups.append(comouts)
                
            if len(oups) == 1:
                oups = oups[0]
        
        if not self.training:
            return (fiddat, badfids, comseqpos)
        else:
            return (inps, oups)

    def idx2tok(self, nodelist, path):
        out = list()
        for idx in path:
            out.append(nodelist[idx])
        return out
