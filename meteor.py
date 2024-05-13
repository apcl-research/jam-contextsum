import sys
import pickle
import argparse
import re
import os
import collections

import numpy as np

from nltk.translate.meteor_score import meteor_score

from myutils import prep, drop, statusout, batch_gen, seq2sent, index2word
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def corpus_meteor(expected, predicted):

    scores = list()
    
    for e, p in zip(expected, predicted):
        #e = [' '.join(x) for x in e]
        #p = ' '.join(p)
        m = meteor_score(e, p)
        #if(m>0.10 and m<0.70):
        scores.append(m)
        
    return scores, np.mean(scores)
    

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.asarray(x)
    x = x.astype(np.float)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def meteor_so_far_m_only(refs, preds):
    scores, m = corpus_meteor(refs, preds)
    m = round(m*100, 2)
    return m

def meteor_so_far(refs, preds):
    
    scores, m = corpus_meteor(refs, preds)
    m = round(m*100, 2)
    
    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('M %s\n' % (m))
    #return scores, m, ret
    return ret

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='./data/jam_cgpt_170k')  
    parser.add_argument('--outdir', dest='outdir', type=str, default='outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    parser.add_argument('--lim-overlap', dest='limoverlap', type=int, default=-1)
    parser.add_argument('--lim-overlap-sdats', dest='limoverlapsdats', type=int, default=-1)
    parser.add_argument('--tdats-filename', dest='tdatsfilename', type=str, default='tdats.test')
    parser.add_argument('--sdats-filename', dest='sdatsfilename', type=str, default='sdats.test')
    parser.add_argument('--coms-filename', dest='comsfilename', type=str, default='cgptcom.test')
    parser.add_argument('--sentence-bleus', dest='sentencebleus', action='store_true', default=False)
    parser.add_argument('--delim', dest='delim', type=str, default='<SEP>')
    
    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    input_file = args.input
    lim_overlap = args.limoverlap
    lim_overlap_sdats = args.limoverlapsdats
    tdatsfilename = args.tdatsfilename
    sdatsfilename = args.sdatsfilename
    comsfilename = args.comsfilename
    sentencebleus = args.sentencebleus
    delim = args.delim

    if input_file is None:
        print('Please provide an input file to test')
        exit()

    if lim_overlap != -1 or lim_overlap_sdats != -1:
        prep('preparing tdats list... ')
        tdats = dict()
        tdatsf = open('%s/%s' % (dataprep, tdatsfilename), 'r')
        for c, line in enumerate(tdatsf):
            try:
                (fid, tdat) = line.split(delim)
                try:
                    fid = int(fid)
                except ValueError:
                    pass
                tdat = tdat.split()
                tdat = fil(tdat)
                tdats[fid] = tdat
            except:
                continue
        tdatsf.close()
        drop()

    if lim_overlap_sdats != -1:
        prep('preparing sdats list... ')
        sdats = dict()
        sdatsf = open('%s/%s' % (dataprep, sdatsfilename), 'r')
        for c, line in enumerate(sdatsf):
            (fid, sdat) = line.split(delim)
            fid = int(fid)
            sdat = sdat.split()
            sdat = fil(sdat)
            sdats[fid] = sdat
        sdatsf.close()
        drop()

    prep('preparing predictions list... ')
    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        #try:
        split_line = line.split('\t')
        (fid, pred) = split_line[0], split_line[-1]
        try:
            fid = int(fid)
        except:
            pass
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = pred
       # except:
       #     continue
    predicts.close()
    drop()
    re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])')

    if(sentencebleus):
        bfn = os.path.basename(input_file)
        bfn = os.path.splitext(bfn)[0]
        bleusf = open('{}/bleus/{}.tsv'.format(outdir, bfn), 'w')

    refs = list()
    newpreds = list()
    d = 0
    targets = open('%s/%s' % (dataprep, comsfilename), 'r')
    for line in targets:
        (fid, com) = line.split(delim)
        try:
            fid = int(fid)
        except ValueError:
            pass
        com = com.split()
        com = fil(com)
        
        if len(com) < 1:
            continue


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

            if not(o_s > lim_overlap_sdats):
            #if not(o_s == lim_overlap_sdats):
                continue


        if lim_overlap != -1:
            t = list(set(com) & set(tdats[fid][:12]))
            overlap = len(t) #/ len(set(com))
            
            if overlap != lim_overlap:
                continue

        try:
            newpreds.append(preds[fid])
            #print(fid, preds[fid])
            
            if(sentencebleus):
                
                Bas = corpus_bleu([[com]], [preds[fid]])
                B1s = corpus_bleu([[com]], [preds[fid]], weights=(1,0,0,0))
                B2s = corpus_bleu([[com]], [preds[fid]], weights=(0,1,0,0))
                B3s = corpus_bleu([[com]], [preds[fid]], weights=(0,0,1,0))
                B4s = corpus_bleu([[com]], [preds[fid]], weights=(0,0,0,1))

                Bas = round(Bas * 100, 4)
                B1s = round(B1s * 100, 4)
                B2s = round(B2s * 100, 4)
                B3s = round(B3s * 100, 4)
                B4s = round(B4s * 100, 4)
                
                bleusf.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(fid, Bas, B1s, B2s, B3s, B4s))
            
        except Exception as ex:
            #newpreds.append([])
            continue

        refs.append([com])
    
    if(sentencebleus):
        bleusf.close()

    print('final status')
    print(meteor_so_far(refs, newpreds))

