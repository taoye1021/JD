#coding:utf-8
import pandas as pd
import numpy as np

'''
运行下面的getScore方法即可得出分数
'''
def getPrecision(pre,res):
    '''
    计算准确率
    '''
    F11_prec=0
    F12_prec=0
    for user,sku in zip(list(pre['user_id']),list(pre['sku_id'])):
        if user in list(res['user_id']):
            F11_prec+=1
        tmp=res[(res['user_id']==user)&(res['sku_id']==sku)]
        if tmp.empty==False:
            F12_prec+=len(tmp)

    F11_precision=F11_prec/float(len(pre))
    F12_precision=F12_prec/float(len(pre))
    print "F11_precision",F11_precision
    print "F12_precision",F12_precision
    return F11_precision,F12_precision


def getRecall(pre,res):
    '''
    计算召回率
    :param pre:
    :param res:
    :return:
    '''

    F12_TP=0
    F12_FN=0
    pre_user=set(pre['user_id'])
    res_user=set(res['user_id'])
    pre_sku=set(pre['sku_id'])
    res_sku=set(res['sku_id'])
    F11_TP=len((pre_user)&(res_user))
    F11_FN=len(res_user-pre_user)
    F11_rec=F11_TP/float(F11_TP+F11_FN)
    for user,sku in zip(list(pre['user_id']),list(pre['sku_id'])):
        tmp=res[(res['user_id']==user)&(res['sku_id']==sku)]
        if tmp.empty==False:
            F12_TP+=1
    F12_FN=len(res_sku-pre_sku)
    F12_rec=F12_TP/float((F12_TP+F12_FN))
    print "F11_rec",F11_rec
    print "F12_rec",F12_rec
    return F11_rec,F12_rec


##pre：预测的用户->商品表
##res:真实的用户->商品表
def getScore(pre,res):
    '''
    计算得分
    :param Reca:
    :param Prec:
    :return:
    '''
    F11_precision,F12_precision=getPrecision(pre,res)
    F11_rec,F12_rec=getRecall(pre,res)
    F11=6*F11_rec*F11_precision/(5*F11_rec+F11_precision)
    F12=5*F12_rec*F12_precision/(2*F12_rec+3*F12_precision)
    score=0.4*F11+0.6*F12
    print 'F11',F11
    print 'F12',F12
    print 'score',score


