#coding:utf-8
import pandas as pd
import numpy as np

def getFile():
    '''
    加载读取商品子集P，预测集合pre和真实集合res并返回
    '''
    products=pd.read_csv('../cleanData/clean_products.csv')
    pre=pd.read_csv('pred.csv')
    res=pd.read_csv('resu.csv')
    return products,pre,res


def getPrecision(pre,res):
    '''
    计算准确率

    '''
    prec=0
    for user,sku in zip(list(pre['user_id']),list(pre['sku_id'])):
        tmp=res[(res['user_id']==user)&(res['sku_id']==sku)]
        if tmp.empty==False:
            prec+=len(tmp)
    precision=prec/float(len(pre))
    print "precision",precision
    return precision


def getRecall(pre,res):
    '''
    计算召回率
    :param pre:
    :param res:
    :return:
    '''
    rec=0
    TP=0
    FN=0
    pre_sku=set(pre['sku_id'])
    res_sku=set(res['sku_id'])
    for user,sku in zip(list(pre['user_id']),list(pre['sku_id'])):
        tmp=res[(res['user_id']==user)&(res['sku_id']==sku)]
        if tmp.empty==False:
            TP+=1
    FN=len(res_sku-pre_sku)
    rec=TP/float((TP+FN))
    print "rec",rec
    return rec


def getScore(Reca,Prec):
    '''
    计算得分
    :param Reca:
    :param Prec:
    :return:
    '''
    F11=6*Reca*Prec/(5*Reca+Prec)
    F12=5*Reca*Prec/(2*Reca+3*Prec)
    score=0.4*F11+0.6*F12
    return F11,F12,score


if __name__=='__main__':
    products,pre,res=getFile()
    Prec=getPrecision(pre,res)
    Reca=getRecall(pre,res)
    F11,F12,score=getScore(Reca,Prec)
    print 'F11',F11
    print 'F12',F12
    print 'score',score

