#coding:utf-8
import pandas as pd
import numpy as np

##读取清洗后的4月的行为数据
##读取清洗后的商品集合
Action04=pd.read_csv('../cleanData/clean_Action04.csv')
products=pd.read_csv('../cleanData/clean_products.csv')

def getPreItems(products,start,end):
    '''
    读取行为数据中start到end时间段内的加入购物车的用户和商品集合，并且在此基础上除去这段时间内已经下单和从购物车删除的用户商品
    再与商品P集合取交集
    返回用户，商品集合
    '''
    prod=list(products['sku_id'])
    Action=Action04[(Action04['time']>=start)&(Action04['time']<=end)]
    Action_addshop=Action[Action['type']==2]
    Action_addshop_sku=set(Action[Action['type']==2]['sku_id'])
    Action_delshop_sku=set(Action[Action['type']==3]['sku_id'])
    Action_buy_sku=set(Action[Action['type']==4]['sku_id'])
    Action_addshop_sku=list(Action_addshop_sku-Action_delshop_sku-Action_buy_sku)
    Action_addshop=Action_addshop[Action_addshop['sku_id'].isin (Action_addshop_sku)]
    Action_addshop=Action_addshop[Action_addshop['sku_id'].isin (prod)]
    pre=Action_addshop[['user_id','sku_id']]
    pre.sort_values(by='user_id',inplace=True)

    return pre


def pressPre(pre,start,end):
    '''
    上面方法所得的预测pre集合，可能存在一个用户购买多个商品，对于这种情况，一个用户对应多个商品取其点击次数最多的那个商品作为预测商品
    调用下面的getSortClickItem方法
    '''
    pred=dict()
    for i in range(len(pre)):
        user=list(pre.iloc[i])[0]
        sku=list(pre.iloc[i])[1]
        if user not in pred:
            pred[user]=set()
        pred[user].add(sku)

    pred2=dict()
    for user in pred:
        if len(list(pred[user]))==1:
            pred2.update({user:list(pred[user])[0]})

        else:
            pred2.update({user:getSortClickItem(list(pred[user]),start,end)})
    new_pre=pd.DataFrame()
    new_pre.insert(0,'user_id',pred2.keys())
    new_pre.insert(1,'sku_id',pred2.values())
    return new_pre


def getSortClickItem(items,start,end):
    '''
    items是多个商品的列表，以商品id作为字典建值，以在start到end时间段，各个商品点击次数作为values值，对其进行排序
    返回点击次数最多的那个商品
    '''
    Action=Action04[(Action04['time']>=start)&(Action04['time']<=end)]
    Action_click=list(Action[Action['type']==5]['sku_id'])
    occur_freq=dict()
    for item in items:
        occur_freq.update({item:Action_click.count(item)})
    lst=sorted(occur_freq.items(),key=lambda x:x[1],reverse=True)
    return lst[0][0]


def getResItems(products,start,end):
    '''
    取start到end这个时间段内真正下单的用户商品集合，并且与商品P集合作交集
    '''
    prod=list(products['sku_id'])
    Action=Action04[(Action04['time']>=start)&(Action04['time']<=end)]
    buy_items=Action[Action['type']==4]
    res=buy_items[['user_id','sku_id']]
    res=res[res['sku_id'].isin (prod)]
    res.sort_values(by='user_id',inplace=True)
    res=res.drop_duplicates('user_id')
    return res


if __name__=='__main__':
    '''
    线下测试，以4月8到4月10这三天作为训练集，以4月11到4月15这几天真实下单数据作为其label
    '''
    pre=getPreItems(products,'2016-04-08','2016-04-10')
    new_pre=pressPre(pre,'2016-04-08','2016-04-10')
    print len(new_pre['user_id'])==len(set(new_pre['user_id']))
    res=getResItems(products,'2016-04-11','2016-04-15')
    new_pre.to_csv('pred.csv',index=None)
    res.to_csv('resu.csv',index=None)
