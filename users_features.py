#coding:utf-8
import pandas as pd
import numpy as np
import datetime

##x到预测日的距离
def getTopred(pred_day,x):
    n_p=datetime.datetime.strptime(pred_day,"%Y-%m-%d")
    n_d=datetime.datetime.strptime(x,'%Y/%m/%d')
    return (n_p-n_d).days

##获取用户特征
'''
 用户等级分类
    注册日期距离预测日的距离
    用户浏览量
    用户关注量
    用户加入购物车量
    用户下单量
    用户活跃天数
    用户上次下单距离预测日时长
    用户上次活跃距离预测日时长
    某天最大下单量距离预测日的时长
    用户购买量与浏览量的比值
    用户发生二次购买商品数占总购买商品数的比值。
    用户浏览过的商品中发生购买的比值
    用户下单量与关注量的比值
    用户关注的商品中发生购买的比值
    用户加入购物车中下单的比值
'''
def getUserFeatures(all_cleanAction,user_info,start,end,pre_day):
    '''
    :param all_cleanAction: 清洗后的行为数据
    :param user_info: 用户表
    :param start: end: 训练集时间段
    :param pre_day: 预测起始日
    :return:
    '''
    user_info['user_reg_dt']=user_info.user_reg_dt.apply(lambda x:getTopred(pre_day,x))
    ##提取用户性别，等级注册日期到预测日的距离特征，注意没有对类别型变量进行getdummies变换
    sub_user=user_info[['user_id','age','sex','user_lv_cd','user_reg_dt']]

    ##提取start到end这个时间段内的用户行为数据
    sub_Action=all_cleanAction[(all_cleanAction['time']>=start)&(all_cleanAction['time']<=end)]

    ##统计每个用户对每个行为的次数
    t1=sub_Action[['user_id','type']]
    t1['typeCounts']=1
    t1=t1.groupby(['user_id','type']).agg('sum').reset_index()
    temp=t1[['user_id','type','typeCounts']]

    ##用户浏览量
    user_reads=temp[temp['type']==1]
    user_reads=user_reads[['user_id','typeCounts']]
    ##用户下单量
    user_buys=temp[temp['type']==4]
    user_buys=user_buys[['user_id','typeCounts']]

    ##用户发生二次购买商品数占总的购买商品数的比值
    ##这里只能统计购买的人的比值，后期合并时对于这个时间段没发生购买的用户比值用0代替？？
    t2=sub_Action[['user_id','sku_id','type']]
    t2['typeCounts']=1
    t2=t2.groupby(['user_id','sku_id','type']).agg('sum').reset_index()
    temp=t2[t2['type']==4]
    ratio=[]
    for i in range(len(temp)):
        user=list(temp.iloc[i])[0]
        user_temp=temp[temp['user_id']==user]
        two_count=user_temp[user_temp['typeCount']>=2]
        item_counts=len(user_temp)
        rat=two_count/float(item_counts)
        ratio.append(rat)
    temp.insert(4,'two_ratio',ratio)
    two_buyitem_ratio=temp[['user_id','two_ratio']]
    two_buyitem_ratio=two_buyitem_ratio.drop_duplicates(['user_id'])



    ##用户购买量和浏览量的比值
    user_reads_buys=pd.merge(user_reads,user_buys,on='user_id',how='outer')
    user_reads_buys.replace(np.nan,0,inplace=True)
    user_reads_buys['ratio']=user_reads_buys['typeCounts_y']/user_reads_buys['typeCounts_x']

    ##浏览量为0时，比值结果为inf用-1代替
    user_reads_buys.replace(np.inf,-1,inplace=True)
    user_reads_buys=user_reads_buys[['user_id','ratio']]

    ##用户关注量
    user_attention=temp[temp['type']==5]
    user_attention=user_attention[['user_id','typeCounts']]

    ##用户购买量和关注量的比值
    user_attention_buys=pd.merge(user_attention,user_buys,on='user_id',how='outer')
    user_attention_buys.replace(np.nan,0,inplace=True)
    user_attention_buys['ratio']=user_attention_buys['typeCounts_y']/user_reads_buys['typeCounts_x']
    ##浏览量为0时，比值结果为inf用-1代替
    user_attention_buys.replace(np.inf,-1,inplace=True)
    user_attention_buys=user_attention_buys[['user_id','ratio']]




    ##用户加入购物车的量
    user_addshop=temp[temp['type']==2]
    user_addshop=user_addshop[['user_id','typeCounts']]
    ##用户购物车删除的量
    user_delshop=temp[temp['type']==3]
    user_delshop=user_delshop[['user_id','typeCounts']]


    ###计算用户活跃天数
    t2=sub_Action.copy()
    t2['count']=1
    t2=t2.groupby(['user_id','time']).agg('sum').reset_index()
    t2=t2[['user_id']]
    t2['active_days']=1
    t2=t2.groupby('user_id').agg('sum').reset_index()

    ##用户上次下单距离预测日的时长
    buy_allUsers=all_cleanAction[(all_cleanAction['type']==4)&(all_cleanAction['time']<=pre_day)]
    buy_allUsers=buy_allUsers[['user_id','time']]
    buy_allUsers['count']=1
    buy_allUsers=buy_allUsers.groupby(['user_id','time']).agg('sum').reset_index()
    buy_allUsers=buy_allUsers.drop_duplicates(['user_id'],keep='last')
    buy_allUsers.drop('count',axis=1,inplace=True)
    buy_allUsers['time']=buy_allUsers.time.apply(lambda x:getTopred(pre_day,x))
    sub_users=list(sub_Action['user_id'])
    buy_toPre=buy_allUsers[buy_allUsers['user_id'].isin (sub_users)]


    ##用户上次活跃距离预测日的时长
    active_allUsers=all_cleanAction[(all_cleanAction['time']<=pre_day)]
    active_allUsers=active_allUsers[['user_id','time']]
    active_allUsers['count']=1
    active_allUsers=active_allUsers.groupby(['user_id','time']).agg('sum').reset_index()
    active_allUsers=active_allUsers.drop_duplicates(['user_id'],keep='last')
    active_allUsers.drop('count',axis=1,inplace=True)
    active_allUsers['time']=active_allUsers.time.apply(lambda x:getTopred(pre_day,x))
    sub_users=list(sub_Action['user_id'])
    active_toPre=active_allUsers[active_allUsers['user_id'].isin (sub_users)]

    ##用户最大下单量那天距离预测日的时长
    buy_allUsers=all_cleanAction[(all_cleanAction['type']==4)&(all_cleanAction['time']<=pre_day)]
    buy_allUsers=buy_allUsers[['user_id','time']]
    buy_allUsers['count']=1
    buy_allUsers=buy_allUsers.groupby(['user_id','time']).agg('sum').reset_index()
    #取用户最大下单量，如果下单量都一样，则取最晚的那天
    buy_allUsers.sort_values(by=['user_id','count','time'],inplace=True)
    buy_allUsers=buy_allUsers.drop_duplicates(['user_id'],keep='last')
    buy_allUsers.drop('count',axis=1,inplace=True)
    buy_allUsers['time']=buy_allUsers.time.apply(lambda x:getTopred(pre_day,x))
    sub_users=list(sub_Action['user_id'])
    max_buy_toPre=buy_allUsers[buy_allUsers['user_id'].isin (sub_users)]


    ##用户浏览过的商品中发生购买的比值
    t3=sub_Action[['user_id','sku_id','type']]
    t3['typeCounts']=1
    t3=t3.groupby(['user_id','sku_id','type']).agg('sum').reset_index()
    user_sku_reads=t3[t3['type']==1]
    user_sku_reads=user_sku_reads[['user_id','sku_id','typeCounts']]
    user_sku_buys=t3[t3['type']==4]
    user_sku_buys=user_sku_buys[['user_id','sku_id','typeCounts']]
    user_sku_reads_buys=pd.merge(user_sku_reads,user_sku_buys,on=['user_id','sku_id'],how='outer')
    user_sku_reads_buys.replace(np.nan,0,inplace=True)
    user_sku_reads_buys['ratio']=user_sku_reads_buys['typeCounts_y']/user_sku_reads_buys['typeCounts_x']
    user_sku_reads_buys.replace(np.nan,0,inplace=True)
    user_sku_reads_buys=user_sku_reads_buys[['user_id','ratio']]

    ##用户关注过的商品中发生购买的比值
    t4=sub_Action[['user_id','sku_id','type']]
    t4['typeCounts']=1
    t4=t4.groupby(['user_id','sku_id','type']).agg('sum').reset_index()
    user_sku_attention=t4[t4['type']==5]
    user_sku_attention=user_sku_attention[['user_id','sku_id','typeCounts']]
    user_sku_buys=t4[t4['type']==4]
    user_sku_buys=user_sku_buys[['user_id','sku_id','typeCounts']]
    user_sku_attention_buys=pd.merge(user_sku_attention,user_sku_buys,on=['user_id','sku_id'],how='outer')
    user_sku_attention_buys.replace(np.nan,0,inplace=True)
    user_sku_attention_buys['ratio']=user_sku_attention['typeCounts_y']/user_sku_attention_buys['typeCounts_x']
    user_sku_attention_buys.replace(np.nan,0,inplace=True)
    user_sku_attention_buys=user_sku_attention_buys[['user_id','ratio']]

    ##用户加入购物车中下单的比值
    t5=sub_Action[['user_id','sku_id','type']]
    t5['typeCounts']=1
    t5=t5.groupby(['user_id','sku_id','type']).agg('sum').reset_index()
    user_sku_addshop=t5[t5['type']==2]
    user_sku_addshop=user_sku_addshop[['user_id','sku_id','typeCounts']]
    user_sku_buys=t5[t5['type']==4]
    user_sku_buys=user_sku_buys[['user_id','sku_id','typeCounts']]
    user_sku_addshop_buys=pd.merge(user_sku_addshop,user_sku_buys,on=['user_id','sku_id'],how='outer')
    user_sku_addshop_buys.replace(np.nan,0,inplace=True)
    user_sku_addshop_buys['ratio']=user_sku_addshop_buys['typeCounts_y']/user_sku_addshop_buys['typeCounts_x']
    user_sku_addshop_buys.replace(np.nan,0,inplace=True)
    user_sku_addshop_buys=user_sku_addshop_buys[['user_id','ratio']]


















