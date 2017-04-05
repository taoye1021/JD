#coding:utf-8
import pandas as pd
import numpy as np
import datetime

##x到预测日的距离
def getTopred2(pred_day,x):
    n_p=datetime.datetime.strptime(pred_day,"%Y-%m-%d")
    n_d=datetime.datetime.strptime(x,'%Y/%m/%d')
    return (n_p-n_d).days

def getTopred(pred_day,x):
    n_p=datetime.datetime.strptime(pred_day,"%Y-%m-%d")
    n_d=datetime.datetime.strptime(x,'%Y-%m-%d')
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
    用户近三天的行为加权
'''
def getUserFeatures(all_cleanAction,user_info,start,end,pre_day):
    '''
    :param all_cleanAction: 清洗后的行为数据
    :param user_info: 用户表
    :param start: end: 训练集时间段
    :param pre_day: 预测起始日
    :return:
    '''
    user_info['user_reg_dt']=user_info.user_reg_dt.apply(lambda x:getTopred2(pre_day,x))
    ##提取用户性别，等级注册日期到预测日的距离特征，注意没有对类别型变量进行getdummies变换
    sub_info=user_info[['user_id','age','sex','user_lv_cd','user_reg_dt']]
    sub_info_users=list(sub_info['user_id'])
    ##提取start到end这个时间段内的用户行为数据
    sub_Action=all_cleanAction[(all_cleanAction['time']>=start)&(all_cleanAction['time']<=end)]
    sub_Action=pd.merge(sub_info,sub_Action,on='user_id',how='left')
    sub_Action=sub_Action[['user_id','sku_id','time','model_id','type','cate','brand']]
    #sub_Action.replace(np.nan,0,inplace=True)

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
        two_count=len(user_temp[user_temp['typeCount']>=2])
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
    active_days=sub_Action.copy()
    active_days['count']=1
    active_days=active_days.groupby(['user_id','time']).agg('sum').reset_index()
    active_days=active_days[['user_id']]
    active_days['active_days']=1
    active_days=active_days.groupby('user_id').agg('sum').reset_index()
    active_days=active_days[['user_id','active_days']]

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
    buy_toPre=buy_toPre[['user_id','time']]


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
    active_toPre=active_toPre[['user_id','time']]

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
    max_buy_toPre=active_toPre[['user_id','time']]


    ##用户浏览过的商品中发生购买的比值
    t3=sub_Action[['user_id','sku_id','type']]
    t3['typeCounts']=1
    t3=t3.groupby(['user_id','sku_id','type']).agg('sum').reset_index()
    user_sku_reads=t3[t3['type']==1]
    user_sku_reads=user_sku_reads[['user_id','sku_id']]
    read_users=set(user_sku_reads['user_id'])
    user_sku_buys=t3[t3['type']==4]
    user_sku_buys=user_sku_buys[['user_id','sku_id']]
    buy_user=set(user_sku_buys['user_id'])
    all_users=(read_users)|(buy_user)
    buyIn_reads_ratio=pd.DataFrame(columns=['user_id','ratio'])
    i=-1
    for user in list(all_users):
        i+=1
        if user in list(read_users):
            if user in list(buy_user):
                read_df=user_sku_reads[user_sku_reads['user_id']==user]
                read_items=set(read_df['sku_id'])
                buy_df=user_sku_buys[user_sku_buys['user_id']==user]
                buy_items=set(buy_df['sku_id'])
                items=len((read_items)&(buy_items))
                ratio=items/float(len(read_items))
                buyIn_reads_ratio.loc[i]=[user,ratio]
            else:
                buyIn_reads_ratio.loc[i]=[user,0]
        else:
            buyIn_reads_ratio.loc[i]=[user,-1]
    buyIn_reads_ratio=buyIn_reads_ratio[['user_id','ratio']]




    ##用户关注过的商品中发生购买的比值
    t4=sub_Action[['user_id','sku_id','type']]
    t4['typeCounts']=1
    t4=t4.groupby(['user_id','sku_id','type']).agg('sum').reset_index()
    user_sku_attention=t4[t4['type']==5]
    user_sku_attention=user_sku_attention[['user_id','sku_id']]
    attention_users=set(user_sku_attention['user_id'])
    user_sku_buys=t4[t4['type']==4]
    user_sku_buys=user_sku_buys[['user_id','sku_id']]
    buy_user=set(user_sku_buys['user_id'])
    all_users=(attention_users)|(buy_user)
    buyIn_attention_ratio=pd.DataFrame(columns=['user_id','ratio'])
    i=-1
    for user in list(all_users):
        i+=1
        if user in list(attention_users):
            if user in list(buy_user):
                attention_df=user_sku_attention[user_sku_attention['user_id']==user]
                attention_items=set(attention_df['sku_id'])
                buy_df=user_sku_buys[user_sku_buys['user_id']==user]
                buy_items=set(buy_df['sku_id'])
                items=len((attention_items)&(buy_items))
                ratio=items/float(len(attention_items))
                buyIn_attention_ratio.loc[i]=[user,ratio]
            else:
                buyIn_attention_ratio.loc[i]=[user,0]
        else:
            buyIn_attention_ratio.loc[i]=[user,-1]
    buyIn_attention_ratio=buyIn_attention_ratio[['user_id','ratio']]

    ##用户加入购物车中下单的比值
    t5=sub_Action[['user_id','sku_id','type']]
    t5['typeCounts']=1
    t5=t5.groupby(['user_id','sku_id','type']).agg('sum').reset_index()
    user_sku_addshop=t5[t5['type']==2]
    user_sku_addshop=user_sku_addshop[['user_id','sku_id']]
    addshop_users=set(user_sku_addshop['user_id'])
    user_sku_buys=t5[t5['type']==4]
    user_sku_buys=user_sku_buys[['user_id','sku_id']]
    buy_user=set(user_sku_buys['user_id'])
    all_users=(addshop_users)|(buy_user)
    buyIn_addshop_ratio=pd.DataFrame(columns=['user_id','ratio'])
    i=-1
    for user in list(all_users):
        i+=1
        if user in list(addshop_users):
            if user in list(buy_user):
                addshop_df=user_sku_addshop[user_sku_addshop['user_id']==user]
                addshop_items=set(addshop_df['sku_id'])
                buy_df=user_sku_buys[user_sku_buys['user_id']==user]
                buy_items=set(buy_df['sku_id'])
                items=len((addshop_items)&(buy_items))
                ratio=items/float(len(addshop_items))
                buyIn_addshop_ratio.loc[i]=[user,ratio]
            else:
                buyIn_addshop_ratio.loc[i]=[user,0]
        else:
            buyIn_addshop_ratio.loc[i]=[user,-1]
    buyIn_addshop_ratio=buyIn_addshop_ratio[['user_id','ratio']]


    ##用户近三天的行为加权,统计每个用户在近三天的每一天的活跃次数，并且越靠近预测日，权值越大
    Thre_days=datetime.timedelta(days=2)
    one_day=datetime.timedelta(days=1)
    endDay=datetime.datetime.strptime(end,'%Y-%m-%d')
    endDay=endDay-Thre_days
    end=endDay.strftime('%Y-%m-%d')
    Action_end_3=all_cleanAction[(all_cleanAction['time']==end)]
    Action_end_3['count_3']=1
    Action_end_3=Action_end_3.groupby(['user_id']).agg('sum').reset_index()
    Action_end_3=Action_end_3[['user_id','count_3']]
    endDay=endDay+one_day
    end=endDay.strftime('%Y-%m-%d')
    Action_end_2=all_cleanAction[(all_cleanAction['time']==end)]
    Action_end_2['count_2']=1
    Action_end_2=Action_end_2.groupby(['user_id']).agg('sum').reset_index()
    Action_end_2=Action_end_2[['user_id','count_2']]
    endDay=endDay+one_day
    end=endDay.strftime('%Y-%m-%d')
    Action_end_1=all_cleanAction[(all_cleanAction['time']==end)]
    Action_end_1['count_1']=1
    Action_end_1=Action_end_1.groupby(['user_id']).agg('sum').reset_index()
    Action_end_1=Action_end_1[['user_id','count_1']]
    ##前一天的行为权值5，前第两天3，前第三天1
    all_users=set(sub_Action['user_id'])
    user_Thre_active=pd.DataFrame(columns=['user_id','weight'])
    i=-1
    for user in list(all_users):
        i+=1
        a=int(list(Action_end_3[Action_end_3['user_id']==user]['count_3'])[0])*5
        b=int(list(Action_end_2[Action_end_2['user_id']==user]['count_2'])[0])*2
        c=int(list(Action_end_1[Action_end_1['user_id']==user]['count_3'])[0])
        weight=sum(a,b,c)
        user_Thre_active.loc[i]=[user,weight]
    user_Thre_active=user_Thre_active[['user_id','weight']]

    ##下面将以上用户一系列特征融合成一张表

























