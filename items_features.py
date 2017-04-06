#coding:utf-8
import pandas as pd
import numpy as np
import datetime
'''
统计商品一系列特征
    商品累计评论数
    商品是否有差评
    商品差评率
    品牌相关特征
    商品被购买次数
    商品被关注次数
    商品购买人数与关注人数的比值
    商品关注人数中发生购买的比值
    商品被点击次数
    商品购买人数与点击人数的比值
    商品点击人数中发生购买的比值
    商品被加入购买的次数
    商品被加入购物车的人数与购买人数的比值
    商品被加入购物车的人中发生购买的比值
    商品被浏览的次数
    商品被购买与被浏览次数的比值
    商品被浏览人数中发生购买的比值
    购买商品的人中发生二次购买的比值
    商品最后一次购买距离预测日的距离
    商品最后一次与用户发生交互距离预测日的距离
    商品最大购买量距离预测日时长
    该商品的交互量占该品类商品的交互量比值
    该商品的交互量占该品牌商品的交互量比值
    该商品的购买量占该品类商品购买量的比值
    该商品的购买量占该品牌商品购买量的比值
'''

def getTopred(pred_day,x):
    n_p=datetime.datetime.strptime(pred_day,"%Y-%m-%d")
    n_d=datetime.datetime.strptime(x,'%Y-%m-%d')
    return (n_p-n_d).days

def getItemsFeatures(clean_allActions,clean_products,comment,pre_day):
    ##商品子集中 sku_id,a1,a2,a3,品类，品牌，特征
    clean_products=clean_products[['sku_id','a1','a2','a3','cate','brand']]
    all_items=clean_products[['sku_id']]

    ##商品子集中商品 评价特征，截止日期距离预测日的距离，累计评分数，是否有差评，差评率
    comment['dt']=comment.dt.apply(lambda x:getTopred(pre_day,x))
    comment.raname(columns={'dt':'getTopre_days'},inplace=True)
    comment=pd.merge(all_items,comment,on='sku_id',how='left')

    ##商品子集中商品与各种交互行为的次数
    sub_items=clean_allActions[['sku_id','type']]
    sub_items['typeCounts']=1
    sub_items=sub_items.groupby(['sku_id','type']).agg('sum').reset_index()
    sub_items=pd.merge(all_items,sub_items,on='sku_id',how='left')
    sub_items.replace(np.nan,0,inplace=True)

    ##商品子集中商品，用户与对应的各种行为的次数
    sub_item_users=clean_allActions[['sku_id','user_id','type']]
    sub_item_users['user_typeCounts']=1
    sub_item_users=sub_item_users.groupby(['sku_id','user_id','type']).agg('sum').reset_index()
    sub_item_users=pd.merge(all_items,sub_item_users,on='sku_id',how='left')
    ##此处可能存在某些商品没有用户对其有交互行为，故此时填充在下面视具体情况填充

    ##商品子集中，商品，时间与对应交互行为的次数
    sub_item_time=clean_allActions[['sku_id','time','type']]
    sub_item_time['time_typeCounts']=1
    sub_item_time=sub_item_time.groupby(['sku_id','time','type']).agg('sum').reset_index()
    sub_item_time=pd.merge(all_items,sub_item_time,on='sku_id',how='left')




    ##商品子集中商品被购买次数
    item_buys=sub_items[sub_items['type']==4][['sku_id','typeCounts']]
    item_buys.rename(columns={'typeCounts':'item_buys'},inplace=True)
    item_buys=pd.merge(all_items,item_buys,on='sku_id',how='left')
    item_buys.replace(np.nan,0,inplace=True)

    ##商品子集中商品被关注次数
    item_attentions=sub_items[sub_items['type']==5][['sku_id','typeCounts']]
    item_attentions.rename(columns={'typeCounts':'item_attentions'},inplace=True)
    item_buys=pd.merge(all_items,item_buys,on='sku_id',how='left')
    item_attentions.replace(np.nan,0,inplace=True)

    ##商品子集中商品购买人数占总关注人数的比值
    items_buy_attention=pd.merge(item_buys,item_attentions,on='sku_id',how='right')
    items_buy_attention.replace(np.nan,0,inplace=True)
    items_buy_attention['buy_attention_ratio']=items_buy_attention['item_buys']/items_buy_attention['item_attentions']
    items_buy_attention=items_buy_attention[['sku_id','buy_attention_ratio']]
    items_buy_attention=pd.merge(all_items,items_buy_attention,on='sku_id',how='left')
    ##关注人数为0时也即是分母为0时，用-1代替
    items_buy_attention.replace(np.inf,-1,inplace=True)

    ##商品关注人数中发生购买的比值
    item_user_attentions=sub_item_users[sub_item_users['type']==5]
    item_user_buys=sub_item_users[sub_item_users['type']==4]
    items_buyIn_attentions=pd.DataFrame(columns=['sku_id','buyIn_attentions_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        if item in list(item_user_attentions['sku_id']):
            if item in list(item_user_buys['sku_id']):
                attention_users=set(item_user_attentions[item_user_attentions['sku_id']==item]['user_id'])
                buy_users=set(item_user_buys[item_user_buys['sku_id']==item]['user_id'])
                buy_attention_users=attention_users&buy_users
                ratio=len(buy_attention_users)/float(len(attention_users))
                items_buyIn_attentions.loc[i]=[item,ratio]
            else:
                items_buyIn_attentions.loc[i]=[item,0]

        items_buyIn_attentions.loc[i]=[item,-1]




    ##商品子集中商品被点击次数
    item_clicks=sub_items[sub_items['type']==6][['sku_id','typeCounts']]
    item_clicks.rename(columns={'typeCounts':'item_clicks'},inplace=True)
    item_clicks=pd.merge(all_items,item_clicks,on='sku_id',how='left')
    item_clicks.replace(np.nan,0,inplace=True)


    ##商品子集中商品被点击的次数和被下单次数的比值
    item_buy_clicks=pd.merge(item_buys,item_clicks,on='sku_id',how='right')
    item_buy_clicks.replace(np.nan,0,inplace=True)
    item_buy_clicks['buy_clicks_ratio']=item_buy_clicks['item_buys']/item_buy_clicks['item_clicks']
    item_buy_clicks=item_buy_clicks[['sku_id','buy_clicks_ratio']]
    item_buy_clicks=pd.merge(all_items,item_buy_clicks,on='sku_id',how='left')
    item_buy_clicks.replace(np.inf,-1,inplace=True)


    ##商品点击人数中发生购买的比值
    item_user_clicks=sub_item_users[sub_item_users['type']==6]
    item_user_buys=sub_item_users[sub_item_users['type']==4]
    items_buyIn_clicks=pd.DataFrame(columns=['sku_id','buyIn_clicks_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        if item in list(item_user_clicks['sku_id']):
            if item in list(item_user_buys['sku_id']):
                click_users=set(item_user_clicks[item_user_clicks['sku_id']==item]['user_id'])
                buy_users=set(item_user_buys[item_user_buys['sku_id']==item]['user_id'])
                buy_click_users=click_users&buy_users
                ratio=len(buy_click_users)/float(len(click_users))
                items_buyIn_clicks.loc[i]=[item,ratio]
            else:
                items_buyIn_clicks.loc[i]=[item,0]

        items_buyIn_clicks.loc[i]=[item,-1]


    ##商品子集中商品被加入购买的次数
    item_addshop=sub_items[sub_items['type']==2][['sku_id','typeCounts']]
    item_addshop.rename(columns={'typeCounts':'item_addshops'},inplace=True)
    item_addshop=pd.merge(all_items,item_addshop,on='sku_id',how='left')
    item_addshop.replace(np.nan,0,inplace=True)

    ##商品子集中商品被购买与被加入购物车的比值
    item_buy_addshop=pd.merge(item_buys,item_addshop,on='sku_id',how='right')
    item_buy_addshop.replace(np.nan,0,inplace=True)
    item_buy_addshop['buy_addshop_ratio']=item_buy_addshop['item_buys']/item_buy_addshop['item_addshops']
    item_buy_addshop=item_buy_addshop[['sku_id','buy_addshop_ratio']]
    item_buy_addshop=pd.merge(all_items,item_buy_addshop,on='sku_id',how='left')
    item_buy_addshop.replace(np.inf,-1,inplace=True)


    ##商品被加入购物车的人中发生购买的比值
    item_user_addshops=sub_item_users[sub_item_users['type']==2]
    item_user_buys=sub_item_users[sub_item_users['type']==4]
    items_buyIn_addshops=pd.DataFrame(columns=['sku_id','buyIn_addshops_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        if item in list(item_user_addshops['sku_id']):
            if item in list(item_user_buys['sku_id']):
                addshop_users=set(item_user_addshops[item_user_addshops['sku_id']==item]['user_id'])
                buy_users=set(item_user_buys[item_user_buys['sku_id']==item]['user_id'])
                buy_addshop_users=addshop_users&buy_users
                ratio=len(buy_addshop_users)/float(len(addshop_users))
                items_buyIn_addshops.loc[i]=[item,ratio]
            else:
                items_buyIn_addshops.loc[i]=[item,0]

        items_buyIn_addshops.loc[i]=[item,-1]


    ##商品子集中商品被浏览的次数
    item_reads=sub_items[sub_items['type']==1][['sku_id','typeCounts']]
    item_reads.rename(columns={'typeCounts':'item_reads'},inplace=True)
    item_reads=pd.merge(all_items,item_reads,on='sku_id',how='left')
    item_reads.replace(np.nan,0,inplace=True)

    ##商品子集中商品被购买次数和被浏览次数的比值
    item_buy_read=pd.merge(item_buys,item_reads,on='sku_id',how='right')
    item_buy_read.replace(np.nan,0,inplace=True)
    item_buy_read['buy_read_ratio']=item_buy_read['item_buys']/item_buy_read['item_reads']
    item_buy_read=item_buy_read[['sku_id','buy_addshop_ratio']]
    item_buy_read=pd.merge(all_items,item_buy_read,on='sku_id',how='left')
    item_buy_read.replace(np.inf,-1,inplace=True)

    ##商品被浏览人数中发生购买的比值
    item_user_reads=sub_item_users[sub_item_users['type']==1]
    item_user_buys=sub_item_users[sub_item_users['type']==4]
    items_buyIn_reads=pd.DataFrame(columns=['sku_id','buyIn_reads_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        if item in list(item_user_reads['sku_id']):
            if item in list(item_user_buys['sku_id']):
                read_users=set(item_user_reads[item_user_reads['sku_id']==item]['user_id'])
                buy_users=set(item_user_buys[item_user_buys['sku_id']==item]['user_id'])
                buy_read_users=read_users&buy_users
                ratio=len(buy_read_users)/float(len(read_users))
                items_buyIn_reads.loc[i]=[item,ratio]
            else:
                items_buyIn_reads.loc[i]=[item,0]

        items_buyIn_reads.loc[i]=[item,-1]


    ##商品被购买里面被二次以上购买的比值
    item_user_buys=sub_item_users[sub_item_users['type']==4][['sku_id','user_id','user_typeCounts']]
    item_buyTwo_ratio=pd.DataFrame(columns=['sku_id','buyTwo_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        if item in list(item_user_buys['sku_id']):
            all_item_users=len(item_user_buys[item_user_buys['sku_id']==item])
            item_buy_two=len(item_user_buys[(item_user_buys['sku_id']==item)&(item_user_buys['user_typeCounts']>=2)])
            ratio=item_buy_two/float(all_item_users)
            item_buyTwo_ratio.loc[i]=[item,ratio]
        else:
            item_buyTwo_ratio.loc[i]=[item,-1]

    ##商品最后一次购买距离预测日的距离
    item_last_time=sub_item_time[sub_item_time['type']==4][['sku_id','time']]
    item_last_time=item_last_time.drop_duplicates(['sku_id'],keep='last')
    item_last_time['time']=item_last_time.time.apply(lambda x:getTopred(pre_day,x))
    item_last_time=pd.merge(all_items,item_last_time,on='sku_id',how='left')
    item_last_time.replace(np.nan,-1,inplace=True)
    item_last_time.rename(columns={'time':'last_buy_toPre'},inplace=True)

    ##商品最后一次与用户发生交互距离预测日的距离
    item_last_time=sub_item_time[['sku_id','time']]
    item_last_time=item_last_time.drop_duplicates(['sku_id'],keep='last')
    item_last_time['time']=item_last_time.time.apply(lambda x:getTopred(pre_day,x))
    item_last_time=pd.merge(all_items,item_last_time,on='sku_id',how='left')
    item_last_time.replace(np.nan,-1,inplace=True)
    item_last_time.rename(columns={'time':'last_active_toPre'},inplace=True)

    ##商品最大购买量距离预测日时长
    item_last_time=sub_item_time[sub_item_time['type']==4][['sku_id','time','time_typeCounts']]
    item_last_time.sort_values(by=['sku_id','time_typeCounts','time'],inplace=True)
    item_last_time=item_last_time.drop_duplicates(['sku_id'],keep='last')
    item_last_time['time']=item_last_time.time.apply(lambda x:getTopred(pre_day,x))
    item_last_time=pd.merge(all_items,item_last_time,on='sku_id',how='left')
    item_last_time.replace(np.nan,-1,inplace=True)
    item_last_time.rename(columns={'time':'max_buy_toPre'},inplace=True)

    ##该商品的交互量占该品类商品的交互量比值
    ##商品子集中，品类与对应交互行为次数统计
    sub_item_cate=clean_allActions[['cate']]
    sub_item_cate['cate_actions']=1
    sub_item_cate=sub_item_cate.groupby(['cate']).agg('sum').reset_index()
    item_cate_ratio=pd.DataFrame(columns=['sku_id','action_cate_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        cate=list(clean_allActions[clean_allActions['sku_id']==item]['cate'])[0]
        item_actions=sum(list(sub_items[sub_items['sku_id']==item]['typeCounts']))
        cate_actions=sum(list(sub_item_cate[sub_item_cate['cate']==cate]['cate_actions']))
        ratio=item_actions/float(cate_actions)
        item_cate_ratio.loc[i]=[item,ratio]


    ##该商品的交互量占该品牌商品的交互量比值
    sub_item_brand=clean_allActions[['brand']]
    sub_item_brand['brand_actions']=1
    sub_item_brand=sub_item_cate.groupby(['brand']).agg('sum').reset_index()
    item_brand_ratio=pd.DataFrame(columns=['sku_id','action_brand_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        brand=list(clean_allActions[clean_allActions['sku_id']==item]['brand'])[0]
        item_actions=sum(list(sub_items[sub_items['sku_id']==item]['typeCounts']))
        brand_actions=sum(list(sub_item_brand[sub_item_brand['brand']==brand]['brand_actions']))
        ratio=item_actions/float(brand_actions)
        item_brand_ratio.loc[i]=[item,ratio]


    ##该商品的购买量占该品类商品购买量的比值
    sub_cate_buy=clean_allActions[clean_allActions['type']==4][['cate']]
    sub_cate_buy['cate_buy_counts']=1
    sub_cate_buy=sub_cate_buy.groupby(['cate']).agg('sum').reset_index()
    sub_item_buys=sub_items[sub_items['type']==4][['sku_id','typeCounts']]
    item_buy_cate_ratio=pd.DataFrame(columns=['sku_id','buy_cate_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        if item in list(sub_cate_buy['sku_id']):
            if item in list(sub_item_buys['sku_id']):
                cate=list(clean_allActions[clean_allActions['sku_id']==item]['cate'])[0]
                item_buys=sum(list(sub_item_buys[sub_item_buys['sku_id']==item]['typeCounts']))
                cate_buys=sum(list(sub_cate_buy[sub_cate_buy['cate']==cate]['cate_buy_counts']))
                ratio=item_buys/float(cate_buys)
                item_buy_cate_ratio.loc[i]=[item,ratio]
            else:
                item_buy_cate_ratio.loc[i]=[item,0]
        else:
            item_buy_cate_ratio.loc[i]=[item,-1]


    ##该商品的购买量占该品牌商品购买量的比值
    sub_brand_buy=clean_allActions[clean_allActions['type']==4][['brand']]
    sub_brand_buy['brand_buy_counts']=1
    sub_brand_buy=sub_brand_buy.groupby(['brand']).agg('sum').reset_index()
    sub_item_buys=sub_items[sub_items['type']==4][['sku_id','typeCounts']]
    item_buy_brand_ratio=pd.DataFrame(columns=['sku_id','buy_brand_ratio'])
    i=-1
    for item in list(all_items['sku_id']):
        i+=1
        if item in list(sub_brand_buy['sku_id']):
            if item in list(sub_item_buys['sku_id']):
                brand=list(clean_allActions[clean_allActions['sku_id']==item]['brand'])[0]
                item_buys=sum(list(sub_item_buys[sub_item_buys['sku_id']==item]['typeCounts']))
                brand_buys=sum(list(sub_brand_buy[sub_brand_buy['brand']==brand]['brand_buy_counts']))
                ratio=item_buys/float(brand_buys)
                item_buy_brand_ratio.loc[i]=[item,ratio]
            else:
                item_buy_brand_ratio.loc[i]=[item,0]
        else:
            item_buy_brand_ratio.loc[i]=[item,-1]



























