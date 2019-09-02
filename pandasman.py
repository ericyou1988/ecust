# pandas 使用 example
import pandas as pd
import numpy as np

# # espn = pd.read_csv('D:/PyCharm/PyCharm 2019.2/pywork/ABS/data/person/example.csv',nrows=5)
# espn = pd.read_csv('D:/PyCharm/PyCharm 2019.2/pywork/ABS/data/person/example.csv')
# print(espn)
# # 生成列表
# print(espn.columns.tolist())
# print(espn.iloc[3].tolist())
# # print(espn['LOAN_NO'].tolist())
# # 统计数值出现频次
# print(espn['DEBTOR_INDIV_PROVINCE'].value_counts(normalize=False).reset_index())


# tnt=pd.DataFrame({ 'a' :[0,0,0],  'b' : [1,1,1]})
# print(tnt)
# # copy和=的区别
# # tnt1=tnt
# tnt1=tnt.copy()
# tnt1['a']=tnt1['a']+1
# print(tnt1)
# print(tnt)


# 横向合并
# df1 = pd.DataFrame(np.arange(12).reshape((3, 4)), index=['a', 'b', 'c'], columns=['A', 'B', 'C', 'D'])
# print(df1)
#
# df2 = pd.DataFrame(np.arange(15).reshape((5, 3)), index=['a', 'c', 'd', 'e', 'f'], columns=['D', 'F', 'G'])
# print(df2)
#
# df = df1.join(df2, lsuffix='l',rsuffix='r')  # suffix后缀的意思
# print(df)
#
# df = df1.join(df2, how='right', lsuffix='l',rsuffix='r')
# print(df)
#
# df = df1.join(df2, how='inner', lsuffix='l',rsuffix='r')
# print(df)

# # 计算缺失值数量
# df = pd.DataFrame({  'id' : [1,2,3],  'c1' :[0,0,np.nan],  'c2' : [np.nan,1,1]})
# print(df)
# print(df[[ 'c1' ,  'c2' ]].isnull())
# print(df[[ 'c1' ,  'c2' ]].isnull().sum(axis=1))
# df[ 'num_nulls' ] = df[[ 'c1' ,  'c2' ]].isnull().sum(axis=1)
# print(df)
# df.head()

# cut函数使用
ages = np.array([1,5,10,40,36,12,58,62,77,89,100,18,20,25,30,32]) #年龄数据
print(pd.cut(ages, [0,5,20,30,50,100], labels=[u"婴儿",u"青年",u"中年",u"壮年",u"老年"]))
print(pd.cut(ages, [0,5,20,30,50,100], labels=[u"婴儿",u"青年",u"中年",u"壮年",u"老年"],retbins=True))
print(pd.cut(ages, [0,5,20,30,50,100], labels=False)
)