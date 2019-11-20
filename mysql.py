import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pymysql
from sqlalchemy import create_engine

# sqlalchemy
engine = create_engine('mysql+pymysql://root:Chilugan6xiang@localhost:3306/test')
# engine = create_engine('mysql+pymysql://ABSIBSusr:vq6KO1blTJQI@10.47.145.50:3306/test')
# sql = 'SELECT * FROM t_cash_loan_asset_pool WHERE RMNG_PNP>48000'
# df = pd.read_sql_query(sql, engine)
# print(df.info())

# # pymysql
# db = pymysql.connect(host='10.47.145.50',
#                              user='ABSIBSusr',
#                              password='vq6KO1blTJQI',
#                              db='test')
# cursor = db.cursor()
# sql = 'SELECT * FROM t_cash_loan_asset_pool WHERE RMNG_PNP>48000'
# try:
#     cursor.execute(sql)
#     results = cursor.fetchone()
#     print(results)
# except:
#     print("Error: unable to fetch data")
# db.close()

# 新建pandas中的DataFrame, 只有id,num两列
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('D:/PyCharm/PyCharm 2019.2/pywork/pydata-book/datasets/movielens/users.dat', sep='::',
                      header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('D:/PyCharm/PyCharm 2019.2/pywork/pydata-book/datasets/movielens/ratings.dat', sep='::',
                        header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('D:/PyCharm/PyCharm 2019.2/pywork/pydata-book/datasets/movielens/movies.dat', sep='::',
                       header=None, names=mnames)

# 将新建的DataFrame储存为MySQL中的数据表，不储存index列
users.to_sql('m_user', engine, index = False)
movies.to_sql('m_movies', engine, index = False)
movies.head(10).to_sql('m_movies_02', engine, index = False)
ratings.to_sql('m_ratings', engine, index = False)

print('Read from and write to Mysql table successfully!')
