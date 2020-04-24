from sklearn import svm
import DC

if __name__ == '__main__':
    stock = '601857.SH'
    dc = DC.data_collect(stock, '2016-03-01', '2017-01-01')
    train = dc.data_train
    target = dc.data_target
    test_case = [dc.test_case]
    model = svm.SVC()               # 建模
    model.fit(train, target)        # 训练
    ans2 = model.predict(test_case) # 预测
    # 输出对2018-03-02的涨跌预测，1表示涨，0表示不涨。
    print(ans2[0])


