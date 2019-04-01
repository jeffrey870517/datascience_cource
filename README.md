# datascience_cource
Data_Sciences_Course

HW1

使用 python3 app.py，需要GPU

資料集：使用台灣電力公司_未來一週電力供需預測之 尖峰負載(MW) 以及歷年尖峰負載備轉容餘作為資料集

Step1

引入經換算過的尖峰附載 2014/01/01 ~ 2019/02/28 作為train_data的y

Step2

引入透過氣象局歷年天氣資料換算過的最高體感溫度作為train_data的x

Step3

將兩筆資料放入ANN中訓練，以資料及取出的test_data計算accuracy

Step4

透過氣象局未來一週體感溫度預測得出尖峰負載，並連續寫入submission1.csv,以accuracy最佳值作為本次作業預測結果並寫入submission.csv

