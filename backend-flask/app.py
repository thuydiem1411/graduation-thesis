from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from lightgbm import LGBMRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from matplotlib.figure import Figure
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from xgboost import XGBRegressor
import optuna
from optuna.samplers import TPESampler
app = Flask(__name__)

# Sample data loading (replace with your actual data loading code)
data_final = pd.read_excel(r"D:\f\stock_data.xlsx")
print(data_final.columns)  # Kiểm tra tên cột

# Tạo một cột mới 'Final price' và gán giá trị mặc định là giá đóng cửa
data_final['GiaCuoiCung'] = data_final['GiaDongCua']

# Xác định các hàng có giá điều chỉnh không phải là null và gán giá điều chỉnh cho cột 'Final price'
adjusted_price_not_null = data_final['GiaDieuChinh'].notnull()
data_final.loc[adjusted_price_not_null, 'GiaCuoiCung'] = data_final.loc[adjusted_price_not_null, 'GiaDieuChinh']
# Drop hai cột dữ liệu 'Closed price' và 'Adjusted price'
data_final.drop(columns=['GiaDongCua', 'GiaDieuChinh'], inplace=True)

# Định nghĩa hàm để giữ lại dòng đầu tiên cho mỗi ngày trong mỗi nhóm mã cổ phiếu
def keep_first_duplicate_days(group):
    return group.drop_duplicates(subset='Ngay', keep='first')

# Áp dụng hàm cho từng nhóm mã cổ phiếu
# df_cleaned = data_final.groupby('MaChungKhoan').apply(keep_first_duplicate_days, include_groups=False).reset_index(drop=True)
# Áp dụng hàm cho từng nhóm mã cổ phiếu
df_cleaned = data_final.groupby('MaChungKhoan').apply(keep_first_duplicate_days).reset_index(drop=True)
data_final = df_cleaned.copy()


# Chuyển đổi cột "Ngày" sang dạng datetime
data_final['Ngay'] = pd.to_datetime(data_final['Ngay'], format='%d/%m/%Y')
# Sắp xếp lại dữ liệu theo thứ tự thời gian
data_final = data_final.sort_values(by='Ngay')
# Kiểm tra tên cột và xem dữ liệu
print(data_final.columns)  # Kiểm tra tên cột
# Danh sách các mã cổ phiếu
list_of_stock_codes = [ 'YEG', 'VCF', 'PTL', 'KBC', 'CCL', 'VCB', 'PVD', 'CCI', 'JVC',
    'ITD', 'CDC', 'VCA', 'QCG', 'ITC', 'CHP', 'PVT', 'VAF', 'KDC',
    'VCG', 'VDS', 'POW', 'KOS', 'BWE', 'VDP', 'PPC', 'PTC', 'KMR',
    'KHP', 'C32', 'VCI', 'PTB', 'KDH', 'C47', 'PSH', 'RAL', 'ITA',
    'CIG', 'CLL', 'ADG', 'TV2', 'SAV', 'HVX', 'SBA', 'ICT', 'HVN',
    'TTF', 'SBT', 'HVH', 'CMG', 'TTE', 'SBV', 'CLW', 'VSC', 'IDI',
    'SAB', 'UIC', 'RDP', 'IMP', 'CII', 'TYA', 'REE', 'ILB', 'ABS',
    'VTO', 'CKG', 'TVT', 'S4A', 'IJC', 'CLC', 'TVS', 'BVH', 'HUB',
    'KPF', 'VGC', 'BFC', 'VNE', 'PDR', 'LSS', 'BHN', 'VMD', 'MBB',
    'PET', 'BIC', 'VJC', 'PGC', 'LM8', 'PGD', 'LIX', 'LPB', 'VRE',
    'PDN', 'BCM', 'AST', 'VOS', 'OGC', 'MIG', 'BCE', 'VNS', 'VNG',
    'OPC', 'BCG', 'VNINDEX', 'PAC', 'MDG', 'PC1', 'MCP', 'MHC', 'BID',
    'VIX', 'PGI', 'VIB', 'PLP', 'LCG', 'BRC', 'VHM', 'PLX', 'BMP',
    'LBM', 'L10', 'BTP', 'VHC', 'PNC', 'KSB', 'BTT', 'PMG', 'LDG',
    'PJT', 'VIC', 'LHG', 'ADS', 'PHC', 'LGL', 'BKG', 'VIP', 'AAT',
    'YBM', 'PHR', 'LGC', 'BMC', 'VID', 'PIT', 'LEC', 'BMI', 'PNJ',
    'MSB', 'CMV', 'SC5', 'STG', 'HAR', 'DBT', 'THG', 'STK', 'HAH',
    'HAS', 'DC4', 'SVC', 'HAG', 'DCL', 'TDW', 'SVD', 'GVR', 'TEG',
    'DCM', 'STB', 'DBD', 'SRF', 'HCM', 'VSH', 'ACL', 'SSB', 'HCD',
    'TIP', 'DAT', 'SSI', 'HBC', 'DBC', 'TIX', 'ST8', 'HAX', 'TLD',
    'TDP', 'SVI', 'GTA', 'GEX', 'DHG', 'TCT', 'TCD', 'GEG', 'DHM',
    'TCB', 'TCR', 'FUCVREIT', 'DIG', 'TCO', 'TCL', 'DMC', 'DLG', 'TCH',
    'GIL', 'TBC', 'TDC', 'SVT', 'GSP', 'DGC', 'TDM', 'SZC', 'GMD',
    'DGW', 'TDH', 'SZL', 'GMC', 'ACC', 'DHA', 'TDG', 'VSI', 'DHC',
    'HDB', 'TTA', 'SRC', 'DAH', 'HT1', 'CRE', 'TPB', 'SGN', 'HSL',
    'SGR', 'SFI', 'HSG', 'TNT', 'SGT', 'HRC', 'CSV', 'TNI', 'SHA',
    'CSM', 'HQC', 'TPC', 'HTI', 'HU1', 'CMX', 'TSC', 'SCR', 'HTV',
    'SCS', 'CRC', 'HTN', 'TRC', 'SFC', 'HTL', 'COM', 'TRA', 'SFG',
    'CNG', 'CTD', 'TNH', 'SHI', 'SMA', 'HHS', 'CVT', 'TMS', 'SMB',
    'HHP', 'HID', 'D2D', 'SMC', 'HDG', 'DAG', 'TLH', 'SPM', 'HDC',
    'TMP', 'SKG', 'TMT', 'CTS', 'HPG', 'CTF', 'TNC', 'SHP', 'HNG',
    'SJD', 'HMC', 'CTG', 'TNA', 'ACB', 'VTB', 'CTI', 'TN1', 'SJS',
    'HII', 'TLG', 'OCB', 'SAM', 'TCM', 'MSN', 'NVL', 'VPD', 'NVT',
    'VPI', 'ANV', 'NT2', 'NAF', 'AGR', 'NHA', 'VPS', 'NLG', 'VPB',
    'NKG', 'NCT', 'MWG', 'NTL', 'ASP', 'APH', 'VPG', 'NBB', 'NHH',
    'ASM', 'NNC', 'MSH', 'VPH', 'AGG', 'NAV', 'APG', 'VRC' ]  # (bỏ qua vì danh sách quá dài)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req_data = request.get_json()
        print(req_data)
        stock_code = req_data['stock_code']
        algorithm = req_data['algorithm']
        parameter = req_data['parameter']

        # Tạo DataFrame cho mã cổ phiếu được chọn
        # df_stock = data_final[data_final['MaChungKhoan'] == stock_code][['Ngay', 'GiaCuoiCung']]
        # Tạo DataFrame cho mã cổ phiếu được chọn
        df_stock = data_final.loc[data_final['MaChungKhoan'] == stock_code, ['Ngay', 'GiaCuoiCung']]
        # print(data_final['MaChungKhoan'] == stock_code)

        df_stock.set_index('Ngay', inplace=True)
        # Chọn ngày 30/01/2024
        split_date = pd.to_datetime('2023-12-29')

        # Chia tập dữ liệu thành train và test
        train_data = df_stock[df_stock.index <= split_date]['GiaCuoiCung'].values.reshape(-1, 1)
        test_data = df_stock[df_stock.index > split_date]['GiaCuoiCung'].values.reshape(-1, 1)

        results =[]
        if algorithm == 'xgboots':
            if parameter == 'default':
                # Chuẩn hóa dữ liệu
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_data_scaled = scaler.fit_transform(train_data)
                test_data_scaled = scaler.transform(test_data)
                # Xây dựng dữ liệu đầu vào cho XGBoost
                X_train, y_train = [], []
                for i in range(60, len(train_data_scaled)):
                        X_train.append(train_data_scaled[i-60:i, 0])
                        y_train.append(train_data_scaled[i, 0])
                X_train, y_train = np.array(X_train), np.array(y_train)
                # Tạo mô hình XGBoost
                model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)

                # Thực hiện k-fold cross-validation với TimeSeriesSplit
                tscv = TimeSeriesSplit(n_splits=5)
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')

                # In kết quả cross-validation
                # print(f'Mã cổ phiếu: {stock_code}')
                # print('CV scores:', -cv_scores)
                # print('Mean CV score:', -np.mean(cv_scores))
                # print('Standard Deviation CV score:', np.std(cv_scores))

                # Huấn luyện mô hình trên toàn bộ dữ liệu train
                model.fit(X_train, y_train)

                # Dự đoán trên tập train
                y_train_predict = model.predict(X_train)
                y_train_predict = scaler.inverse_transform(y_train_predict.reshape(-1, 1))

                # Xử lý dữ liệu test
                total_data = np.concatenate((train_data[-60:], test_data), axis=0)
                inputs = scaler.transform(total_data)
                X_test = []
                for i in range(60, len(inputs)):
                    X_test.append(inputs[i-60:i, 0])
                X_test = np.array(X_test)

                # Dự đoán trên tập test
                y_test = df_stock[df_stock.index > split_date]['GiaCuoiCung'].values  # giá thực
                y_test_predict = model.predict(X_test)
                y_test_predict = scaler.inverse_transform(y_test_predict.reshape(-1, 1))
                # print('Length of y_test:', len(y_test))
                # print('Length of y_test_predict:', len(y_test_predict))



               # Đảm bảo số lượng mẫu của y_test và y_test_predict là như nhau
                if len(y_test) != len(y_test_predict):
                    # Có thể làm điều chỉnh tại đây để đảm bảo số lượng mẫu của y_test và y_test_predict trùng khớp
                    min_len = min(len(y_test), len(y_test_predict))
                    y_test = y_test[:min_len]
                    y_test_predict = y_test_predict[:min_len]

                # Lập biểu đồ so sánh
                # plt.figure(figsize=(24, 8))
                # plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')

                # Chỉ số bắt đầu và kết thúc cho dự đoán train
                train_start_index = 60
                train_end_index = train_start_index + len(y_train_predict)
                train_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                train_predict_plot[:] = np.nan
                train_predict_plot[60:len(y_train_predict) + 60] = y_train_predict.reshape(-1)
                plt.plot(df_stock.index, train_predict_plot, label='Giá dự đoán train', color='green')

                # Chỉ số bắt đầu và kết thúc cho dự đoán test
                test_start_index = len(train_data)
                test_end_index = test_start_index + len(y_test_predict)
                test_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                test_predict_plot[:] = np.nan
                test_predict_plot[test_start_index:test_end_index] = y_test_predict.reshape(-1)
                # plt.plot(df_stock.index, test_predict_plot, label='Giá dự đoán test', color='blue')

                # plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                # plt.xlabel('Thời gian')
                # plt.ylabel('Giá đóng cửa (VNĐ)')
                # plt.legend()
                # plt.show()

                # Đánh giá mô hình
                # print(f'Mã cổ phiếu: {stock_code}')
                train_r2_score = r2_score(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mae = mean_absolute_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_rmse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict, squared=False)
                train_mse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                
                test_r2_score = r2_score(y_test, y_test_predict)
                test_mape = mean_absolute_percentage_error(y_test, y_test_predict)
                test_mae = mean_absolute_error(y_test, y_test_predict)
                test_rmse = mean_squared_error(y_test, y_test_predict, squared=False)
                test_mse = mean_squared_error(y_test, y_test_predict)

                # print('Độ phù hợp tập train:', train_r2_score)
                # print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', train_mae)
                # print('Phần trăm sai số tuyệt đối trung bình tập train:',train_mape)
                # print('Root Mean Squared Error trên tập train (VNĐ):', train_rmse)
                # print('Mean Squared Error trên tập train (VNĐ^2):', train_mse)

                # print('Độ phù hợp tập test:', test_r2_score)
                # print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', test_mae)
                # print('Phần trăm sai số tuyệt đối trung bình tập test:', test_mape)
                # print('Root Mean Squared Error trên tập test (VNĐ):', test_rmse)
                # print('Mean Squared Error trên tập test (VNĐ^2):', test_mse)

                # Dự đoán cho ngày kế tiếp
                x_next = np.array([inputs[-60:, 0]])
                y_next_predict = model.predict(x_next)
                y_next_predict = scaler.inverse_transform(y_next_predict.reshape(-1, 1))

                # Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
                next_date = df_stock.index[-1] + pd.Timedelta(days=1)
                df_next = pd.DataFrame({'Ngay': [next_date], 'GiaCuoiCung': [y_next_predict[0][0]]})
                df_stock = pd.concat([df_stock, df_next.set_index('Ngay')])

                # Vẽ biểu đồ mới với dự đoán cho ngày kế tiếp
                plt.figure(figsize=(15, 5), dpi=72)
                plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')
                plt.plot(df_stock.index[:len(train_predict_plot)], train_predict_plot, label='Giá dự đoán train', color='green')
                plt.plot(df_stock.index[test_start_index:test_start_index + len(y_test_predict)], test_predict_plot[test_start_index:test_start_index + len(y_test_predict)], label='Giá dự đoán test', color='blue')
                plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá đóng cửa (VNĐ)')
                plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                plt.legend()
                # plt.show()
                print("1")
                # Lưu biểu đồ vào buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                data_uri = base64.b64encode(buffer.read()).decode()
                plt.close()

                # Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
                actual_closing_price = df_stock['GiaCuoiCung'].iloc[-2]
                print("2")
                comparison_data_final = pd.DataFrame({'Ngay': [next_date], 'GiaDuDoan': [y_next_predict[0][0]], 'GiaNgayTruoc': [actual_closing_price]})
                print("3")
                print(comparison_data_final)
                results.append({
                    'MaChungKhoan': stock_code,
                    'DoPhuHopTrain_XGB': train_r2_score,
                    'SaiSoTrungBinhTrain_XGB': train_mae,
                    'RMSETrain_XGB': train_rmse,
                    'PhanTramSaiSoTrain_XGB': float(train_mape),
                    'MSETrain_XGB': train_mse,
                    'DoPhuHopTest_XGB': test_r2_score,
                    'SaiSoTrungBinhTest_XGB': test_mae,
                    'RMSETest_XGB': test_rmse,
                    'PhanTramSaiSoTest_XGB': test_mape,
                    'MSETest_XGB': test_mse,
                    'DuDoanCuoiCung_XGB': float(y_next_predict[0][0]),
                    'GiaThucTeCuoi_XGB': comparison_data_final.to_dict(),
                    'chart': data_uri
                })
                print("135",results)
                return jsonify(results)

            else :
                # Chuẩn hóa dữ liệu
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_data_scaled = scaler.fit_transform(train_data)
                test_data_scaled = scaler.transform(test_data)
                # Xây dựng dữ liệu đầu vào cho XGBoost
                X_train, y_train = [], []
                for i in range(60, len(train_data_scaled)):
                        X_train.append(train_data_scaled[i-60:i, 0])
                        y_train.append(train_data_scaled[i, 0])
                X_train, y_train = np.array(X_train), np.array(y_train)
                # Định nghĩa hàm mục tiêu cho Optuna
                # Định nghĩa không gian tìm kiếm cho Bayesian Optimization
                def objective(trial):
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                        'max_depth': trial.suggest_int('max_depth', 1, 13),
                        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 1.0),
                        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 1.0),
                        'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
                        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.8, 1.0)
                    }
                    model = XGBRegressor(**params, objective='reg:squarederror')
                    tscv = TimeSeriesSplit(n_splits=10)
                    scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
                    return -scores.mean()

                # Tạo một study và tối ưu hóa nó
                study = optuna.create_study(sampler=TPESampler(), direction='minimize')
                study.optimize(objective, n_trials=100)

                # In ra các siêu tham số tốt nhất
                print(f'Mã cổ phiếu: {stock_code}')
                print("Best parameters found: ", study.best_params)
                print("Best CV score: ", study.best_value)

                # Huấn luyện mô hình với siêu tham số tốt nhất
                best_model = XGBRegressor(**study.best_params, objective='reg:squarederror')
                best_model.fit(X_train, y_train)

                # Dự đoán trên tập train
                y_train_predict = best_model.predict(X_train)
                y_train_predict = scaler.inverse_transform(y_train_predict.reshape(-1, 1))

                # Xử lý dữ liệu test
                total_data = np.concatenate((train_data[-60:], test_data), axis=0)
                inputs = scaler.transform(total_data)
                X_test = []
                for i in range(60, len(inputs)):
                    X_test.append(inputs[i-60:i, 0])
                X_test = np.array(X_test)

                # Dự đoán trên tập test
                y_test = df_stock[df_stock.index > split_date]['GiaCuoiCung'].values  # giá thực
                y_test_predict = best_model.predict(X_test)
                y_test_predict = scaler.inverse_transform(y_test_predict.reshape(-1, 1))
                


                # Đảm bảo số lượng mẫu của y_test và y_test_predict là như nhau
                if len(y_test) != len(y_test_predict):
                    # Có thể làm điều chỉnh tại đây để đảm bảo số lượng mẫu của y_test và y_test_predict trùng khớp
                    min_len = min(len(y_test), len(y_test_predict))
                    y_test = y_test[:min_len]
                    y_test_predict = y_test_predict[:min_len]
                # Lập biểu đồ so sánh
                plt.figure(figsize=(24, 8))
                plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')

                # Chỉ số bắt đầu và kết thúc cho dự đoán train
                train_start_index = 60
                train_end_index = train_start_index + len(y_train_predict)
                train_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                train_predict_plot[:] = np.nan
                train_predict_plot[60:len(y_train_predict) + 60] = y_train_predict.reshape(-1)
                plt.plot(df_stock.index, train_predict_plot, label='Giá dự đoán train', color='green')

                # Chỉ số bắt đầu và kết thúc cho dự đoán test
                test_start_index = len(train_data)
                test_end_index = test_start_index + len(y_test_predict)
                test_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                test_predict_plot[:] = np.nan
                test_predict_plot[test_start_index:test_end_index] = y_test_predict.reshape(-1)
                plt.plot(df_stock.index, test_predict_plot, label='Giá dự đoán test', color='blue')

                plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá đóng cửa (VND)')
                plt.legend()
                plt.show()

                print(f'Mã cổ phiếu: {stock_code}')
                train_r2_score = r2_score(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mae = mean_absolute_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_rmse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict, squared=False)
                train_mse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                
                test_r2_score = r2_score(y_test, y_test_predict)
                test_mape = mean_absolute_percentage_error(y_test, y_test_predict)
                test_mae = mean_absolute_error(y_test, y_test_predict)
                test_rmse = mean_squared_error(y_test, y_test_predict, squared=False)
                test_mse = mean_squared_error(y_test, y_test_predict)

                print('Độ phù hợp tập train:', train_r2_score)
                print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', train_mae)
                print('Phần trăm sai số tuyệt đối trung bình tập train:',train_mape)
                print('Root Mean Squared Error trên tập train (VNĐ):', train_rmse)
                print('Mean Squared Error trên tập train (VNĐ^2):', train_mse)

                print('Độ phù hợp tập test:', test_r2_score)
                print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', test_mae)
                print('Phần trăm sai số tuyệt đối trung bình tập test:', test_mape)
                print('Root Mean Squared Error trên tập test (VNĐ):', test_rmse)
                print('Mean Squared Error trên tập test (VNĐ^2):', test_mse)

                # Dự đoán cho ngày kế tiếp
                x_next = np.array([inputs[-60:, 0]])
                y_next_predict = best_model.predict(x_next)
                y_next_predict = scaler.inverse_transform(y_next_predict.reshape(-1, 1))

                # Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
                next_date = df_stock.index[-1] + pd.Timedelta(days=1)
                df_next = pd.DataFrame({'Ngay': [next_date], 'GiaCuoiCung': [y_next_predict[0][0]]})
                df_stock = pd.concat([df_stock, df_next.set_index('Ngay')])

                # Vẽ biểu đồ mới với dự đoán cho ngày kế tiếp
                plt.figure(figsize=(15, 5))
                plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')
                plt.plot(df_stock.index[:len(train_predict_plot)], train_predict_plot, label='Giá dự đoán train', color='green')
                plt.plot(df_stock.index[test_start_index:test_start_index + len(y_test_predict)], test_predict_plot[test_start_index:test_start_index + len(y_test_predict)], label='Giá dự đoán test', color='blue')
                plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá đóng cửa (VND)')
                plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                plt.legend()
                # Lưu biểu đồ vào buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                data_uri = base64.b64encode(buffer.read()).decode()
                plt.close()
                print("1")
                
                # Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
                actual_closing_price = df_stock['GiaCuoiCung'].iloc[-2]
                print("2")
                comparison_data_final = pd.DataFrame({'Ngay': [next_date], 'GiaDuDoan': [y_next_predict[0][0]], 'GiaNgayTruoc': [actual_closing_price]})
                print("3")
                results.append({
                'MaChungKhoan': stock_code,
                'DoPhuHopTrain_XGB': train_r2_score,
                'SaiSoTrungBinhTrain_XGB': train_mae,
                'RMSETrain_XGB': train_rmse,
                'PhanTramSaiSoTrain_XGB': train_mape,
                'MSETrain_XGB': train_mse,

                'DoPhuHopTest_XGB': test_r2_score,
                'SaiSoTrungBinhTest_XGB': test_mae,
                'RMSETest_XGB': test_rmse,
                'PhanTramSaiSoTest_XGB': test_mape,
                'MSETest_XGB': test_mse,
                'DuDoanCuoiCung_XGB': y_next_predict[0][0],
                'GiaThucTeCuoi_XGB': comparison_data_final.to_dict()
                })
                return jsonify(results)
        elif algorithm =='lightgbm':
            if parameter == 'default': 
                 # Chuẩn hóa dữ liệu
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_data_scaled = scaler.fit_transform(train_data)
                test_data_scaled = scaler.transform(test_data)
                # Xây dựng dữ liệu đầu vào cho LightGBM
                X_train, y_train = [], []
                for i in range(60, len(train_data_scaled)):
                    X_train.append(train_data_scaled[i-60:i, 0])
                    y_train.append(train_data_scaled[i, 0])
                X_train, y_train = np.array(X_train), np.array(y_train)
                # Huấn luyện mô hình LightGBM
                model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
                model.fit(X_train, y_train)
                # Dự đoán trên tập train
                y_train_predict = model.predict(X_train)
                y_train_predict = scaler.inverse_transform(y_train_predict.reshape(-1, 1))
                 # Xử lý dữ liệu test
                total_data = np.concatenate((train_data[-60:], test_data), axis=0)
                inputs = scaler.transform(total_data)
                X_test = []
                for i in range(60, len(inputs)):
                    X_test.append(inputs[i-60:i, 0])
                X_test = np.array(X_test)
                 # Dự đoán trên tập test
                y_test = df_stock[df_stock.index > split_date]['GiaCuoiCung'].values  # giá thực
                y_test_predict = model.predict(X_test)
                y_test_predict = scaler.inverse_transform(y_test_predict.reshape(-1, 1))
                # Lập biểu đồ so sánh
                plt.figure(figsize=(24, 8))
                plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')

                # Chỉ số bắt đầu và kết thúc cho dự đoán train
                train_start_index = 60
                train_end_index = train_start_index + len(y_train_predict)
                train_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                train_predict_plot[:] = np.nan
                train_predict_plot[60:len(y_train_predict) + 60] = y_train_predict.reshape(-1)
                plt.plot(df_stock.index, train_predict_plot, label='Giá dự đoán train', color='green')

                # Chỉ số bắt đầu và kết thúc cho dự đoán test
                test_start_index = len(train_data)
                test_end_index = test_start_index + len(y_test_predict)
                test_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                test_predict_plot[:] = np.nan
                test_predict_plot[test_start_index:test_end_index] = y_test_predict.reshape(-1)
                plt.plot(df_stock.index, test_predict_plot, label='Giá dự đoán test', color='blue')

                plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá đóng cửa (VNĐ)')
                plt.legend()
                # plt.show()

                print(f'Mã cổ phiếu: {stock_code}')
                train_r2_score = r2_score(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mae = mean_absolute_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_rmse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict, squared=False)
                train_mse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                
                test_r2_score = r2_score(y_test, y_test_predict)
                test_mape = mean_absolute_percentage_error(y_test, y_test_predict)
                test_mae = mean_absolute_error(y_test, y_test_predict)
                test_rmse = mean_squared_error(y_test, y_test_predict, squared=False)
                test_mse = mean_squared_error(y_test, y_test_predict)

                print('Độ phù hợp tập train:', train_r2_score)
                print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', train_mae)
                print('Phần trăm sai số tuyệt đối trung bình tập train:',train_mape)
                print('Root Mean Squared Error trên tập train (VNĐ):', train_rmse)
                print('Mean Squared Error trên tập train (VNĐ^2):', train_mse)

                print('Độ phù hợp tập test:', test_r2_score)
                print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', test_mae)
                print('Phần trăm sai số tuyệt đối trung bình tập test:', test_mape)
                print('Root Mean Squared Error trên tập test (VNĐ):', test_rmse)
                print('Mean Squared Error trên tập test (VNĐ^2):', test_mse)
                # Dự đoán cho ngày kế tiếp
                x_next = np.array([inputs[-60:, 0]])
                y_next_predict = model.predict(x_next)
                y_next_predict = scaler.inverse_transform(y_next_predict.reshape(-1, 1))

                # Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
                next_date = df_stock.index[-1] + pd.Timedelta(days=1)
                df_next = pd.DataFrame({'Ngay': [next_date], 'GiaCuoiCung': [y_next_predict[0][0]]})
                df_stock = pd.concat([df_stock, df_next.set_index('Ngay')])
                 # Vẽ biểu đồ mới với dự đoán cho ngày kế tiếp
                plt.figure(figsize=(15, 5))
                plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')
                plt.plot(df_stock.index[:len(train_predict_plot)], train_predict_plot, label='Giá dự đoán train', color='green')
                plt.plot(df_stock.index[test_start_index:test_start_index + len(y_test_predict)], test_predict_plot[test_start_index:test_start_index + len(y_test_predict)], label='Giá dự đoán test', color='blue')
                plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá đóng cửa (VNĐ)')
                plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                plt.legend()
                # plt.show()
                # Lưu biểu đồ vào buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                data_uri = base64.b64encode(buffer.read()).decode()
                plt.close()

                # Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
                actual_closing_price = df_stock['GiaCuoiCung'].iloc[-2]
                comparison_data_final = pd.DataFrame({'Ngay': [next_date], 'GiaDuDoan': [y_next_predict[0][0]], 'GiaNgayTruoc': [actual_closing_price]})
                print(comparison_data_final)
                results.append({
                    'MaChungKhoan': stock_code,
                    'DoPhuHopTrain_LightGBM': train_r2_score,
                    'SaiSoTrungBinhTrain_LightGBM': train_mae,
                    'RMSETrain_LightGBM': train_rmse,
                    'PhanTramSaiSoTrain_LightGBM': float(train_mape),
                    'MSETrain_LightGBM': train_mse,
                    'DoPhuHopTest_LightGBM': test_r2_score,
                    'SaiSoTrungBinhTest_LightGBM': test_mae,
                    'RMSETest_LightGBM': test_rmse,
                    'PhanTramSaiSoTest_LightGBM': test_mape,
                    'MSETest_LightGBM': test_mse,
                    'DuDoanCuoiCung_LightGBM': float(y_next_predict[0][0]),
                    'GiaThucTeCuoi_LightGBM': comparison_data_final.to_dict(),
                    'chart': data_uri
                })
                print("135",results)
                return jsonify(results)
            else:
                # Chuẩn hóa dữ liệu
                scaler = MinMaxScaler(feature_range=(0, 1))
                train_data_scaled = scaler.fit_transform(train_data)
                test_data_scaled = scaler.transform(test_data)

                # Xây dựng dữ liệu đầu vào cho LightGBM
                X_train, y_train = [], []
                for i in range(60, len(train_data_scaled)):
                    X_train.append(train_data_scaled[i-60:i, 0])
                    y_train.append(train_data_scaled[i, 0])

                X_train, y_train = np.array(X_train), np.array(y_train)

                # Định nghĩa không gian tìm kiếm cho Bayesian Optimization
                param_space = {
                    'n_estimators': Integer(50, 200),  # Tổng số cây
                    'learning_rate': Real(0.001, 0.1, 'log-uniform'),  # Tốc độ học
                    'max_depth': Integer(1, 10),  # Độ sâu tối đa của cây
                    'reg_alpha': Real(0.001, 1.0, 'log-uniform'),  # L2 regularization alpha
                    'reg_lambda': Real(0.001, 1.0, 'log-uniform'),  # L1 regularization lambda
                    'subsample': Real(0.8, 1.0),  # Tỷ lệ mẫu
                    'colsample_bytree': Real(0.8, 1.0)  # Tỷ lệ cột
                }

                # Sử dụng Bayesian Optimization để tối ưu hóa các siêu tham số
                opt = BayesSearchCV(
                    estimator=LGBMRegressor(),
                    search_spaces=param_space,
                    scoring='neg_mean_absolute_error',
                    cv=5,
                    n_jobs=-1,
                    verbose=1
                )

                # Tìm kiếm siêu tham số tốt nhất
                opt.fit(X_train, y_train)

                # In ra các tham số tốt nhất và điểm số tốt nhất
                print(f'Mã cổ phiếu: {stock_code}')
                print("Best parameters found: ", opt.best_params_)
                print("Best CV score: ", -opt.best_score_)

                # Huấn luyện lại mô hình với các siêu tham số tốt nhất
                best_model = opt.best_estimator_
                best_model.fit(X_train, y_train)

                # Dự đoán trên tập train
                y_train_predict = best_model.predict(X_train)
                y_train_predict = scaler.inverse_transform(y_train_predict.reshape(-1, 1))

                # Xử lý dữ liệu test
                total_data = np.concatenate((train_data[-60:], test_data), axis=0)
                inputs = scaler.transform(total_data)
                X_test = []
                for i in range(60, len(inputs)):
                    X_test.append(inputs[i-60:i, 0])
                X_test = np.array(X_test)

                # Dự đoán trên tập test
                y_test = df_stock[df_stock.index > split_date]['GiaCuoiCung'].values  # giá thực
                y_test_predict = best_model.predict(X_test)
                y_test_predict = scaler.inverse_transform(y_test_predict.reshape(-1, 1))

                # Lập biểu đồ so sánh
                plt.figure(figsize=(24, 8))
                plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')

                # Chỉ số bắt đầu và kết thúc cho dự đoán train
                train_start_index = 60
                train_end_index = train_start_index + len(y_train_predict)
                train_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                train_predict_plot[:] = np.nan
                train_predict_plot[60:len(y_train_predict) + 60] = y_train_predict.reshape(-1)
                plt.plot(df_stock.index, train_predict_plot, label='Giá dự đoán train', color='green')

                # Chỉ số bắt đầu và kết thúc cho dự đoán test
                test_start_index = len(train_data)
                test_end_index = test_start_index + len(y_test_predict)
                test_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
                test_predict_plot[:] = np.nan
                test_predict_plot[test_start_index:test_end_index] = y_test_predict.reshape(-1)
                plt.plot(df_stock.index, test_predict_plot, label='Giá dự đoán test', color='blue')

                plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá đóng cửa (VNĐ)')
                plt.legend()
                # plt.show()

                print(f'Mã cổ phiếu: {stock_code}')
                train_r2_score = r2_score(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_mae = mean_absolute_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                train_rmse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict, squared=False)
                train_mse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
                
                test_r2_score = r2_score(y_test, y_test_predict)
                test_mape = mean_absolute_percentage_error(y_test, y_test_predict)
                test_mae = mean_absolute_error(y_test, y_test_predict)
                test_rmse = mean_squared_error(y_test, y_test_predict, squared=False)
                test_mse = mean_squared_error(y_test, y_test_predict)

                print('Độ phù hợp tập train:', train_r2_score)
                print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', train_mae)
                print('Phần trăm sai số tuyệt đối trung bình tập train:',train_mape)
                print('Root Mean Squared Error trên tập train (VNĐ):', train_rmse)
                print('Mean Squared Error trên tập train (VNĐ^2):', train_mse)

                print('Độ phù hợp tập test:', test_r2_score)
                print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', test_mae)
                print('Phần trăm sai số tuyệt đối trung bình tập test:', test_mape)
                print('Root Mean Squared Error trên tập test (VNĐ):', test_rmse)
                print('Mean Squared Error trên tập test (VNĐ^2):', test_mse)

                # Dự đoán cho ngày kế tiếp
                x_next = np.array([inputs[-60:, 0]])
                y_next_predict = best_model.predict(x_next)
                y_next_predict = scaler.inverse_transform(y_next_predict.reshape(-1, 1))

                # Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
                next_date = df_stock.index[-1] + pd.Timedelta(days=1)
                df_next = pd.DataFrame({'Ngay': [next_date], 'GiaCuoiCung': [y_next_predict[0][0]]})
                df_stock = pd.concat([df_stock, df_next.set_index('Ngay')])

                # Vẽ biểu đồ mới với dự đoán
                    # cho ngày kế tiếp
                plt.figure(figsize=(15, 5))
                plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')
                plt.plot(df_stock.index[:len(train_predict_plot)], train_predict_plot, label='Giá dự đoán train', color='green')
                plt.plot(df_stock.index[test_start_index:test_start_index + len(y_test_predict)], test_predict_plot[test_start_index:test_start_index + len(y_test_predict)], label='Giá dự đoán test', color='blue')
                plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
                plt.xlabel('Thời gian')
                plt.ylabel('Giá đóng cửa (VNĐ)')
                plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
                plt.legend()
                # plt.show()
                # Lưu biểu đồ vào buffer
                buffer = io.BytesIO()
                plt.savefig(buffer, format='jpg')
                buffer.seek(0)
                data_uri = base64.b64encode(buffer.read()).decode()
                plt.close(fig)

                # Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
                actual_closing_price = df_stock['GiaCuoiCung'].iloc[-2]
                comparison_data_final = pd.DataFrame({'Ngay': [next_date], 'GiaDuDoan': [y_next_predict[0][0]], 'GiaNgayTruoc': [actual_closing_price]})
                print(comparison_data_final)
                results.append({
                    'MaChungKhoan': stock_code,
                    'DoPhuHopTrain_LightGBM': train_r2_score,
                    'SaiSoTrungBinhTrain_LightGBM': train_mae,
                    'RMSETrain_LightGBM': train_rmse,
                    'PhanTramSaiSoTrain_LightGBM': float(train_mape),
                    'MSETrain_LightGBM': train_mse,
                    'DoPhuHopTest_LightGBM': test_r2_score,
                    'SaiSoTrungBinhTest_LightGBM': test_mae,
                    'RMSETest_LightGBM': test_rmse,
                    'PhanTramSaiSoTest_LightGBM': test_mape,
                    'MSETest_LightGBM': test_mse,
                    'DuDoanCuoiCung_LightGBM': float(y_next_predict[0][0]),
                    'GiaThucTeCuoi_LightGBM': comparison_data_final.to_dict(),
                    'chart': data_uri
                })
                print("135",results)
                return jsonify(results)
                
        elif algorithm == 'lstm':
            # Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            train_data_scaled = scaler.fit_transform(train_data)
            test_data_scaled = scaler.transform(test_data)

            # Xây dựng dữ liệu đầu vào cho XGBoost
            X_train, y_train = [], []
            for i in range(60, len(train_data_scaled)):
                X_train.append(train_data_scaled[i-60:i, 0])
                y_train.append(train_data_scaled[i, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)

            # Định nghĩa hàm mục tiêu cho Optuna
            # Định nghĩa không gian tìm kiếm cho Bayesian Optimization
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.001, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.001, 1.0),
                    'subsample': trial.suggest_uniform('subsample', 0.8, 1.0),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.8, 1.0)
                }
                model = XGBRegressor(**params, objective='reg:squarederror')
                tscv = TimeSeriesSplit(n_splits=10)
                scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')
                return -scores.mean()

            # Tạo một study và tối ưu hóa nó
            study = optuna.create_study(sampler=TPESampler(), direction='minimize')
            study.optimize(objective, n_trials=100)

            # In ra các siêu tham số tốt nhất
            print(f'Mã cổ phiếu: {stock_code}')
            print("Best parameters found: ", study.best_params)
            print("Best CV score: ", study.best_value)

            # Huấn luyện mô hình với siêu tham số tốt nhất
            best_model = XGBRegressor(**study.best_params, objective='reg:squarederror')
            best_model.fit(X_train, y_train)

            # Dự đoán trên tập train
            y_train_predict = best_model.predict(X_train)
            y_train_predict = scaler.inverse_transform(y_train_predict.reshape(-1, 1))

            # Xử lý dữ liệu test
            total_data = np.concatenate((train_data[-60:], test_data), axis=0)
            inputs = scaler.transform(total_data)
            X_test = []
            for i in range(60, len(inputs)):
                X_test.append(inputs[i-60:i, 0])
            X_test = np.array(X_test)

            # Dự đoán trên tập test
            y_test = df_stock[df_stock.index > split_date]['GiaCuoiCung'].values  # giá thực
            y_test_predict = best_model.predict(X_test)
            y_test_predict = scaler.inverse_transform(y_test_predict.reshape(-1, 1))

            # Lập biểu đồ so sánh
            plt.figure(figsize=(24, 8))
            plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')

            # Chỉ số bắt đầu và kết thúc cho dự đoán train
            train_start_index = 60
            train_end_index = train_start_index + len(y_train_predict)
            train_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
            train_predict_plot[:] = np.nan
            train_predict_plot[60:len(y_train_predict) + 60] = y_train_predict.reshape(-1)
            plt.plot(df_stock.index, train_predict_plot, label='Giá dự đoán train', color='green')

            # Chỉ số bắt đầu và kết thúc cho dự đoán test
            test_start_index = len(train_data)
            test_end_index = test_start_index + len(y_test_predict)
            test_predict_plot = np.empty(len(df_stock['GiaCuoiCung']))
            test_predict_plot[:] = np.nan
            test_predict_plot[test_start_index:test_end_index] = y_test_predict.reshape(-1)
            plt.plot(df_stock.index, test_predict_plot, label='Giá dự đoán test', color='blue')

            plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
            plt.xlabel('Thời gian')
            plt.ylabel('Giá đóng cửa (VND)')
            plt.legend()
            plt.show()

            # Đánh giá mô hình
            print(f'Mã cổ phiếu: {stock_code}')
            train_r2_score = r2_score(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
            train_mape = mean_absolute_percentage_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
            train_mae = mean_absolute_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
            train_rmse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict, squared=False)
            train_mse = mean_squared_error(scaler.inverse_transform(y_train.reshape(-1, 1)), y_train_predict)
            
            test_r2_score = r2_score(y_test, y_test_predict)
            test_mape = mean_absolute_percentage_error(y_test, y_test_predict)
            test_mae = mean_absolute_error(y_test, y_test_predict)
            test_rmse = mean_squared_error(y_test, y_test_predict, squared=False)
            test_mse = mean_squared_error(y_test, y_test_predict)

            print('Độ phù hợp tập train:', train_r2_score)
            print('Sai số tuyệt đối trung bình trên tập train (VNĐ):', train_mae)
            print('Phần trăm sai số tuyệt đối trung bình tập train:',train_mape)
            print('Root Mean Squared Error trên tập train (VNĐ):', train_rmse)
            print('Mean Squared Error trên tập train (VNĐ^2):', train_mse)

            print('Độ phù hợp tập test:', test_r2_score)
            print('Sai số tuyệt đối trung bình trên tập test (VNĐ):', test_mae)
            print('Phần trăm sai số tuyệt đối trung bình tập test:', test_mape)
            print('Root Mean Squared Error trên tập test (VNĐ):', test_rmse)
            print('Mean Squared Error trên tập test (VNĐ^2):', test_mse)

            # Dự đoán cho ngày kế tiếp
            x_next = np.array([inputs[-60:, 0]])
            y_next_predict = best_model.predict(x_next)
            y_next_predict = scaler.inverse_transform(y_next_predict.reshape(-1, 1))

            # Thêm dữ liệu dự đoán của ngày kế tiếp vào DataFrame
            next_date = df_stock.index[-1] + pd.Timedelta(days=1)
            df_next = pd.DataFrame({'Ngay': [next_date], 'GiaCuoiCung': [y_next_predict[0][0]]})
            df_stock = pd.concat([df_stock, df_next.set_index('Ngay')])

            # Vẽ biểu đồ mới với dự đoán cho ngày kế tiếp
            plt.figure(figsize=(15, 5))
            plt.plot(df_stock.index, df_stock['GiaCuoiCung'], label='Giá thực tế', color='red')
            plt.plot(df_stock.index[:len(train_predict_plot)], train_predict_plot, label='Giá dự đoán train', color='green')
            plt.plot(df_stock.index[test_start_index:test_start_index + len(y_test_predict)], test_predict_plot[test_start_index:test_start_index + len(y_test_predict)], label='Giá dự đoán test', color='blue')
            plt.scatter([next_date], [y_next_predict[0][0]], color='orange', label='Dự đoán ngày kế tiếp')
            plt.xlabel('Thời gian')
            plt.ylabel('Giá đóng cửa (VND)')
            plt.title(f'So sánh giá dự báo và giá thực tế cho mã cổ phiếu {stock_code}')
            plt.legend()
            plt.show()

            # Tạo DataFrame so sánh giá dự đoán với giá ngày cuối trong tập dữ liệu
            actual_closing_price = df_stock['GiaCuoiCung'].iloc[-2]
            comparison_data_final = pd.DataFrame({'Ngay': [next_date], 'GiaDuDoan': [y_next_predict[0][0]], 'GiaNgayTruoc': [actual_closing_price]})
            print(comparison_data_final)
    except KeyError as e:
        error_message = f'Missing key in request JSON: {str(e)}'
        return jsonify({'error': error_message}), 400  # Trả về mã lỗi 400 - Bad Request

    except Exception as e:
        error_message = f'Internal Server Error: {str(e)}'
        return jsonify({'error': error_message}), 500  # Trả về mã lỗi 500 - Internal Server Error

if __name__ == '__main__':
    app.run(debug=False)