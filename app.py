import joblib

def load_model(MaNganh, GioiTinh, NamSinh, NamNhapHoc, QueQuan, DanToc, GDTC1, TA1, CSLT, TBC_HK1, TBCTL_HK1, TBC_HK2, TBCTL_HK2):
    # columns = ['MaNganh', 'GioiTinh', 'NamSinh', 'NamNhapHoc', 'QueQuan', 'MaDanToc', 'GDTC 1',	'T.Anh 1', 'CSLT', 'TBC_HK1', 'TBCTL_HK1',	'TBC_HK2', 'TBCTL_HK2']
    scaler = joblib.load('model/scaler.pkl')  # Load the scaler object from file
    model = joblib.load('model/trained_model.pkl')    # Load the model object from file
    
    sample = dict()
    sample['MaNganh'] = int(MaNganh)
    sample['GioiTinh'] = int(GioiTinh)
    sample['NamSinh'] = int(NamSinh)
    sample['NamNhapHoc'] = int(NamNhapHoc)
    sample['ChenhTuoi'] = int(NamNhapHoc) - int(NamSinh) - 18
    sample['GDTC1'] = float(GDTC1)
    sample['TA1'] = float(TA1)
    sample['CSLT'] = float(CSLT)
    sample['TBC_HK1'] = float(TBC_HK1)
    sample['TBCTL_HK1'] = float(TBCTL_HK1)
    sample['TBC_HK2'] = float(TBC_HK2)
    sample['TBCTL_HK2'] = float(TBCTL_HK2)

    que_quan = QueQuan
    que_quan_cols = ['QueQuan_Laos', 'QueQuan_Quảng Ngãi', 'QueQuan_Tỉnh khác']
    for col in que_quan_cols:
        sample[col] = False

    if que_quan == 'Quảng Ngãi':
        sample['QueQuan_Quảng Ngãi'] = True
    elif que_quan == 'Laos':
        sample['QueQuan_Laos'] = True
    else:
        sample['QueQuan_Tỉnh khác'] = True

    dan_toc = DanToc
    dan_toc_cols = ['MaDanToc_HRE', 'MaDanToc_KH', 'MaDanToc_KOR', 'MaDanToc_LAO']
    for col in dan_toc_cols:
        sample[col] = False

    if dan_toc == 'HRE':
        sample['MaDanToc_HRE'] = True
    elif dan_toc == 'KH':
        sample['MaDanToc_KH'] = True
    elif dan_toc == 'KOR':
        sample['MaDanToc_KOR'] = True
    else:
        sample['MaDanToc_LAO'] = True

    df3 = pd.DataFrame(data = [sample])
    sample = df3.iloc[0:1]
    # sample = np.array(sample).reshape(1, -1)
    sample = scaler.transform(sample)
    return df3, model.predict(sample)[0]

from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
import pandas as pd
pd.set_option('display.expand_frame_repr', False)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', que_quan_list = ['Quảng Ngãi', 'Laos', 'Tỉnh khác'],
                           dan_toc_list = ['HRE', 'KH', 'KOR', 'LAO'])

@app.route('/getprediction', methods=['POST'])
def getprediction():
    MaNganh = request.form['MaNganh']
    GioiTinh = request.form['GioiTinh']
    NamSinh = request.form['NamSinh']
    NamNhapHoc = request.form['NamNhapHoc']
    QueQuan = request.form['QueQuan']
    DanToc = request.form['DanToc']
    GDTC1 = request.form['GDTC1']
    TA1 = request.form['TA1']
    CSLT = request.form['CSLT']
    TBC_HK1 = request.form['TBC_HK1']
    TBCTL_HK1 = request.form['TBCTL_HK1']
    TBC_HK2 = request.form['TBC_HK2']
    TBCTL_HK2 = request.form['TBCTL_HK2']
    df3, answer = load_model(MaNganh, GioiTinh, NamSinh, NamNhapHoc, QueQuan, DanToc, GDTC1, TA1, CSLT, TBC_HK1, TBCTL_HK1, TBC_HK2, TBCTL_HK2)
    print(df3)
    return render_template('index.html', ma_nganh = MaNganh, gioi_tinh = GioiTinh, nam_sinh = NamSinh, nam_nhap_hoc = NamNhapHoc,
                           que_quan = QueQuan, dan_toc = DanToc, que_quan_list = ['Quảng Ngãi', 'Laos', 'Tỉnh khác'],
                           dan_toc_list = ['HRE', 'KH', 'KOR', 'LAO'], gdtc1 = GDTC1, ta1 = TA1, cslt = CSLT,
                           tbc_hk1 = TBC_HK1, tbctl_hk1 = TBCTL_HK1, tbc_hk2 = TBC_HK2, tbctl_hk2 = TBCTL_HK2,
                           answer = answer)
if __name__ == "__main__":
    app.run()  # Run the Flask app
