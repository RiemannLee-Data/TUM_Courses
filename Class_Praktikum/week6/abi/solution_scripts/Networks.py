import datetime
import numpy as np
import pandas as pd
import keras
import locale
import pickle
import joblib
from keras.models import Sequential
from keras.callbacks import History 
from keras import backend as K
from keras.layers import Dense, LSTM, Flatten, Embedding
from fbprophet import Prophet
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelBinarizer
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.api import VAR as var
import optuna
import tensorflow as tf
import tensorflow_addons as tfa
from keras.utils.vis_utils import plot_model

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
class LSTM():

    def __init__(self, df, train, test, input_columns, output_column):
        self.input_columns = input_columns
        self.output_column = output_column
        self.train_var = train
        self.test_var = test

        

    def create_dataset(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def scale(self):
        train = self.train_var
        test = self.test_var
        #scale features
        input_transformer = RobustScaler()
        #scale trade value
        output_transformer = RobustScaler()
        input_transformer = input_transformer.fit(train[self.input_columns].to_numpy())
        output_transformer = output_transformer.fit(train[[self.output_column]])
        train.loc[:, self.input_columns] = input_transformer.transform(train[self.input_columns].to_numpy())
        train[self.output_column] = output_transformer.transform(train[[self.output_column]])
        test.loc[:, self.input_columns] = input_transformer.transform(test[self.input_columns].to_numpy())
        test[self.output_column] = output_transformer.transform(test[[self.output_column]])
        self.train_var = train
        self.test_var = test

    def get_callbacks(self):
        return[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=40),
            keras.callbacks.History()
        ]    
    def train(self, hparams, filename):
        X_train, y_train = self.create_dataset(self.train_var, self.train_var[self.output_column], hparams["time_steps"])
        #X_test, y_test = self.create_dataset(test, test[output_column], hparams["time_steps"])
        model = keras.Sequential()
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(units=hparams["neurons"], activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]))))
        model.add(keras.layers.Dropout(rate=hparams["dropout"]))
        model.add(keras.layers.Dense(units=hparams["layers"]))
        optimizer = keras.optimizers.Adam(clipvalue=1.0, name='adam', learning_rate=hparams['lr'])
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        #model.summary()
        history = model.fit(X_train, y_train, callbacks=self.get_callbacks(),
        epochs=hparams["epochs"], batch_size=hparams["batch"], validation_split=0.2, shuffle=False)
        hist_df = pd.DataFrame(history.history)
        n = hparams["neurons"]
        hist_csv_file = f'{filename}/history_{n}.csv'
        with open(hist_csv_file, mode='w') as f:
          hist_df.to_csv(f)
        self.model = model
        
        return model, hist_df

    def test(self, time_steps):
        X_test, y_test = self.create_dataset(self.test_var, self.test_var[self.output_column], time_steps)
        y_pred = self.model.predict(X_test)
        return y_test, y_pred

class Facebook():
    def prepare(self, df, columns_to_remove, output_column, year_to_pred):
        data = df.drop(columns=columns_to_remove)
        data = data.groupby(['Time','Category Code']).agg({output_column:'mean'}).reset_index().set_index('Time')
        main_group = data.groupby(['Category Code'])
        data = data[data.index.year<year_to_pred]
        data = data.reset_index().rename(columns={'Time': 'ds', output_column:'y'})
        data["ds"] = data["ds"].apply(lambda x : x.to_timestamp())
        grouped = data.groupby(['Category Code'])
        return main_group, grouped

    def predict(self, data,  columns_to_remove, output_column,periods=1, freq='Y', year_to_pred =2019):
        final = pd.DataFrame()
        predictions = pd.DataFrame(columns=list(['Time','group', 'y_true', 'y_pred', 'mae']))
        main_group, grouped = self.prepare(data, columns_to_remove, output_column, year_to_pred)
        for g in grouped.groups:
            group = grouped.get_group(g)
            curr = main_group.get_group(g)
            y_true = curr[curr.index.year==year_to_pred][output_column].values
            indexes = curr[curr.index.year==year_to_pred].index.values
            #print(indexes)
            #print(y_true[0])
            m = Prophet()
            m.fit(group)
            future = m.make_future_dataframe(periods=periods, freq=freq)
            forecast = m.predict(future)
            y_pred = forecast['yhat'].tail(12).values
            for i in range(12):
                mae = mean_absolute_error([y_true[i]], [y_pred[i]])
                predictions = predictions.append({'Time':indexes[i] ,'group': str(g), 'y_true':y_true[i], 'y_pred':y_pred[i], 'mae':mae}, ignore_index=True)
            #print('MAE: %.3f' % mae)
            #print(forecast.tail())
            forecast = forecast.rename(columns={'yhat': 'yhat_'+str(g)})
            final = pd.merge(final, forecast.set_index('ds'), how='outer', left_index=True, right_index=True)
        final = final[['yhat_' + str(g) for g in grouped.groups.keys()]]
        return predictions, final

class ANN():
    def __init__(self, df, output_column, train_raw, val_raw, test_raw, continuous_cols):
        self.output_column = output_column
        self.train = train_raw
        self.val = val_raw
        self.test = test_raw
        self.continous = continuous_cols
        self.df = df
        self.train_size = train_raw.size
        #keras.backend.set_epsilon(1)
        

    def scale(self, train, test, val):
        #scale features
        cs = RobustScaler()
        
        return (trainX, valX, testX)
        

    def prepare(self):
        self.cs = RobustScaler()
        self.trainX = self.cs.fit_transform(self.train[self.continous])
        self.valX = self.cs.transform(self.val[self.continous])
        self.testX = self.cs.transform(self.test[self.continous])
        #self.trainX, self.valX, self.testX = self.scale(self.train, self.test, self.val)
        self.trainY = self.cs.fit_transform(self.train[[self.output_column]])
        self.testY = self.cs.transform(self.test[[self.output_column]])
        self.valY = self.cs.transform(self.val[[self.output_column]])
        return self.trainX, self.trainY, self.valX, self.valY
   
    def create_model(self,  neurons=256):
        model = Sequential()
        initializer = keras.initializers.HeNormal()
        model.add(Dense(neurons, input_dim=self.trainX.shape[1], name='InputLayer', activation='relu', kernel_initializer=initializer))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.36))
        
        neurons = neurons/2
        model.add(Dense(neurons, kernel_initializer=initializer, activation='relu', name='H1'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.37))
        
        neurons = neurons/2
        model.add(Dense(neurons, kernel_initializer=initializer, activation='relu', name='H2'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.16))
        
        model.add(Dense(1, activation='linear', name="OutputLayer", kernel_initializer=initializer))
        self.model = model
        return model
    def get_callbacks(self):
        return[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
            keras.callbacks.History()
        ]

    def train_trials(self, n_trials):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params

    def objective(self, trial):
        K.clear_session()
        model = Sequential()
        neurons = 512
        initializer = keras.initializers.HeNormal()
        initializer = keras.initializers.HeNormal()
        model.add(Dense(neurons, input_dim=self.trainX.shape[1], name='InputLayer', activation='relu', kernel_initializer=initializer))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.36))
        
        neurons = neurons/2
        model.add(Dense(neurons, kernel_initializer=initializer, activation='relu', name='H1'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.37))
        
        neurons = neurons/2
        model.add(Dense(neurons, kernel_initializer=initializer, activation='relu', name='H2'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=0.16))
        
        model.add(Dense(neurons, kernel_initializer=initializer, activation='relu', name='H3'))
        model.add(Dense(1, activation='linear', name="OutputLayer", kernel_initializer=initializer))
        trainY= self.trainY
        testY =self.testY
        trainX = self.trainX
        model = self.model
        opt = Adam(lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True), decay=trial.suggest_float('decay', 1e-5, 0.1, log=True))
        model.compile(loss='mae', optimizer=opt)
        history = model.fit(x=trainX, y=trainY, validation_data=(self.valX, self.valY), epochs=15, verbose=2,
                            batch_size=trial.suggest_int('batchsize',68, 512, step=12))
        return history.history["val_loss"][-1]
       
        

    

    
    def train_model(self, hparams, filename, historyname):
        opt = Adam(lr=hparams['lr'], decay=hparams['decay_rate'])
        #opt = tfa.optimizers.MovingAverage(opt)
        #keras.backend.set_epsilon(1e-7)
        print("[INFO] processing data")
        trainY= self.trainY
        testY =self.testY
        trainX = self.trainX
        model = self.model
        model.compile(loss='mae', optimizer=opt, metrics=[root_mean_squared_error])
        model.summary()
        print("[INFO] training model...")
        history = model.fit(x=trainX, y=trainY, validation_data=(self.valX, self.valY), verbose=2,
	                epochs=hparams["epochs"], batch_size=hparams["batch"], callbacks=self.get_callbacks())
        model.save(filename)
        hist_df = pd.DataFrame(history.history)
        hist_csv_file = f'{filename}/{historyname}.csv'
        with open(hist_csv_file, mode='w') as f:
          hist_df.to_csv(f)
        self.model = model
        plot_model(model, to_file=f'{filename}/model_archi.png', show_shapes=True, show_layer_names=True)
        return history, model
        
    def predict(self):
        model = self.model
        print("[INFO] predicting trade value...")
        preds = model.predict(self.testX)
        #print(preds)
        df = self.df
        # compute the difference between the *predicted*  *actual* , then compute the percentage difference and
        # the absolute percentage difference
        diff = preds.flatten() - self.testY
        percentDiff = (diff / self.testY) * 100
        absPercentDiff = np.abs(percentDiff)
        # compute the mean and standard deviation of the absolute percentage difference
        mean = np.mean(absPercentDiff)
        std = np.std(absPercentDiff)
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        print("[INFO] avg. trade value: {}, std trade value: {}".format(
	        locale.currency(df[self.output_column].mean(), grouping=True),
	        locale.currency(df[self.output_column].std(), grouping=True)))
        print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
        return self.cs.inverse_transform(preds)
        


class VAR():
    def __init__(self, df, output_column, columns_to_drop=None):
        self.df = df
        self.output_column = output_column
        self.columns_to_drop = columns_to_drop
        self.train_var, self.test_var = self.prepare()
    
    def prepare(self):
        data = self.df
        if(self.columns_to_drop != None):
            data = data.drop(columns=self.columns_to_drop)
        data = data.groupby(['Time','Reporter', 'Category Code']).agg({self.output_column:'mean', 'Population':'last'}).reset_index().set_index('Time')
        train = data[data.index<"2019"]
        test = data[data.index >= "2019"]
        print(train.size, test.size)
        return train, test

    def adf_test(self, ts, signif=0.05):
        dftest = adfuller(ts, autolag='AIC')
        adf = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags','# Observations'])
        for key,value in dftest[4].items():
            adf['Critical Value (%s)'%key] = value
        print (adf)
        p = adf['p-value']    
        if p <= signif:
            print(f" Series is Stationary")
        else:
            print(f" Series is Non-Stationary")
    
    def show_lag_orders(self, lags):
        train_var = self.train_var
        grouped = train_var.groupby(['Category Code'])
        group = grouped.get_group(next(iter(grouped.groups)))
        group = group.drop(columns={'Category Code'})
        model = var(group)
        x = model.select_order(maxlags=max(lags))
        print(x.summary())
        for i in lags:
            result = model.fit(i)
            print('Lag Order =', i)
            print('AIC : ', result.aic)
            print('BIC : ', result.bic)
            print('FPE : ', result.fpe)
            print('HQIC: ', result.hqic, '\n')

    def train(self, opt_lag, grouped):
        train_var = self.train_var
        train_results = {}
        for g in grouped.groups:
            group = grouped.get_group(g)
            group = group.drop(columns={'Reporter'})
            group = group.loc[:, (group != group.iloc[0]).any()]
            model = var(group)
            result = model.fit(maxlags=opt_lag, ic='aic',trend='c')
            train_results[g] = result
            out = durbin_watson(result.resid)
            for col, val in zip(group.columns, out):
                print((col), ':', round(val, 2))

        return train_results

    def forecast(self, steps, grouped_train, grouped_test, train_results, products_mapping):
        forecasts = pd.DataFrame()
        test_var = self.test_var
        unique_index = grouped_test.get_group(next(iter(grouped_test.groups))).index.unique()
        for key, value in train_results.items():
            lag_order = value.k_ar
            group = grouped_train.get_group(key)
            group = group.drop(columns={'Reporter'})
            group = group.loc[:, (group != group.iloc[0]).any()]
            test_group = grouped_test.get_group(key)
            forecast_input = group.values[-lag_order:]
            fc = value.forecast(y=forecast_input, steps=steps)
            index = unique_index[:lag_order].values.tolist()
            df_forecast = pd.DataFrame(fc, index=index, columns=group.columns)
            df_forecast['Product'] = products_mapping[str(key)]
            forecasts = forecasts.append(df_forecast)

        return forecasts

        
        




