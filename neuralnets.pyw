from connbd import connectdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QDialog,QTableWidgetItem,QMessageBox,QWidget
from PyQt5 import uic
import seaborn as sns
import pickle as pk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn import metrics,ensemble
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from LogReg import LogisticReg

class NeuralNets(QDialog):
	def __init__(self,fn=None,parent=None):
		super(NeuralNets,self).__init__(parent,flags=Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowCloseButtonHint)
		uic.loadUi("neuralnets.ui",self)
		self.loadData()
		
		self.btnperceptron.clicked.connect(self.Perceptron)
		self.btnmlp.clicked.connect(self.MLP)
		self.btnnn.clicked.connect(self.trainingacc)
		#Defining Hyperparameters
		self.inputLayerSize = 1
		self.outputLayerSize = 1
		self.hiddenLayerSize = 100
		self.epochs = 10000
		self.minibatches = 1
		self.learning_rate = 0.01
		self.Lambda = 0.001 #regularization parameter
		#Weights (parameters)
		self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

		

	def loadData(self):
		#conn=connectdb()
		#sql = "SELECT clientes.ruccliente as idclient,factxcob.cliente as cliente,str_to_date(factxcob.fecha,'%d/%m/%Y') as fechadoc,factxcob.importe as importe,max(str_to_date(cobranza.fecha, '%d/%m/%Y')) as fechacob FROM factxcob join clientes on factxcob.cliente=clientes.razonsocial join facturas on factxcob.nfact = facturas.numfact join cobranza on factxcob.nfact=cobranza.nfact where estado='c' and facturas.pago='Credito' group by factxcob.nfact"
		titles='Idi IdClient Client DocDate Amount PayDate'.split()
		self.dfcredit=pd.read_excel('../data/credit_data.xls')
		#self.dfcredit=pd.read_sql(sql,con=conn)
		self.dfcredit.columns=titles
	
		self.dfcredit['NumDays'] = (self.dfcredit['PayDate'] - self.dfcredit['DocDate']).dt.days 
		self.dfcredit['TypeCli'] = self.dfcredit['Client'].apply(self.findtype)
		self.dfcredit['Class'] = self.dfcredit['NumDays'].apply(lambda x: 1 if int(x)<=10 else(2 if int(x)>10 and int(x)<=30 else 0))
		#conn.close()

		#making datagrid view/table
		self.dgvnn.setColumnCount(9)
		self.dgvnn.setHorizontalHeaderLabels(['Idi','IdClient','Client','DocDate','Amount','PayDate','NumDays','TypeCli','Class'])

		for i in range(len(self.dfcredit)):
			row = self.dfcredit.iloc[i]
			self.dgvnn.insertRow(i)
			Idi = QTableWidgetItem(str(row[0]))
			idclient = QTableWidgetItem(str(row[1]))
			client = QTableWidgetItem(str(row[2]))
			docdate = QTableWidgetItem(str(row[3]))
			amount = QTableWidgetItem(str(row[4]))
			paydate = QTableWidgetItem(str(row[5]))
			numdays = QTableWidgetItem(str(row[6]))
			typecli = QTableWidgetItem(str(row[7]))
			classcli = QTableWidgetItem(str(row[8]))
			self.dgvnn.setItem(i, 0, Idi)
			self.dgvnn.setItem(i, 1, idclient)
			self.dgvnn.setItem(i, 2, client)
			self.dgvnn.setItem(i, 3, docdate)
			self.dgvnn.setItem(i, 4, amount)
			self.dgvnn.setItem(i, 5, paydate)
			self.dgvnn.setItem(i, 6, numdays)
			self.dgvnn.setItem(i, 7, typecli)
			self.dgvnn.setItem(i, 8, classcli)

		self.dgvnn.resizeColumnsToContents()
		
	def findtype(self,dfvalue):
		if 'SAC' in dfvalue.replace('.','') or 'SA' in dfvalue.replace('.',''):
			return 1
		elif 'EIRL' in dfvalue.replace('.','') or 'SRL' in dfvalue.replace('.',''):
			return 0
		else:
			return 2
			

	def Perceptron(self):
		lr = LogisticReg()
		self.loadData
		y = self.dfcredit['Class']
		X = self.dfcredit[['Amount','TypeCli']]

		X_train_std, X_test_std, y_train, y_test = lr.traindata(X,y)

		cls = Perceptron(max_iter=10000, eta0=0.0001,
				penalty='l2',shuffle=True,
				random_state=101)
		cls.fit(X_train_std, y_train)

		y_pred = cls.predict(X_test_std)
		report = classification_report(y_test,y_pred)
		accscore = accuracy_score(y_test,y_pred)
		self.lblcr.setText('Accuracy Score: %.2f' % accscore + '\n'+ 'Classification Report'+'\n'+report)

		crosstab = pd.crosstab(y_test,
								y_pred,
								rownames=['Actual'],
								colnames=['Predicted'])
		self.lblcrosstab.setText(str(crosstab))
		x_tot = np.vstack((X_train_std, X_test_std))
		y_tot = np.hstack((y_train, y_test))
		lr.plot_regions(X=x_tot,y=y_tot,classifier=cls) 
		
	def MLP(self):
		lr = LogisticReg()
		self.loadData
		y = self.dfcredit['Class']
		X = self.dfcredit[['Amount','TypeCli']]

		X_train_std, X_test_std, y_train, y_test = lr.traindata(X,y)

		cls = MLPClassifier(max_iter=10000, hidden_layer_sizes=(50,2),
				activation='relu', solver='sgd', batch_size=50,
				learning_rate_init=0.001, learning_rate='constant',
			 	shuffle=True,random_state=101)
		cls.fit(X_train_std, y_train)

		filename = 'MLP_model.sav'
		pk.dump(cls, open(filename, 'wb'))

		loaded_model = pk.load(open(filename, 'rb'))

		y_pred = loaded_model.predict(X_test_std)
		result = loaded_model.score(X_test_std,y_test)
		
		report = classification_report(y_test,y_pred)
		accscore = accuracy_score(y_test,y_pred)
		self.lblcr.setText('Accuracy Score: %.2f' % accscore + '\n'+ 'Classification Report'+'\n'+report)

		crosstab = pd.crosstab(y_test,
								y_pred,
								rownames=['Actual'],
								colnames=['Predicted'])
		self.lblcrosstab.setText(str(crosstab))
		x_tot = np.vstack((X_train_std, X_test_std))
		y_tot = np.hstack((y_train, y_test))
		lr.plot_regions(X=x_tot,y=y_tot,classifier=cls) 	

	def loadfunction(self):
		X = np.linspace(-2,2,200)
		y = np.sin(X*np.pi/4)
		scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
		y_resh= y.reshape(-1,1)
		y_std = scaler.fit_transform(y_resh)
		return X.reshape(200,1),y_std.reshape(200,1)

	def testdata(self):
		X_test = np.sort(np.random.uniform(-2, 2,(200,1)),axis=0)
		y_test = np.sin(X_test*np.pi/4)
		scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
		y_test_std = scaler.fit_transform(y_test)		
		return X_test, y_test_std
		
	def minmaxScaler(self,X_train,X_test,y_train,y_test):
		scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
		X_train_std = scaler.fit_transform(X_train)
		X_test_std = scaler.fit_transform(X_test)
		y_train_std = scaler.fit_transform(y_train)
		y_test_std = scaler.fit_transform(y_test)
		return X_train_std, X_test_std, y_train_std,y_test_std


	def stdScaler(self,X_train,X_test,y_train,y_test):
		_scaler = StandardScaler()
		X_train_std = _scaler.fit_transform(X_train)
		y_train_std = _scaler.fit_transform(y_train)
		X_test_std = _scaler.transform(X_test)
		y_test_std = _scaler.transform(y_test)     	
		return X_train_std, y_train_std, X_test_std, y_test_std	

	def relu(self,z):
		return np.maximum(0,z) #z * (z > 0)

	def reluPrime(self, z):
		return 1. * (z > 0)		

	def sigmoid(self,z):
		return 1/(1+np.exp(-z))

	def sigmoidPrime(self,z):
		#Derivative of sigmoid function
		return np.exp(-z)/((1+np.exp(-z))**2)		

	def forward(self, X):
	    #Propagate inputs through network
	    self.z2 = np.dot(X, self.W1)
	    self.a2 = self.sigmoid(self.z2)
	    self.z3 = np.dot(self.a2, self.W2)
	    yHat = self.sigmoid(self.z3) 
	    return yHat
	    

	def costFunctionPrime(self, X, y):
		#Compute derivative with respect to W1 and W2 for a given X and y:
		self.yHat = self.forward(X)
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)/X.shape[0] + self.Lambda*self.W2
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)/X.shape[0] + self.Lambda*self.W1  
        
		return dJdW1, dJdW2

	def costFunction(self, X, y):
		#Compute cost for given X,y, use weights already stored in class.
		self.yHat = self.forward(X)
		J = 0.5*np.sum((y-self.yHat)**2)/X.shape[0] + (0.5*self.Lambda)*(np.sum(self.W1**2)+np.sum(self.W2**2))
		return J
	
	def fit(self, X, y):
		self.cost = []
		self.costtest = []
		X_data, y_data = X.copy(), y.copy()
		X_test, y_test = self.testdata()
		X_std, y_std, X_test_std, y_test_std = X_data, y_data,X_test,y_test #self.minmaxScaler(X_data, y_data,X_test,y_test)
		
		for i in range(self.epochs):			
			mini = np.array_split(range(y_test.shape[0]), self.minibatches)	
			for idx in mini:
				ypred = self.forward(X_test_std[idx])
				
				cost = self.costFunction(X_std[idx], y_std[idx])
				self.cost.append(cost)

				costtest = self.costFunction(X_test_std[idx], y_test_std[idx])
				self.costtest.append(costtest)

			#	if self.epochs %100 == 0:
			#		print('cost: %.2f' % cost)
				grad1, grad2 = self.costFunctionPrime(X_std[idx],y_std[idx])		
				self.W2 -= self.learning_rate * grad2
				self.W1 -= self.learning_rate * grad1		

		fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(6,6))
		
		ax[0].plot(y_data,lw=2, color='green',label='Data')
		ax[0].plot(ypred,lw=2, color='red',label='Predictions')
		ax[0].legend(loc='best')
		ax[0].set_ylabel('Y values')
		ax[0].set_xlabel('X values')
		
		ax[1].plot(self.cost,lw=2, color='blue',label='TrainingData')
		ax[1].plot(self.costtest,lw=2, color='red',label='TestData')
		ax[1].legend(loc='best')
		ax[1].set_ylabel('Cost Function')
		ax[1].set_xlabel('Cycles')
			
		plt.show()	

	def trainingacc(self):
		X, y = self.loadfunction()
		self.fit(X,y)
					