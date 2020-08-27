from connbd import connectdb
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import seaborn as sns
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QDialog,QTableWidgetItem
from PyQt5 import uic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report


class LogisticReg(QDialog):
	def __init__(self,fn=None,parent=None):
		super(LogisticReg,self).__init__(parent,flags=Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowCloseButtonHint)
		uic.loadUi("logisticreg.ui",self)
		self.loadData()
		self.btnlr.clicked.connect(self.RegLog)
		self.btndata.clicked.connect(self.DataAnalysis)

	def loadData(self):
		#conn=connectdb()
		#sql = "SELECT clientes.ruccliente as idclient,factxcob.cliente as cliente,str_to_date(factxcob.fecha,'%d/%m/%Y') as fechadoc,factxcob.importe as importe,max(str_to_date(cobranza.fecha, '%d/%m/%Y')) as fechacob FROM factxcob join clientes on factxcob.cliente=clientes.razonsocial join facturas on factxcob.nfact = facturas.numfact join cobranza on factxcob.nfact=cobranza.nfact where estado='c' and facturas.pago='Credito' group by factxcob.nfact"
		titles='Idi IdClient Client DocDate Amount PayDate'.split()
		#self.dfcredit=pd.read_sql(sql,con=conn)
		#self.dfcredit.to_excel('../data/credit_data.xls')
		self.dfcredit=pd.read_excel('../data/credit_data.xls')
		self.dfcredit.columns=titles
	
		self.dfcredit['NumDays'] = (self.dfcredit['PayDate'] - self.dfcredit['DocDate']).dt.days 
		self.dfcredit['TypeCli'] = self.dfcredit['Client'].apply(self.findtype)
		self.dfcredit['Class'] = self.dfcredit['NumDays'].apply(lambda x: 1 if int(x)<=10 else(2 if int(x)>10 and int(x)<=30 else 0))
		#conn.close()

		#making datagrid view/table
		self.dgvLR.setColumnCount(9)
		self.dgvLR.setHorizontalHeaderLabels(['IdClient','Client','DocDate','Amount','PayDate','NumDays','TypeCli','Class','#'])

		for i in range(len(self.dfcredit)):
			row = self.dfcredit.iloc[i]
			self.dgvLR.insertRow(i)
			idi = QTableWidgetItem(str(row[0]))
			idclient = QTableWidgetItem(str(row[1]))
			client = QTableWidgetItem(str(row[2]))
			docdate = QTableWidgetItem(str(row[3]))
			amount = QTableWidgetItem(str(row[4]))
			paydate = QTableWidgetItem(str(row[5]))
			numdays = QTableWidgetItem(str(row[6]))
			typecli = QTableWidgetItem(str(row[7]))
			classcli = QTableWidgetItem(str(row[8]))
			self.dgvLR.setItem(i, 0, idi)
			self.dgvLR.setItem(i, 0, idclient)
			self.dgvLR.setItem(i, 1, client)
			self.dgvLR.setItem(i, 2, docdate)
			self.dgvLR.setItem(i, 3, amount)
			self.dgvLR.setItem(i, 4, paydate)
			self.dgvLR.setItem(i, 5, numdays)
			self.dgvLR.setItem(i, 6, typecli)
			self.dgvLR.setItem(i, 7, classcli)


		self.dgvLR.resizeColumnsToContents()	

	def findtype(self,dfvalue):
		if 'SAC' in dfvalue.replace('.','') or 'SA' in dfvalue.replace('.',''):
			return 1
		elif 'EIRL' in dfvalue.replace('.','') or 'SRL' in dfvalue.replace('.',''):
			return 0
		else:
			return 2	
			

	def traindata(self,X,y):
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
		X_train_std, X_test_std = self.stdScaler(X_train,X_test)

		return X_train_std, X_test_std, y_train, y_test

	def stdScaler(self,X_train,X_test):
		sc = StandardScaler()
		sc.fit(X_train)
		X_train_std = sc.transform(X_train)
		X_test_std = sc.transform(X_test)
		
		x1 = self.txttype.text()
		x2 = self.txtamount.text()
	
		if x1 != "" and x2 != "":
			x1 = int(self.txttype.text())
			x2 = float(self.txtamount.text())	
			self.X_newdata = sc.transform([[x1,x2]])
	
		else:
			return X_train_std, X_test_std


		return X_train_std, X_test_std
		
	def RegLog(self):
		lr = LogisticRegression(penalty='l2',C=.0001, random_state=0)
		y = self.dfcredit['Class']
		X = self.dfcredit[['Amount','TypeCli']]

		X_train, X_test, y_train, y_test = self.traindata(X,y)
		lr.fit(X_train,y_train)
		y_pred = lr.predict(X_test)
		report = classification_report(y_test,y_pred)
		accscore = accuracy_score(y_test,y_pred)
		self.lblcr.setText('Accuracy Score: %.2f' % accscore + '\n'+ 'Classification Report'+'\n'+report)
		crosstab = pd.crosstab(y_test,
								y_pred,
								rownames=['Actual'],
								colnames=['Predicted'])
		self.lbltab.setText(str(crosstab))
		x_tot = np.vstack((X_train, X_test))
		y_tot = np.hstack((y_train, y_test))
		self.plot_regions(X=x_tot,y=y_tot,classifier=lr)

		if self.txttype.text() != "" and self.txtamount.text() != "":
			self.lblclass.setText(str(lr.predict(self.X_newdata)))
			self.lblprob.setText(str(np.around(lr.predict_proba(self.X_newdata),decimals=2)))
		else:
			pass	
	
	def plot_regions(self,X, y, classifier):
		markers = ('x','>','s')
		colors = ('red','blue','yellow','green')
		cmap = ListedColormap(colors[:len(np.unique(y))])
		res = 0.02
		#Plot regions
		x1min, x1max = X[:,0].min() -1, X[:,0].max() + 1
		x2min, x2max = X[:,1].min() -1, X[:,1].max() + 1	
		xx1, xx2 = np.meshgrid(np.arange(x1min,x1max,res),np.arange(x2min,x2max,res))

		output = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		output = output.reshape(xx1.shape)
		plt.figure(figsize=(8,8))
		plt.pcolormesh(xx1,xx2, output, cmap=plt.cm.Accent)
	
		#PLOT ALL SAMPLES
		for index, item in enumerate(np.unique(y)):
			plt.scatter(x=X[y == item, 0], y=X[y == item, 1],alpha=0.8, c=cmap(index),
				marker=markers[index], label=item)

		plt.xlabel('Amount in Thousand')
		plt.ylabel('Size of Company')		
		plt.xlim(xx1.min(),xx1.max())
		plt.ylim(xx2.min(),xx2.max())
		plt.legend(loc='upper center',bbox_to_anchor=(0.5, 1.05),
          ncol=3,fancybox=True, shadow=True)
		plt.show()
	
	def DataAnalysis(self):
		sns.set_style('whitegrid')
		dataplot = self.dfcredit[['Class','Amount','TypeCli']]
		sns.pairplot(dataplot,hue='Class',markers=["o", "s", "D"],diag_kind="kde",palette='husl',size=3)
		plt.show()		