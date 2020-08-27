from connbd import connectdb
import pandas as pd
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QDialog,QTableWidgetItem,QMessageBox,QWidget
from PyQt5 import uic
from itertools import combinations, permutations
import pickle as pk


class MBA(QDialog):
	def __init__(self,fn=None,parent=None):
		super(MBA,self).__init__(parent,flags=Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowCloseButtonHint)
		uic.loadUi("mba.ui",self)
		self.loadData()

	def loadData(self):
		#conn=connectdb()
		#sql = "SELECT fecha,numfact,codigo as code from facturas group by numfact, codigo"
		#self.df=pd.read_sql(sql,con=conn)
		#conn.close()
		self.df=pd.read_pickle('transactions_data.sav')
		#SHOWING DATA ON TABLE
		self.tablemba.setColumnCount(3)
		self.tablemba.setHorizontalHeaderLabels(['Date','TicketNo','Code'])
		for i in range(len(self.df)):
			row = self.df.iloc[i]
			self.tablemba.insertRow(i)
			date = QTableWidgetItem(str(row[0]))
			ticket = QTableWidgetItem(str(row[1]))
			code = QTableWidgetItem(str(row[2]))
			self.tablemba.setItem(i, 0, date)
			self.tablemba.setItem(i, 1, ticket)
			self.tablemba.setItem(i, 2, code)
			
		self.tablemba.resizeColumnsToContents()

		#FIRST STEP: CALCULATING INITIAL SUPPORT
		supportp = 0.15
		trans = self.df['numfact'].unique()
		numtrans = len(self.df['numfact'].unique())
		dfcode = self.df['code'].unique()
		
		groups = self.df.groupby('numfact').groups
		# groups form: dictionary(unique value, index positions)
		listcodes = []
		for i in groups:
			codes = []
			for j in groups[i]:
				item = self.df['code'][j] 
				codes.append(item)
			listcodes.append([i,codes])

		l_ini = []
		l_codes = []
		for n in range(len(dfcode)):
			count = 0
			for i in range(len(self.df)):
				if dfcode[n] == self.df['code'][i]:
					count += 1
			self.support = count / numtrans
			if self.support > supportp:
				l_ini.append([dfcode[n],round(self.support,2)])
				l_codes.append(dfcode[n])
		
		self.lblini.setText(str(l_ini))	
		

		#SECOND STEP: FINDING FREQUENT ITEMSETS
		l2comb = []
		dfmodel2 = pd.DataFrame()
		l2strcomb = []
		l2support = []
		l2confidence = []
		l2lift = []
		for s in range(2,len(l_codes)):
			for comb in combinations(l_codes, s):
						
				self.support = self.getsupport_list(comb,listcodes)
							
				if self.support > 0.12:
					l2comb.append(comb)
					self.confidence = self.support / self.getsupport_list(comb[:-1],listcodes)
					self.lift = self.confidence / self.getsupport_list([comb[-1]],listcodes)
					l2support.append(round(self.support,2))
					l2confidence.append(round(self.confidence,2))
					l2lift.append(round(self.lift,2))
					str_comb = ','.join(comb)
					l2strcomb.append(str_comb)
							
		dfmodel2['Combination'] = l2strcomb
		dfmodel2['Support'] = l2support
		dfmodel2['Confidence'] = l2confidence
		dfmodel2['Lift'] = l2lift
		dfmodel2.titles = ['Combination','Support','Confidence','Lift']

		self.tablecombinations.setColumnCount(4)
		self.tablecombinations.setHorizontalHeaderLabels(['Combination','Support','Confidence','Lift'])
		for i in range(len(dfmodel2)):
			row = dfmodel2.iloc[i]
			self.tablecombinations.insertRow(i)
			comb = QTableWidgetItem(str(row[0]))
			supp = QTableWidgetItem(str(row[1]))
			conf = QTableWidgetItem(str(row[2]))
			lft = QTableWidgetItem(str(row[3]))
			self.tablecombinations.setItem(i, 0, comb)
			self.tablecombinations.setItem(i, 1, supp)
			self.tablecombinations.setItem(i, 2, conf)
			self.tablecombinations.setItem(i, 3, lft)
			
		self.tablecombinations.resizeColumnsToContents()	


		#GENERATING STRONG ASSOCIATION RULES FROM THE FREQUENT ITEMSETS
		dfmodelf = pd.DataFrame()
		lfstrcomb = []
		lfsupport = []
		lfconfidence = []
		lflift = []
		lfstrperm = []
		
		
		for i in range(len(l2comb)):
			for perm in permutations(l2comb[i], len(l2comb[i])):		
				if len(perm) < 3:
					self.support = self.getsupport_list(perm,listcodes)
					self.confidence = self.support / self.getsupport_list(perm[:-1],listcodes)
					self.lift = self.confidence / self.getsupport_list([perm[-1]],listcodes)
							
					if self.confidence > 0.5 and self.lift > 1:		
						lfsupport.append(round(self.support,2))
						lfconfidence.append(round(self.confidence,2))
						lflift.append(round(self.lift,2))
						str_perm = ''.join(str(perm[0])+" --> "+str([perm[1]]))
						lfstrperm.append(str_perm)

				else:
					self.support = self.getsupport_list(perm,listcodes)
					self.confidence = self.support / self.getsupport_list([perm[0]],listcodes)
					self.lift = self.confidence / self.getsupport_list(perm[1:],listcodes)
					
					self.confidence1 = self.support / self.getsupport_list(perm[:-1],listcodes)
					self.lift1 = self.confidence / self.getsupport_list([perm[-1]],listcodes)
							
					if self.confidence > 0.5 and self.lift > 1:		
						lfsupport.append(round(self.support,2))
						lfconfidence.append(round(self.confidence,2))
						lflift.append(round(self.lift,2))
						str_perm = ''.join(str(perm[0])+" --> "+str([perm[1:]]))
						lfstrperm.append(str_perm)	

					if self.confidence1 > 0.5 and self.lift1 > 1:		
						lfsupport.append(round(self.support,2))
						lfconfidence.append(round(self.confidence1,2))
						lflift.append(round(self.lift1,2))
						str_perm = ''.join(str(perm[:-1])+" --> "+str([perm[-1]]))
						lfstrperm.append(str_perm)	

							
		dfmodelf['combination'] = lfstrperm
		dfmodelf['Support'] = lfsupport
		dfmodelf['Confidence'] = lfconfidence
		dfmodelf['Lift'] = lflift
		dfmodelf.titles = ['Combination','Support','Confidence','Lift'] 

		self.tablefinal.setColumnCount(4)
		self.tablefinal.setHorizontalHeaderLabels(['Rule','Support','Confidence','Lift'])
		for i in range(len(dfmodelf)):
			row = dfmodelf.iloc[i]
			self.tablefinal.insertRow(i)
			comb = QTableWidgetItem(str(row[0]))
			supp = QTableWidgetItem(str(row[1]))
			conf = QTableWidgetItem(str(row[2]))
			lft = QTableWidgetItem(str(row[3]))
			self.tablefinal.setItem(i, 0, comb)
			self.tablefinal.setItem(i, 1, supp)
			self.tablefinal.setItem(i, 2, conf)
			self.tablefinal.setItem(i, 3, lft)
			
		self.tablefinal.resizeColumnsToContents()


	def getsupport_list(self,X,datalist):
		N = len(datalist)
		c = 0
		for i in range(len(datalist)):
			if set(X).issubset(set(datalist[i][1])):
				c += 1
		
		support = c / N		

		return support	
			