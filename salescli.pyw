import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication,QDialog,QTableWidgetItem,QMessageBox
from connbd import connectdb
import pandas as pd
import matplotlib.pyplot as plt
import chart_studio.plotly as py
import numpy as np
from PyQt5 import uic

class SalesCli(QDialog,QTableWidgetItem):
 def __init__(self,fn=None,parent=None):
#   QDialog.__init__(self)
    super(SalesCli,self).__init__(parent,\
                                   flags=Qt.WindowMinimizeButtonHint|Qt.WindowMaximizeButtonHint|Qt.WindowCloseButtonHint)
    uic.loadUi("salescli.ui",self)
    self.salescli()
    self.btnexcel.clicked.connect(self.export_excel)
    
    
 def export_excel(self):
    try:        
        writer = pd.ExcelWriter('../data/salesxcli_export.xlsx',engine='xlsxwriter')
        self.df.to_excel(writer,sheet_name='salescli_export', header=True, index=False)
        self.dfpivot.to_excel(writer,sheet_name='salespivotcli', header=True, index=False)
        writer.save()
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Successful Export")
        msg.setWindowTitle("Exporting File")
        msg.exec_()
    except RuntimeError as e:
        print('Cannot write document', e)

        
    
 def salescli(self):
    #conn=connectdb()
    #sql = "select concat(substring(fecha,7,4),'-',substring(fecha,4,2)) as periodo, giro, cliente, sum(importe) as Total, sum(utilidad) as Util from listafact group by str_to_date(periodo, '%Y-%m') desc, cliente order by periodo desc,giro"
    
    titles='index Period MSegment Client Total Util Profit(%)'.split()
    #df.to_excel('../data/salesxprod.xls')
    #self.df=pd.read_sql(sql,con=conn)
    #self.df.to_excel('../data/salesxcli.xls')
    self.df=pd.read_excel('../data/salesxcli.xls')
    
    self.df[['Total','Util']] =self.df[['Total','Util']].apply(lambda x: round(x,2))
    self.df['Profit(%)'] = ((self.df['Util'] / self.df['Total'])*100).apply(lambda x: round(x,2))
    
    self.df.columns = titles
   
    #conn.close()
    
    #making datagrid view/table
    self.tablecli.setColumnCount(6)
    self.tablecli.setHorizontalHeaderLabels(['index','Period', 'MSegment', 'Client', 'Total', 'Util', 'Profit(%)'])
              
    for i in range(len(self.df)):
            row = self.df.iloc[i]
            self.tablecli.insertRow(i)
            index = QTableWidgetItem(str(row[0]))
            periodo = QTableWidgetItem(str(row[1]))
            giro = QTableWidgetItem(str(row[2]))
            cliente = QTableWidgetItem(str(row[3]))
            total = QTableWidgetItem(str(row[4]))
            util = QTableWidgetItem(str(row[5]))
            rentab = QTableWidgetItem(str(row[6]))
            self.tablecli.setItem(i, 0, index)
            self.tablecli.setItem(i, 1, periodo)
            self.tablecli.setItem(i, 2, giro)
            self.tablecli.setItem(i, 3, cliente)
            self.tablecli.setItem(i, 4, total)
            self.tablecli.setItem(i, 5, util)
            self.tablecli.setItem(i, 6, rentab)
            
    self.tablecli.resizeColumnsToContents()

    self.dfpivot=self.df.pivot_table(values='Total',index=['MSegment','Client'],columns=['Period'])
    self.dfpivot = self.dfpivot.replace('nan','')
    self.dfpivot = self.dfpivot.reset_index()
    dfpivotcol=self.dfpivot.columns.values.tolist()
    
        
    self.tablaxmes.setColumnCount(len(self.dfpivot.columns))
    self.tablaxmes.setHorizontalHeaderLabels(dfpivotcol)
   
    for i in range(len(self.dfpivot.index)):
        fila = self.dfpivot.iloc[i]
        self.tablaxmes.insertRow(i)
        
        for j in range(len(self.dfpivot.columns)):
            columna = QTableWidgetItem(str(fila[j]))
            self.tablaxmes.setItem(i,j,columna)
            
    
    self.tablaxmes.resizeColumnsToContents()