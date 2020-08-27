import sys
from PyQt5.QtWidgets import QMainWindow,QApplication,QAction
from PyQt5 import uic
from PyQt5.QtGui import QIcon, QPalette, QBrush,QPixmap
from salestot import SalesTot
from salescli import SalesCli
from salesprodmonth import Salesprodmonth
from LogReg import LogisticReg
from MBA import MBA
from neuralnets import NeuralNets

class WindowMain(QMainWindow):
 def __init__(self):
    QMainWindow.__init__(self)
    uic.loadUi("main.ui",self)
    self.resize(1000,600)
    
    self.salescli = SalesCli()
    self.salestot = SalesTot()
    self.salesprodmonth = Salesprodmonth()
    self.logReg = LogisticReg()
    self.mba = MBA()
    self.nn = NeuralNets()

    menu = self.menuBar()
    menu_sales = menu.addMenu("&Data Analysis")
  
    #MENU DATA ANALYSIS (SALES X PRODUCT X MONTH)
    menu_salesprodm = QAction(QIcon("../Pics/pie16.png"), "&Montly Product Sales", self)
    menu_salesprodm.setShortcut("Ctrl+o") #Shortcut
    menu_salesprodm.triggered.connect(self.openSalesProdM) #Launcher
    menu_sales.addAction(menu_salesprodm)
    
    #MENU DATA ANALYSIS CLIENT SALES
    menu_salescli = QAction(QIcon("../Pics/statistics16.png"), "&Client Sales", self)
    menu_salescli.setShortcut("Ctrl+w") #Shorcut
    menu_salescli.triggered.connect(self.openSalesCli) #Laucher
    menu_sales.addAction(menu_salescli)
    
    
    #MACHINE LEARNING MENU
    menuML = self.menuBar()
    menu_MachineL = menuML.addMenu("Machine Learning")
    #MENU ventas x prod x mes al menu ventas
    menu_regression = QAction(QIcon("../Pics/regression24.png"), "&Regression Models(Sales)", self)
    menu_regression.setShortcut("Ctrl+r") #Shorcut
    menu_regression.triggered.connect(self.openSalesTot) #Launcher
    menu_MachineL.addAction(menu_regression)
    
    #LOGISTIC REGRESSION MENU
    menulr = QAction(QIcon("../Pics/corrmatrix24.png"), "&Logistic Regression(Credit Evaluation)", self)
    menulr.setShortcut("Ctrl+k") #Shorcut
    menulr.triggered.connect(self.openLogReg) #Launcher
    menu_MachineL.addAction(menulr)

     
     #Market Basket Analysis
    menuMBA = self.menuBar()
    menu_mba = menuMBA.addMenu("Market Basket Analysis")
    menumba = QAction(QIcon("../Pics/cart24.png"), "&MBA - Purchases", self)
    menumba.setShortcut("Ctrl+l") #Shorcut
    menumba.triggered.connect(self.openMBA) #Launcher
    menu_mba.addAction(menumba)

    #NEURAL NETS MENU
    menuNN = self.menuBar()
    menu_NeuralNet = menuNN.addMenu("Neural Nets")
    # Neural Nets for regression
    menup = QAction(QIcon("../Pics/nn24.png"), "&Neural Nets Models", self)
    menup.setShortcut("Ctrl+n") #Shorcut
    menup.triggered.connect(self.openNN) #Launcher
    menu_NeuralNet.addAction(menup)
 
    
 def openSalesCli(self):
    self.salescli.exec_()
    
 def openSalesTot(self):
    self.salestot.exec_()

 def openSalesProdM(self):      
    self.salesprodmonth.exec_()

 def openLogReg(self):
    self.logReg.exec_()
     
 def openMBA(self):
    self.mba.exec_()

 def openNN(self):
    self.nn.exec_()     

    
if __name__== '__main__':
    app = QApplication(sys.argv)
    dialog = WindowMain()
    palette	= QPalette()
    palette.setBrush(QPalette.Background,QBrush(QPixmap("../Pics/Webp.net-resizeimage.jpg")))
    dialog.setPalette(palette)
    dialog.show()
    app.exec_()