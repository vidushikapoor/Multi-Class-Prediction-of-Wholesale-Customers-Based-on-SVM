
# Date: 5th May 2016
# Python Version(s) 2.7.3:


from sklearn import svm
import pdb
import csv
import numpy as np
import matplotlib.pyplot as plt


## Combined Class Creation
class Classdef:
    ##tuple for class name (cn)
    cn = ('0','1','2','3','4','5')
    ## Dictionary
    ##Decode class
    decode_class = {'11': cn[0], '12': cn[1], '13':cn[2], '21':cn[3], '22':cn[4], '23':cn[5]}

## Read data into the CSV format and Clean it
class Readcleandata:
    """Read in cleaned data.
    >>> Readcleandata()
    Traceback (most recent call last):
        ...
    TypeError: __init__() takes exactly 2 arguments (1 given)
    """
    ## Init Function
    def __init__(self,filename1):
        self.filename = filename1

    def cvsformat(self):
        ## For reading the CSV format
        tempcsvfile = open(self.filename,  "r") # opens the file named in tempcsvfile
        tempcsv_reader_obj = csv.reader(tempcsvfile, delimiter=",") # csv.reader reads
        rows = []
        for row in tempcsv_reader_obj:
            ## Refine the Data avoiding any "N/A" or "Empty Data"
            flag = 0
            for cell in row:
                if cell.strip() == 'NA' or len(cell.strip()) == 0:
                    flag = 1
                    break;
            if flag == 0:
                rows.append(row)

        tempcsvfile.close()
        return rows

    ## str function
    def __str__(self):
        return "This is Readcleandata Class which is used to read data from CSV file and clean it"

###Normalization #####
def normalization(array):
    np_data = np.array(array)
    number_atteribute= int(np.size(np_data,1))

    diff_min_max_array = []
    mean_array = []

    for xx in range(0, number_atteribute):
        diff_min_max = max(np_data[:,xx]) - min(np_data[:,xx])
        temp_mean = np.mean(np_data[:,xx])
        diff_min_max_array.append(diff_min_max)
        mean_array.append(temp_mean)

    nornmalized_array = []
    for row in array:
        for xx in range(0, number_atteribute):
            temp = (row[xx] -  mean_array[xx])/diff_min_max_array[xx]
            row[xx] = temp
        nornmalized_array.append(row)

    return nornmalized_array



######### Converting from String to int/float ##########
def convertStr(s):
    """Convert string to an integer or a float and return it.
    >>> convertStr("0")
    0
    >>> convertStr("0.0")
    0.0
    >>> convertStr("0.1")
    0.1
    >>> convertStr("1234")
    1234
    >>> convertStr("-1239.4")
    -1239.4
    """

    if s.isalpha() and s.isalnum():
        return None # s is alphanumeric
    try:
        ret = int(s) # check if an integer
    except ValueError:
        ret = float(s) # must be float

    return ret

### Function with defult value
def data_train_test(array, percent = 60, attribute1 = [0,0, 1,1,1,1,1,1,1], normal = 1):
    
    temp_array_rows_attribute = []
    array_rows_class = []
    for row in array:
         row1 = map(convertStr, row)
         mod_row = []
         for xx in range(0, len(attribute1)):
             if attribute1[xx] == 1:
                mod_row.append(row1[xx])
         temp_array_rows_attribute.append(mod_row[:-1])
         array_rows_class.append(mod_row[-1])


    ########## Normalization ###########
    if normal == 1:
        array_rows_attribute = normalization(temp_array_rows_attribute)
    else:
        array_rows_attribute = temp_array_rows_attribute
    ####################################

    length = len(array_rows_attribute)
    traininglength = int(length*percent/100)

    trainingdata_attribute = array_rows_attribute[:traininglength]
    trainingdata_class = array_rows_class[:traininglength]

    testingdata_attribute  = array_rows_attribute[traininglength:]
    testingdata_class = array_rows_class[traininglength:]

    return trainingdata_attribute, trainingdata_class, testingdata_attribute, testingdata_class


## Combined Class Creation and inheritance
class Multiclass(Classdef):
    """Class for multiclass map.
    >>> Multiclass()
    Traceback (most recent call last):
        ...
    TypeError: __init__() takes exactly 2 arguments (1 given)
    """
    ##init function
    def __init__(self,matrix):
         self.matrix = matrix

    def classmap(self, matrix):
        for row in matrix:
            tempkey = str(row[0]) + str(row[1])
            ##print "TempKey : ", tempkey
            tempclass = self.decode_class.get(tempkey, '999')
            row.append(tempclass)

        return matrix

    ## str function
    def __str__(self):
        return "This is MultiClass Class which defines the Classmap"


### Using __getitem__ 
### Attributes Options (Fresh, Milk, Grocery)
class Attributessub1:
    
    def __init__(self):
        ##Decode class
        ##self.attribute_names = { 1 : 'Fresh', 2: 'Milk' , 3 : 'Grocery' , 4 : 'Frozen' , 5 : 'Detergents', 6 : 'Delicassen'}
        self.attribute_names = { 1 : 'Fresh', 2: 'Milk' , 3 : 'Grocery'}

    
    def __getitem__(self, sliced) :
        tempkey = sliced + 1
        return self.attribute_names[tempkey]

        
### Using __iter__ and __next __ 
## Other Attributs (Frozen, Detergents, Delicassen)  
class Attributessub2:                                                       
    
    def __init__(self):
        self.options = ['Frozen', 'Detergents', 'Delicassen']
        self.length = len(self.options)
        self.index  = 0
        
    def __iter__(self):
        return self
        
    def next(self):  ## Python 3 : def __next__(self)
        if self.index == self.length:
            raise StopIteration
        option_type = self.options[self.index]
        self.index = self.index + 1
        return option_type


############## Coding of the Main Code Starts ###############
############## Coding of the Main Code Starts ###############
############## Coding of the Main Code Starts ###############
############## Coding of the Main Code Starts ###############
if __name__ == "__main__":
    print "doctest running..."
    import doctest
    doctest.testmod()

readcleandata = Readcleandata('Wholesale Customers.csv')
print str(readcleandata)
#readdata = Readdata('copy.csv')
matrix = readcleandata.cvsformat()
#print matrix
#print "Length of matrix : " , len(matrix)

### Class Mapping using Tuple and Dictionary
multiclass = Multiclass(matrix)
print str(multiclass)

finaldata_withheader = multiclass.classmap(matrix)

## Slicing of the Array
finaldata = finaldata_withheader[1:]

##print "Sample Raw data : " , finaldata[0:4]
##print finaldata[0:4]

print "*********************************"
print "*********************************"
print "For this project SVM is used "
print "You have Couple of options to build the model and predict"
print "Please ENTER 1 or 0 for Selection"
print "For Selection 1 == Yes"
print "For Selection 0 == NO"

## Creating the Attribute instance
attsub1 = Attributessub1()                                                       
att_select_sub1 = []
Feature=[]
# for loops call __getitem__ 
print("***Select two attributes to look at the SVM Graph***")
for xx in range(0,len(attsub1.attribute_names.keys())):
    print "*********************************"
    while True:
        try:
            print "Please enter Selection for Attribute --> ", attsub1[xx]
            temp_selection = int(raw_input("Choose an INTEGER 1 == Yes, 0 == NO : "))
            if temp_selection == 1 or temp_selection == 0:
                att_select_sub1.append(temp_selection)
                break
            else:
                print "**** Selection is not Valid, Please choose again an INTEGER 1 == Yes, 0 == NO"
        except:
            print "****  Selection is not Valid, Please choose again an INTEGER 1 == Yes, 0 == NO"     
                                         
if att_select_sub1[0]==1:
    Feature.append("Fresh")
if att_select_sub1[1]==1:
    Feature.append("Milk")
if att_select_sub1[2]==1:
    Feature.append("Grocery")
    

attsub2 = Attributessub2()
att_select_sub2 = []

# for loops call __iter__ and next
for xx in attsub2:
    print "*********************************"
    while True:
        try:
            print "Please enter Selection for Attribute --> ", xx
            temp_selection = int(raw_input("Choose an INTEGER 1 == Yes, 0 == NO : "))
            if temp_selection == 1 or temp_selection == 0:
                att_select_sub2.append(temp_selection)
                break
            else:
                print "**** Selection is not Valid, Please choose again an INTEGER 1 == Yes, 0 == NO"
        except:
            print "****  Selection is not Valid, Please choose again an INTEGER 1 == Yes, 0 == NO"     

if att_select_sub2[0]==1:
    Feature.append("Frozen")
if att_select_sub2[1]==1:
    Feature.append("Detergent")
if att_select_sub2[2]==1:
    Feature.append("Delicassen")

#print "attribut sub1 : ", att_select_sub1
#print "attribut sub2 : ", att_select_sub2
att_select = att_select_sub1 + att_select_sub2

#print "Printing Attribute Selection  : " , att_select 

att_selection = [ 0, 0] + att_select + [1]


print "*********************************"
print "Do you want to NORMALIZE Attribute"
print "Please ENTER 1 or 0 for Selection"
print "For Selection 1 == Yes"
print "For Selection 0 == NO"

while True:
    try:
        Normalize = int(raw_input("Selection for Normalization: "))
        if Normalize == 1 or Normalize == 0:
            break
        else:
            print "Selection is not Valid, Please choose again an integer 1 == Yes, 0 == NO"
    except:
        print "Selection is not Valid, Please choose again an integer 1 == Yes, 0 == NO"   


print "*********************************"
print "*********************************"
print "Total number of samples are : " , len(matrix)
print "*********************************"
print "Please ENTER INTEGER VALUE of % of data use for training (e.g 60 ) "

while True:
    try:
        percentage = int(raw_input("Selection for % of data for Training:  "))
        if percentage == 0 or percentage >= 100:
            print "Selection is not Valid, Please choose again an integer between [0 , 100]"
        else:
            break
    except:
        print "Selection is not Valid, Please choose again an INTEGER between [0 , 100]"



######## Training and Testing Data ##########
data_training,  data_training_class, data_testing, data_testing_class = data_train_test(finaldata, percent = percentage, attribute1 = att_selection, normal = Normalize)

#print "length of training data : ", len(data_training)
#print "length of testing data : ", len(data_testing)
#print "Sample Training data : " , data_training[0:4]
#print "Sample Training Class : " , data_training_class[0:4]

## SVM Model Development

ListX = data_training
X = np.asarray(ListX)
data_train_X = X
Listy = data_training_class
y = np.asarray(Listy)
data_train_y = y
data_test=np.asarray(data_testing)
C = 1.0

print "*********************************"
#print "Please choose the kernel option "
print  "Please choose INTEGER for kernel option (Linear = 0, RBF = 1, POLY = 2, Linear SVC = 3)"
#print "For Selection 1 == Yes"
#print "For Selection 0 == NO"

while True:
    try:
        kernal_option = int(raw_input("Selection for option: "))
        if kernal_option in [0,1,2,3 ] :
            break
        else:
            print "Selection not legal , Please choose again INTEGER (Linear = 0, RBF = 1, POLY = 2, Linear SVC = 3)"
    except:
        print "Selection not legal , Please choose again INTEGER (Linear = 0, RBF = 1, POLY = 2, Linear SVC = 3)"

if kernal_option ==0 :
    print " LINEAR Selection is choosen"
    clf = svm.SVC(kernel='linear').fit(X, y)
elif kernal_option == 1: 
    print " RBF Selction is choosen"
    clf = svm.SVC(kernel='rbf').fit(X, y)
elif kernal_option == 2:
    print " POLY Selction is choosen"
    clf = svm.SVC(kernel='poly').fit(X, y)
elif kernal_option == 3:
    print " SVM LINER Selection is choosen"
    clf = svm.LinearSVC().fit(X, y)



######## Compute the Predicted vs Actual ###########
######## Compute the Predicted vs Actual ###########
predicted_class = clf.predict(data_testing)
##predicted_class = clf.predict(data_training)


output_result_file = "Results_" + "Attribute=" + str(att_select[:])  + "_Normalized=" + str(Normalize) + "_Test=" + str(percentage) + "_.csv"
#print output_result_file
##output_result_file = "Results.csv"
csvwrite = open(output_result_file,'w')
cvs_write_pointer = csv.writer(csvwrite)
cvs_write_pointer.writerow(["Actual Class" , "Predicted Class"])

total_count = 0
total_correct  = 0
total_count_class = [0]*6
total_correct_class = [0]*6

yy = data_testing_class
##yy = data_training_class

for xx in range(0,len(yy)):
    ##Concatenation for writing in the file
    row = [yy[xx], predicted_class[xx]]
    cvs_write_pointer.writerow(row)
    ## Prediction accuracy calculations
    total_count = total_count + 1
    total_count_class[predicted_class[xx]] = total_count_class[predicted_class[xx]] + 1
    if predicted_class[xx] == yy[xx] :
        total_correct = total_correct + 1
        total_correct_class[predicted_class[xx]] = total_correct_class[predicted_class[xx]] + 1



## Closing the File
csvwrite.close()

## Accuracy calculations
total_accuracy = str((float(total_correct)/total_count) * 100)

total_accuracy_class = ['Not Applicable']*6
for xx in range(0,6):
    if total_count_class[xx] != 0:
        total_accuracy_class[xx] = str((float(total_correct_class[xx])/total_count_class[xx]) * 100)



print "\n"
print "*********************************"
print "*********************************"
print "****** RESULTS and SUMMARY ******"                                        
                                                                                
print "*********************************"
print "*********************************"

if kernal_option ==0 :
    print " Type of SVM Kernal : LINEAR"
elif kernal_option == 1: 
    print " Type of SVM Kernal : RBF"
elif kernal_option == 2:
    print " Type of SVM Kernal : POLY"
elif kernal_option == 3:
    print " SVM LINER Selection is choosen"   

print "*********************************"
print(' SVM Model Used = {0}'.format(clf))

if kernal_option != 3 : 
    print '\n Support vectors Generated : \n'
    print clf.support_vectors_
    
    print '\n Indices of support vectors: \n'
    print clf.support_
    
    print '\n Number of support vectors for each class \n'
    print clf.n_support_

print "*********************************"
print "*********************************"
print('{0:*^25s}'.format('SUMMARY OF OPTIONS CHOOSEN'))
print "*********************************"
print " Choose any two attributes to have a plot"
print "*********************************"
print(' ATTRIBUTES : Fresh = {0} , Milk = {1}, Grocery = {2}, Frozen = {3}, Detergent = {4} ,  Delicassen = {5} '.format(att_select[0], att_select[1], att_select[2], att_select[3], att_select[4], att_select[5]))
print " OTHERS     :"
print " Normalization = ", Normalize
print (' Training set = {0} % of Data'.format(percentage))
print "*********************************"
print "*********************************"
print "****** Prediction Matrix ********"
print "*********************************"
if kernal_option ==0 :
    print " Type of SVM Kernal : LINEAR"
elif kernal_option == 1: 
    print " Type of SVM Kernal : RBF"
elif kernal_option == 2:
    print " Type of SVM Kernal : POLY"
elif kernal_option == 3:
    print " SVM LINER Selection is choosen"
print "*********************************"
   
#    decode_class = {'11': cn[0], '12': cn[1], '13':cn[2], '21':cn[3], '22':cn[4], '23':cn[5]}
print('Class 0 (Channel = 1, Region = 1) :: Total count = {0:5d}  || Total correct = {1:5d} || Accuracy  =  {2} % '.format( total_count_class[0] , total_correct_class[0] , total_accuracy_class[0]))
print('Class 1 (Channel = 1, Region = 2) :: Total count = {0:5d}  || Total correct = {1:5d} || Accuracy  =  {2} % '.format( total_count_class[1] , total_correct_class[1] , total_accuracy_class[1]))
print('Class 2 (Channel = 1, Region = 3) :: Total count = {0:5d}  || Total correct = {1:5d} || Accuracy  =  {2} % '.format( total_count_class[2] , total_correct_class[2] , total_accuracy_class[2]))
print('Class 3 (Channel = 2, Region = 1) :: Total count = {0:5d}  || Total correct = {1:5d} || Accuracy  =  {2} % '.format( total_count_class[3] , total_correct_class[3] , total_accuracy_class[3]))
print('Class 4 (Channel = 2, Region = 2) :: Total count = {0:5d}  || Total correct = {1:5d} || Accuracy  =  {2} % '.format( total_count_class[4] , total_correct_class[4] , total_accuracy_class[4]))
print('Class 5 (Channel = 2, Region = 3) :: Total count = {0:5d}  || Total correct = {1:5d} || Accuracy  =  {2} % '.format( total_count_class[5] , total_correct_class[5] , total_accuracy_class[5]))
print "*********************************"
print('Cummulative of All Classes        :: Total count = {0:5d}  || Total correct = {1:5d} || Accuracy  =  {2} % '.format( total_count , total_correct , total_accuracy))
print "*********************************"
print "*********************************"

#print " Actual Class for few samples : ", data_testing_class[15:20]
#print " Predicted Class for few samples : ", predicted_class[15:20]



#### ============= Ploting ================= 

print "*********************************"
print "*********************************"
X = data_train_X
y = data_train_y
 
svc = svm.SVC(kernel='linear').fit(X, y)
rbf_svc = svm.SVC(kernel='rbf').fit(X, y)
poly_svc = svm.SVC(kernel='poly').fit(X, y)
lin_svc = svm.LinearSVC().fit(X, y)

print "LINER : " , svc
print "RBF : ", rbf_svc
print "Poly SVC : ", poly_svc
print "LIN SVC : ", lin_svc


############### Put the check that the if attributes are more than 2 then there will be no plots
############### Also, mention what is Feature 1 and Feature 2.
############### You can use att_select = att_select_sub1 + att_select_sub2 to figure that out


h = .02
if(len(ListX[0])==2):
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    titles = ['SVC with linear kernel',
            'LinearSVC (linear kernel)',
            'SVC with RBF kernel',
            'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    if(len(ListX[0])==2):
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    
        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
        plt.xlabel(Feature[0])
        plt.ylabel(Feature[1])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

plt.show()
