'''

k-NN: simple k-nearest neighbour classifier implementation with sklearn

This purpose of this code is to educate k-NN classifier in a simplified way.

                    Author  : Viki (a) Vignesh Natarajan
                    Contact : vikiworks.io
'''


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Change this value to True to run in debug mode
#Change this value to False to run in no debug mode
#debug=True
debug=False


''' Step 1 : Training Phase '''

#Independent Variable / Input

    #training_input1  -> represented in x-axis
    #training_input2  -> represented in y-axis

#Dependent Variable / Output

    # training_output -> represented in the 2d plane

''' Step 2 : Testing Phase '''

#Independent Variable / Input

    #test_input1      -> represented in x-axis
    #test_input2      -> represented in y-axis

#Dependent Variable / Output
    # test_output     -> represented in the 2d plane



min_k = 2
max_k = 10
csv1 = None
csv2 = None
training_input = None
training_output = None
test_input = None
test_output = None


def read_csv(file_name, skip_rows=None):
    csv_data = pd.read_csv(file_name, skiprows=skip_rows, sep=',',header=None)
    return csv_data

def print_csv_head(csv_data):
    print csv_data.head()
    print "\n"

def get_column(csv_data, index):
    return csv_data.iloc[:, index].to_numpy()

# club two 1D arrays side by side | ( next to each other )
def club_arrays_side_by_side(array1, array2):
    return np.stack((array1, array2), axis=1)

def create_knn(neighbor):
    model = KNeighborsClassifier(n_neighbors=neighbor)
    return model


def train_knn(model, input, output):
    model.fit(input, output)
    return model

def knn_predict(model, input):
    return model.predict(input)

#Optimal K Value for this input
def knn_compute_optimal_k_value():
    optimal_k = 0
    highest_accuracy = 0
    k_arr = []
    acc_score_arr = []

    for k in range(min_k, max_k):
        model = create_knn(k)
        model = train_knn(model, training_input, training_output)
        predicted_output = knn_predict(model, test_input)
        acc_score = accuracy_score( test_output, predicted_output)

        if acc_score > highest_accuracy:
            highest_accuracy = acc_score
            optimal_k = k

        if debug:
            print "[ k = "+str(k)+" ]  [ accuracy = "+str(acc_score)+"]"

        acc_score_arr.append(acc_score)
        k_arr.append(k)

    if debug:
        print "\n"

    return optimal_k


def knn_plot_k_vs_accuracy():
    k_arr = []
    acc_score_arr = []

    for k in range(min_k, max_k):
        model = create_knn(k)
        model = train_knn(model, training_input, training_output)
        predicted_output = knn_predict(model, test_input)
        acc_score = accuracy_score( test_output, predicted_output)
        acc_score_arr.append(acc_score)
        k_arr.append(k)

    plt.plot(k_arr,acc_score_arr)
    plt.show()

def does_this_person_drink_milk(model, age, height):
    predicted_output = np.array([0])
    new_input=np.array([[age, height]])
    predicted_output = knn_predict(model, new_input)

	print "[ Caution ] This is computer predicted, may or may not be true!"
    if predicted_output[0] == 1:
        print "A person of age ( "+str(age)+" ) and height ( "+str(height)+" cm ) drink milk? Yes"
    else:
        print "A person of age ( "+str(age)+" ) and height ( "+str(height)+" cm ) drink milk? No"


def get_age():
    while True:
        try:
            age = int(input("Enter your age: "))
        except ValueError:
            print("Age should be an integer. Please try again")
        else:
            break

    return age


def get_height():
    while True:
        try:
            height = int(input("Enter your height ( in cm ): "))
        except ValueError:
            print("Height should be an integer. Please try again")
        else:
            break

    return height


print "\n"

csv1 = read_csv("test_age_vs_like_milk.csv", 1)
csv2 = read_csv("training_age_vs_like_milk.csv",1)

if debug:
    print "Sample content for csv1 :\n"
    print_csv_head(csv1)

if debug:
    print "Sample content for csv2 :\n"
    print_csv_head(csv2)

training_input1 = get_column(csv1, 0)
training_input2 = get_column(csv1, 1)
training_output = get_column(csv1, 2)

#Clubbed Training Input
training_input = club_arrays_side_by_side(training_input1, training_input2)

if debug:
    print "Training Input :\n"
    print training_input

if debug:
    print "Training Output :\n"
    print training_output

test_input1 = get_column(csv2, 0)
test_input2 = get_column(csv2, 1)
test_output = get_column(csv2, 2)

#Clubbed Test Input
test_input = club_arrays_side_by_side(test_input1, test_input2)

if debug:
    print "Test Input :\n"
    print test_input

if debug:
    print "Test Output :\n"
    print test_output


optimal_k = knn_compute_optimal_k_value()

print "Optimal K value : "+str(optimal_k)+"\n"

if debug:
    print "Plotting [ K value ] vs [ accuracy score ] "
    knn_plot_k_vs_accuracy()

neighbor = optimal_k
model = create_knn(neighbor)
model = train_knn(model, training_input, training_output)

#Example Input
age=get_age()
height=get_height()
print "\n"

#prediction
does_this_person_drink_milk(model, age, height)



