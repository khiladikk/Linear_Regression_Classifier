from sklearn import linear_model
import matplotlib.pyplot as plt 

features= [[140],[150],[160]]

labels= [1000,1200,1500]

plt.scatter(features, labels, color= 'black')

plt.xlabel('values')
plt.ylabel('price')

clf= linear_model.LinearRegression()
clf= clf.fit(features, labels)
result= clf.predict([[127],[190]])

plt.plot([[130],[195]],result, color='blue', linewidth=2 )

plt.show()