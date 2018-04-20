import matplotlib.pyplot as plt
import matplotlib

x_values = list(range(1,5000))
y_values = [x**3 for x in x_values]
plt.scatter(x_values,y_values,c=y_values,cmap=plt.cm.Blues,edgecolors='none',
            s=40)
plt.show()
