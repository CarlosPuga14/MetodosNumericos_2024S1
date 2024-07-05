from matplotlib import pyplot as plt

data = [0.673542425905569, 0.11149814453502095, 0.006645639492674947, 0.006412141835255881, 0.0021858066887519693]
x = [1, 2, 3, 4, 5]

plt.figure()
plt.semilogy(x, data)
plt.xlabel("Polynomial Order")
plt.ylabel("Error")
plt.grid(True)
plt.show()
