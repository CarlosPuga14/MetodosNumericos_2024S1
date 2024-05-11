from List4.Results import resnormCG, resnormCGJ, resnormCGSSOR
from matplotlib import pyplot as plt

plt.figure()
plt.semilogy(range(len(resnormCG)), resnormCG, label = "CG")
plt.semilogy(range(len(resnormCGJ)), resnormCGJ, label = "CG-J")
plt.semilogy(range(len(resnormCGSSOR)), resnormCGSSOR, label = "CG-SSOR")
plt.xlabel("Iterations")
plt.ylabel("Residual norm")
plt.legend(loc = 'center left', bbox_to_anchor=(1, .9))
plt.grid(True)
plt.savefig("Results.pdf", format="pdf", bbox_inches="tight")
plt.show()