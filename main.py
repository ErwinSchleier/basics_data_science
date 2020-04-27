import probabilities
import matplotlib.pyplot as plt

xs = [x / 10.0 for x in range(-50, 50)]
for x in xs:
    pdf = probabilities.normal_pdf(x, sigma=1) 
    print(str(x) + ": " + str(pdf))

plt.plot(xs,[probabilities.normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[probabilities.normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[probabilities.normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[probabilities.normal_pdf(x,mu=-1)   for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()

for x in xs:
    cdf = probabilities.normal_cdf(x, sigma=1)
    print(str(x) + ": " + str(cdf))

print("Inverse Normal PDF")
for x in xs:
    i_cdf = probabilities.inverse_normal_cdf(x, sigma=1)
    print(str(i_cdf))

print("***end***")

