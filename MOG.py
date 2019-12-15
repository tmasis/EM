import sys
import random
import scipy.stats as s
import math

clusters = int(sys.argv[3])
maximum = float(sys.argv[5])

values = []
f = open(sys.argv[1],'r')
for line in f: values.append(float(line.strip()))

means = [random.uniform(min(values),max(values)) for x in range(clusters)]
varis = [float(sys.argv[4]) for x in range(clusters)]
mixs = [1.0/clusters for x in range(clusters)]

def print_clusters(p):
    for each in range(len(means)):
        print("Cluster " + str(each+1) + ":\t%.4f\t%.4f\t\t%.4f" % (means[each],
                                                                varis[each], mixs[each]))
    print("Log likelihood: " + str(p) + "\n")

def e_step():
    g = []      # List of responsibility lists per value
    p = 0
    for n in values:
        g_n = []    # Responsibilities of a value per cluster
        g_all = 0
        temp = 0
        for k in range(len(means)):
            prob = mixs[k]*s.norm(means[k],math.sqrt(varis[k])).pdf(n)
            g_n.append(prob)
            temp += prob
        p += math.log(temp)
        g.append(g_n/sum(g_n))

    N = []
    for k in range(len(means)):
        n_all = 0
        for n in g:
            n_all += n[k]
        N.append(n_all)
    return g, N, p

def m_step(g, N):
    oldmixs = list(mixs)
    highest = N.index(max(N))
    for k in range(len(means)):
        s, v = 0, 0
        for n in range(len(values)):
            s += g[n][k]*values[n]
        means[k] = s/N[k]
        for n in range(len(values)):
            v += g[n][k]*(values[n]-means[k])**2
        varis[k] = v/N[k]
        if k == highest:
            mixs[k] = (maximum*len(values)+N[k])/(maximum*len(values)+len(values))
        else:
            mixs[k] = N[k]/(maximum*len(values)+len(values))
    newMeans = []
    newVaris = []
    newMixs = []
    for k in range(len(means)):
        if oldmixs[k] >= 0.05:
            newMeans.append(means[k])
            newVaris.append(varis[k])
            newMixs.append(mixs[k])
    return newMeans, newVaris, newMixs/sum(newMixs)
    
                        
print("Iteration 0:\tMean\tVariance\tMixing Coefficient")
#print_clusters()

for each in range(int(sys.argv[2])):
    g, N, p = e_step()
    means, varis, mixs = m_step(g, N)
    print("Iteration " + str(each+1) + ":")
    print_clusters(p)
