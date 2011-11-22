import math

prop_index = 9
ili_index = 13
med_index = 11

mapping = {
    "Thousand": 0,
    "TenThou": 1,
    "HundredThou": 2,
    "Million": 3
}

if __name__ == "__main__":
    pc = open("PropCostBayesian")
    ili = open("IliCostBayesian")
    med = open("MedCostBayesian")
    training = open("training.csv")

    pc.next()
    ili.next()
    med.next()
    training.next()

    likelihood = 0

    try:
        while training and pc and ili and med:
            propcost = pc.next().rstrip("\n").split(";")
            ilicost = ili.next().rstrip("\n").split(";")
            medcost = med.next().rstrip("\n").split(";")

            t = training.next()
            t = t.rstrip("\n").split(",")

            likelihood += math.log(float(propcost[mapping[t[prop_index]]]))
            likelihood += math.log(float(ilicost[mapping[t[med_index]]]))
            likelihood += math.log(float(medcost[mapping[t[med_index]]]))
    except StopIteration:
        pass
    print likelihood

