from skmultiflow.data import LEDGenerator, ConceptDriftStream, LEDGeneratorDrift
from skmultiflow.drift_detection import ADWIN, PageHinkley, KSWIN, DDM, EDDM
from skmultiflow.lazy import KNNClassifier
from scipy.stats import wilcoxon
import numpy as np

def calculateStatistic(array):
    avg = np.average(array)
    std = np.std(array)
    return avg, std
knn = KNNClassifier()

driftStream = ConceptDriftStream(width=500, position=6500)

n_samples = 0
corrects = 0

coldstartData = []
while n_samples < 2000:
    X, y = driftStream.next_sample()
    my_pred = knn.predict(X)
    if y[0] == my_pred[0]:
        corrects += 1
    knn = knn.partial_fit(X, y)
    n_samples += 1
    coldstartData.append(corrects/n_samples)

print(corrects, n_samples)


# detectors
adwin = ADWIN()
adwin_results = []

ddm = DDM()
ddm_results = []

kswin = KSWIN(alpha=0.02, window_size=600, stat_size=60,data=coldstartData)
kswin_results = []

eddm = EDDM()
eddm_results = []

#dobrac parametry do PH
ph = PageHinkley()
ph_results = []


while n_samples < 8000:
    driftDataX, driftDataY = driftStream.next_sample()
    my_pred = knn.predict(driftDataX)
    correct = driftDataY[0] == my_pred[0]
    if correct:
        corrects += 1
    n_samples += 1

    adwin.add_element(0 if correct else 1)
    if adwin.detected_change():
        print('ADWIN', n_samples)
        adwin_results.append(n_samples)

    ddm.add_element(0 if correct else 1)
    # if ddm.detected_warning_zone():
        # print('DDM warning', n_samples)
    if ddm.detected_change():
        print('DDM', n_samples)
        ddm_results.append(n_samples)

    ph.add_element(0 if correct else 1)
    if ph.detected_change():
        print('PH', n_samples)
        ph_results.append(n_samples)

    kswin.add_element(corrects/n_samples)
    if kswin.detected_change():
        print('KSWIN', n_samples)
        kswin_results.append(n_samples)

    eddm.add_element(0 if correct else 1)
    if eddm.detected_change():
        print('EDDM', n_samples)
        eddm_results.append(n_samples)

print(corrects, n_samples)

detectors = [adwin_results, ddm_results, ph_results, kswin_results, eddm_results]

wil = np.zeros((len(detectors),len(detectors)))

for d1 in range(0,len(detectors)):
    for d2 in range(0,len(detectors)):
        if d1 != d2 and len(detectors[d1]) == len(detectors[d2]) and len(detectors[d1]) > 0:
            wil[d1, d2] = wilcoxon(detectors[d1], detectors[d2])
        else:
            wil[d1, d2] = '-1'
            
print(wil)
exit()
print('ADWIN')
calculateStatistic(adwin_results)
print('DDM')
calculateStatistic(ddm_results)
print('PH')
calculateStatistic(ph_results)
print('KSWIN')
calculateStatistic(kswin_results)
print('EDDM')
calculateStatistic(eddm_results)
