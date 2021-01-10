from skmultiflow.data import LEDGenerator, ConceptDriftStream, LEDGeneratorDrift
from skmultiflow.drift_detection import ADWIN, PageHinkley, KSWIN, DDM, EDDM
from skmultiflow.lazy import KNNClassifier
import numpy as np

knn = KNNClassifier()

driftStream = ConceptDriftStream()

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
ddm = DDM()
kswin = KSWIN(alpha=0.005, window_size=300, stat_size=60,data=coldstartData)
eddm = EDDM()
#dobrac parametry do PH
ph = PageHinkley()


while n_samples < 8000:
    driftDataX, driftDataY = driftStream.next_sample()
    my_pred = knn.predict(driftDataX)
    correct = driftDataY[0] == my_pred[0]
    if correct:
        corrects += 1
    n_samples += 1

    adwin.add_element(corrects/n_samples)
    if adwin.detected_change():
        print('ADWIN', n_samples)

    ddm.add_element(0 if correct else 1)
    # if ddm.detected_warning_zone():
        # print('DDM warning', n_samples)
    if ddm.detected_change():
        print('DDM', n_samples)

    ph.add_element(corrects/n_samples)
    if ph.detected_change():
        print('PH', n_samples)

    kswin.add_element(corrects/n_samples)
    if kswin.detected_change():
        print('KSWIN', n_samples)

    eddm.add_element(0 if correct else 1)
    if eddm.detected_change():
        print('EDDM', n_samples)

print(corrects, n_samples)
