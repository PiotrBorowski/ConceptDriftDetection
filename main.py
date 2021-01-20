from skmultiflow.data import LEDGenerator, ConceptDriftStream, LEDGeneratorDrift
from skmultiflow.drift_detection import ADWIN, PageHinkley, KSWIN, DDM, EDDM
from skmultiflow.lazy import KNNClassifier

knn = KNNClassifier()

driftStream = ConceptDriftStream(width=500, position=10000, random_state=111)

coldstartData = []
TRAIN_SAMPLES = 2000

trainX, trainY = driftStream.next_sample(TRAIN_SAMPLES)
knn.partial_fit(trainX, trainY)

n_samples = 0
corrects = 0

while n_samples < 4000:
    X, y = driftStream.next_sample()
    my_pred = knn.predict(X)
    if y[0] == my_pred[0]:
        corrects += 1
    knn = knn.partial_fit(X, y)
    n_samples += 1
    coldstartData.append(corrects / n_samples)

print(corrects, n_samples)

# detectors
adwin = ADWIN()
ddm = DDM()
kswin = KSWIN(alpha=0.01, data=coldstartData)
eddm = EDDM()
ph = PageHinkley()

while n_samples < 15000:
    driftDataX, driftDataY = driftStream.next_sample()
    my_pred = knn.predict(driftDataX)
    correct = driftDataY[0] == my_pred[0]
    if correct:
        corrects += 1
    n_samples += 1

    adwin.add_element(0 if correct else 1)
    if adwin.detected_change():
        print('ADWIN', TRAIN_SAMPLES + n_samples)

    ddm.add_element(0 if correct else 1)
    if ddm.detected_change():
        print('DDM', TRAIN_SAMPLES + n_samples)

    ph.add_element(0 if correct else 1)
    if ph.detected_change():
        print('PH', TRAIN_SAMPLES + n_samples)

    kswin.add_element(corrects / n_samples)
    if kswin.detected_change():
        print('KSWIN', TRAIN_SAMPLES + n_samples)

    eddm.add_element(0 if correct else 1)
    if eddm.detected_change():
        print('EDDM', TRAIN_SAMPLES + n_samples)

print(corrects, n_samples)
