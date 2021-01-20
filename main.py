from skmultiflow.data import LEDGenerator, ConceptDriftStream, LEDGeneratorDrift
from skmultiflow.drift_detection import ADWIN, PageHinkley, KSWIN, DDM, EDDM
from skmultiflow.lazy import KNNClassifier
from scipy.stats import wilcoxon
import numpy as np

def calculateStatistic(array):
    avg = np.average(array)
    std = np.std(array)
    return avg, std


randomStates = np.arange(1,25);

DRIFT_CENTRAL = 8000
DRIFT_WIDTH = 1000
DRIFT_BORDER = np.round(DRIFT_CENTRAL - DRIFT_WIDTH/2)


driftStreams = [ConceptDriftStream(width=DRIFT_WIDTH, position=DRIFT_BORDER + 2000, random_state=i) for i in randomStates]

##EKSPERYENT 1
adwin_param = [0.002, 0.005, 0.01]
ddm_param = [3,5,7]
ks_param1 = [100,150,200]
ks_param2 = [30,50,100]
ph_param1 = [25,50,75]
ph_param2 = [0.005,0.01,0.02]

knn = KNNClassifier()

stream = driftStreams[0]

for i in range(0,3):
    trainX, trainY = stream.next_sample(2000)
    knn.partial_fit(trainX, trainY)

    adwin = ADWIN(delta=adwin_param[i])
    ddm = DDM(out_control_level=ddm_param[i])
    kswin1 = KSWIN(window_size=ks_param1[i])
    # kswin2 = KSWIN(stat_size=ks_param2[i])
    ph1 = PageHinkley(threshold=ph_param1[i])
    ph2 = PageHinkley(delta=ph_param2[i])

    adwin_results = []
    ddm_results = []
    kswin1_results = []
    kswin2_results = []
    ph1_results = []
    ph2_results = []

    n_samples = 0
    corrects = 0

    coldstartData = []
    while n_samples < 2000:
        X, y = stream.next_sample()
        my_pred = knn.predict(X)
        if y[0] == my_pred[0]:
            corrects += 1
        knn = knn.partial_fit(X, y)
        n_samples += 1
        coldstartData.append(corrects/n_samples)

    print(corrects, n_samples)



    while n_samples < 20000:
        driftDataX, driftDataY = stream.next_sample()
        my_pred = knn.predict(driftDataX)
        correct = driftDataY[0] == my_pred[0]
        if correct:
            corrects += 1
        n_samples += 1

        adwin.add_element(0 if correct else 1)
        if adwin.detected_change():
            # print('ADWIN', n_samples)
            adwin_results.append(n_samples)

        ddm.add_element(0 if correct else 1)
        if ddm.detected_change():
            # print('DDM', n_samples)
            ddm_results.append(n_samples)

        ph1.add_element(0 if correct else 1)
        if ph1.detected_change():
            # print('PH', n_samples)
            ph1_results.append(n_samples)

        ph2.add_element(0 if correct else 1)
        if ph2.detected_change():
            # print('PH', n_samples)
            ph2_results.append(n_samples)

        kswin1.add_element(corrects/n_samples)
        if kswin1.detected_change():
            # print('KSWIN', n_samples)
            kswin1_results.append(n_samples)

        # kswin2.add_element(corrects / n_samples)
        # if kswin2.detected_change():
        #     print('KSWIN', n_samples)
        #     kswin2_results.append(n_samples)


    print(corrects, n_samples)

    detectors = [adwin_results, ddm_results, ph1_results, ph2_results, kswin1_results]

    for i, results in enumerate(detectors):
        results = np.array(results)
        print("FALSE ALARMS", len(results[results < 8000]))
        print("DETECTION", results[results >= 8000][0] if np.any(results[results >= 8000]) else 'NONE');


#EKSPERYMENT 2

for s in driftStreams:
    s.restart()


# detectors
adwin = ADWIN()
adwin_results = []
adwin_fas = []

ddm = DDM()
ddm_results = []
ddm_fas = []


kswin = KSWIN(alpha=0.02, window_size=600, stat_size=60,data=coldstartData)
kswin_results = []
kswin_fas = []

eddm = EDDM()
eddm_results = []
eddm_fas = []

ph = PageHinkley()
ph_results = []
ph_fas = []

for j, s in enumerate(driftStreams):
    n_samples = 0
    corrects = 0
    coldstartData = []
    knn = KNNClassifier()

    adwin.reset()
    ddm.reset()
    kswin.reset()
    eddm.reset()
    ph.reset()

    adwin_fa = 0
    ddm_fa = 0
    kswin_fa = 0
    eddm_fa = 0
    ph_fa = 0

    trainX, trainY = stream.next_sample(2000)
    knn.partial_fit(trainX, trainY)

    while n_samples < 2000:
        X, y = s.next_sample()
        my_pred = knn.predict(X)
        if y[0] == my_pred[0]:
            corrects += 1
        knn = knn.partial_fit(X, y)
        n_samples += 1
        coldstartData.append(corrects/n_samples)

    print(corrects, n_samples)

    while n_samples < 20000:
        driftDataX, driftDataY = s.next_sample()
        my_pred = knn.predict(driftDataX)
        correct = driftDataY[0] == my_pred[0]
        if correct:
            corrects += 1
        n_samples += 1

        adwin.add_element(0 if correct else 1)
        if adwin.detected_change():
            # print('ADWIN', n_samples)
            if n_samples < DRIFT_BORDER:
                adwin_fa += 1
            if len(adwin_results) <= j and n_samples >= DRIFT_BORDER:
                adwin_results.append(n_samples)

        ddm.add_element(0 if correct else 1)
        if ddm.detected_change():
            # print('DDM', n_samples)
            if n_samples < DRIFT_BORDER:
                ddm_fa += 1
            if len(ddm_results) <= j and n_samples >= DRIFT_BORDER:
                ddm_results.append(n_samples)

        ph.add_element(0 if correct else 1)
        if ph.detected_change():
            # print('PH', n_samples)
            if n_samples < DRIFT_BORDER:
                ph_fa += 1
            if len(ph_results) <= j and n_samples >= DRIFT_BORDER:
                ph_results.append(n_samples)

        kswin.add_element(corrects/n_samples)
        if kswin.detected_change():
            # print('KSWIN', n_samples)
            if n_samples < DRIFT_BORDER:
                kswin_fa += 1
            if len(kswin_results) <= j and n_samples >= DRIFT_BORDER:
                kswin_results.append(n_samples)

        eddm.add_element(0 if correct else 1)
        if eddm.detected_change():
            # print('EDDM', n_samples)
            if n_samples < DRIFT_BORDER:
                eddm_fa += 1
            if len(eddm_results) <= j and n_samples >= DRIFT_BORDER:
                eddm_results.append(n_samples)

    adwin_fas.append(adwin_fa)
    ddm_fas.append(ddm_fa)
    kswin_fas.append(kswin_fa)
    ph_fas.append(ph_fa)
    eddm_fas.append(eddm_fa)

falseAlarms = [adwin_fas, ddm_fas, ph_fas, kswin_fas, eddm_fas]

print(falseAlarms)


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
