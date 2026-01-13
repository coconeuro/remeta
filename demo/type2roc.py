import numpy as np

def type2roc(correct, conf, nbins=5):
    # Calculate area under type 2 ROC
    #
    # correct - vector of 1 x ntrials, 0 for error, 1 for correct
    # conf - vector of continuous confidence ratings between 0 and 1
    # nbins - how many bins to use for discretization

    bs = 1 / nbins
    H2, FA2 = np.full(nbins, np.nan), np.full(nbins, np.nan)
    for c in range(nbins):
        if c:
            H2[nbins - c - 1] = np.sum((conf > c*bs) & (conf <= (c+1)*bs) & (correct).astype(bool)) + 0.5
            FA2[nbins - c - 1] = np.sum((conf > c*bs) & (conf <= (c+1)*bs) & ~(correct).astype(bool)) + 0.5
        else:
            H2[nbins - c - 1] = np.sum((conf >= c * bs) & (conf <= (c + 1) * bs) & (correct).astype(bool)) + 0.5
            FA2[nbins - c - 1] = np.sum((conf >= c * bs) & (conf <= (c + 1) * bs) & ~(correct).astype(bool)) + 0.5

    H2 /= np.sum(H2)
    FA2 /= np.sum(FA2)
    cum_H2 = np.hstack((0, np.cumsum(H2)))
    cum_FA2 = np.hstack((0, np.cumsum(FA2)))

    k = np.full(nbins, np.nan)
    for c in range(nbins):
        k[c] = (cum_H2[c+1] - cum_FA2[c])**2 - (cum_H2[c] - cum_FA2[c+1])**2

    auroc2 = 0.5 + 0.25*np.sum(k)

    return auroc2

def type2roc_disc(correct, conf, nRatings=5):
    # Ratings have to start with 0!!!
    # Calculate area under type 2 ROC
    #
    # correct - vector of 1 x ntrials, 0 for error, 1 for correct
    # conf - vector of 1 x ntrials of confidence ratings taking values 0:Nratings-1
    # nRatings - how many confidence levels available

    correct = np.array(correct)
    conf = np.array(conf)

    H2, FA2 = np.full(nRatings, np.nan), np.full(nRatings, np.nan)
    for c in range(nRatings):
        H2[nRatings - c - 1] = np.sum((conf == c) & (correct).astype(bool)) + 0.5
        FA2[nRatings - c - 1] = np.sum((conf == c) & ~(correct).astype(bool)) + 0.5

    H2 /= np.sum(H2)
    FA2 /= np.sum(FA2)
    cum_H2 = np.hstack((0, np.cumsum(H2)))
    cum_FA2 = np.hstack((0, np.cumsum(FA2)))

    k = np.full(nRatings, np.nan)
    for c in range(nRatings):
        k[c] = (cum_H2[c+1] - cum_FA2[c])**2 - (cum_H2[c] - cum_FA2[c+1])**2

    auroc2 = 0.5 + 0.25*np.sum(k)

    return auroc2

def type2roc_disc2(stim, response, conf, nRatings=5):
    return type2roc_disc((np.array(stim) == np.array(response)).astype(int), conf, nRatings=nRatings)


def convert_for_mratio(stimuli, correct, confidence, nbins=3):
    #  see https://github.com/metacoglab/HMeta-d/wiki/HMeta-d-tutorial
    #  and http://www.columbia.edu/~bsm2105/type2sdt/archive/index.html

    bs = 1 / nbins
    nR_S = np.full((2, nbins * 2), np.nan)
    for s in range(2):  # iterate over presented stimuli
        for c in range(2):  # iterate over correctness
            conf = confidence[(stimuli == s) & (correct == c)]
            xrange = (range(nbins, 2*nbins), range(nbins-1, -1, -1))[(1, 0)[c] if s else c]
            for i in range(nbins):  # iterate over confidence bins
                if i == 0:
                    nR_S[s, xrange[i]] = np.sum((conf >= i * bs) & (conf <= (i + 1) * bs))
                else:
                    nR_S[s, xrange[i]] = np.sum((conf > i * bs) & (conf <= (i + 1) * bs))

    return nR_S[0], nR_S[1]