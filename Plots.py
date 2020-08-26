import matplotlib.pyplot as plt
from copy import copy

#### Given the scoresList computed by a GridSearchCV call plot the coefficients of the Ridge regressor versus alpha coefficients
def plotCoef(ax,scoresList):
    scoresList=copy(scoresList)
    scoresList.sort(key=lambda e:e["hparams"]["alpha"])
    ax.plot([ x["hparams"]["alpha"] for x in scoresList ],[ y["coefs"] for y in scoresList ])
    ax.set_xlabel("alpha")
    ax.set_ylabel("coeffs")

def plotTestErr(ax,scoresList):
    scoresList=copy(scoresList)
    scoresList.sort(key=lambda e:e["hparams"]["alpha"])
    ax.plot([ x["hparams"]["alpha"] for x in scoresList ],[ y["meanScore"] for y in scoresList ])
    ax.set_xlabel("alpha")
    ax.set_ylabel("meanScore")

def plotPredVariance(ax,scoresList):
    scoresList=copy(scoresList)
    scoresList.sort(key=lambda e:e["hparams"]["alpha"])
    ax.plot([ x["hparams"]["alpha"] for x in scoresList ],[ y["variance"] for y in scoresList ])
    ax.set_xlabel("alpha")
    ax.set_ylabel("variance")
