"""Exact ordered batch costs for a frozen-trace diagnostic, never a fallback."""
import numpy as np
class StrictStageLookup:
    def __init__(self, features, seconds):
        self.features=np.asarray(features,dtype=float)
        self.seconds=np.asarray(seconds,dtype=float)
        assert len(self.features)==len(self.seconds) and np.all(self.seconds>=0)
        self.index=0
    def predict(self, x):
        a=np.asarray(x,dtype=float);i=self.index
        if a.shape!=(1,self.features.shape[1]) or i>=len(self.seconds):
            raise ValueError('replay exhausted or non-single batch')
        if not np.array_equal(a[0],self.features[i]):
            raise ValueError('replay batch features diverged at row %d'%i)
        self.index+=1
        return self.seconds[i:i+1]
def self_test():
    m=StrictStageLookup([[1.,2.],[3.,4.]],[.1,.2])
    assert m.predict([[1.,2.]])[0]==.1
    try:m.predict([[5.,6.]])
    except ValueError:pass
    else:raise AssertionError('mismatched batch was accepted')
    assert m.index==1 and m.predict([[3.,4.]])[0]==.2
    try:m.predict([[3.,4.]])
    except ValueError:pass
    else:raise AssertionError('missing row was silently accepted')
