import unittest
import numpy as np
import pandas as pd
import xarray as xr
from src.probability_phase4_data import assemble_phase4,predict_frozen,check_gate
from src.probability_complete_features import DAILY,NIGHT,assemble_common_year

class Phase4Tests(unittest.TestCase):
    def test_common_mask_and_holdout(self):
        registry=pd.DataFrame({'latitude':[0,1,2,3,4,5],'longitude':[0]*6,'spatial_role':['train']*3+['holdout']*3})
        variables=list(DAILY)+list(NIGHT)+['night_cooling20to07_c']
        ds=xr.Dataset({n:(('date','latitude','longitude'),np.full((92,6,1),30.)) for n in variables},coords={'date':pd.date_range('2000-06-01',periods=92),'latitude':range(6),'longitude':[0]})
        sources=[[0,1,2]]*3+[[3,4,5]]*3
        old,_,_=assemble_common_year(ds,registry,sources,strict=True)
        new,_,_=assemble_phase4(ds,registry,sources,scope='strict',stage='calibration')
        for k in old: np.testing.assert_array_equal(old[k],new[k])
        hold,_,_=assemble_phase4(ds,registry,sources,scope='strict',stage='final_test')
        self.assertEqual(set(hold['grid_index']),{3,4,5})
        sources[3]=[0,4,5]
        with self.assertRaises(ValueError): assemble_phase4(ds,registry,sources,scope='strict',stage='final_test')
    def test_lr_restoration(self):
        state={'features':['a'],'scaler_mean':[2.],'scaler_scale':[2.],'coefficient':[1.],'intercept':0.}
        p=predict_frozen(state,{'X1':np.array([[2.],[4.]])},['a'],None,2000,1,'temporal','logistic_F0')
        np.testing.assert_allclose(p,[.5,1/(1+np.exp(-1))])
    def test_missing_gate(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=".") as d:
            with self.assertRaises(FileNotFoundError): check_gate(d,'final_test')

if __name__=='__main__': unittest.main()
