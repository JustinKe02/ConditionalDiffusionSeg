import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from conditional_diffusion.trainer import validate_config

class P1HighNoiseConfigTests(unittest.TestCase):
    def test_registered_p1_is_single_factor(self):
        base=json.loads((ROOT/'configs/pilot.json').read_text())
        config=json.loads((ROOT/'configs/pilot_high_noise.json').read_text())
        validate_config(config)
        diff={key for key in set(base)|set(config) if base.get(key)!=config.get(key)}
        self.assertEqual(diff,{'timestep_sampling','high_noise_start','experiment_variant'})
        self.assertEqual(config['epochs']*120//config['batch_size'],2400)

    def test_reject_unregistered_or_wrong_stage(self):
        base=json.loads((ROOT/'configs/pilot_high_noise.json').read_text())
        for patch in [{'experiment_variant':'unregistered'},{'phase':'P2'},{'method':'direct'},{'debug':True},{'batch_size':3}]:
            config=dict(base,**patch)
            with self.assertRaises(ValueError):validate_config(config)

    def test_existing_configs_keep_working(self):
        for name in ['debug','pilot','formal','p0','p0_high_noise','p0_cosine_lr']:
            validate_config(json.loads((ROOT/'configs'/f'{name}.json').read_text()))

if __name__=='__main__':unittest.main()
