'''
    无需registration，只进行icp
'''
import yaml
from core.dataset import MakePointCloudData
from core.icp import ICP

if __name__ == '__main__':
    cfg_path = 'config/icp.yaml'
    cfg = yaml.load(open(cfg_path, 'r'), Loader = yaml.FullLoader)

    # Dataset
    print('Use Synthetic Data: {}'.format(cfg['data']['synthetic']['use']))

    if cfg['data']['synthetic']['use']:
        data = MakePointCloudData(cfg)
        origin_data, target_data = data.make_target_data()

    else:
        pass

    # ICP
    allow_error = cfg['icp']['allow_error']
    icp = ICP(origin_data, target_data, cfg)
    icp.run_icp()