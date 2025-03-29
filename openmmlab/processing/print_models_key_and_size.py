import torch
import os
import os.path as osp

model_path = 'openmmlab/decouple/work_dirs'
saved_path = 'openmmlab/decouple/models_key_and_size'
models = os.listdir(model_path)
for name in models:
    try:
        model = torch.load(osp.join(model_path, name, 'latest.pth'))
        with open(osp.join(saved_path, name + '.txt'), 'w') as f:
            print('#' * 20 + name)
            # print(model.keys())
            if 'state_dict' in model:
                model = model['state_dict']
            for k, v in model.items():
                print_str = '{:<60s}{}'.format(k, v.shape)
                print(print_str)
                f.write(print_str + '\n')
    except:
        print(name)
