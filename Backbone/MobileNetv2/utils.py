import requests
import os


def download_file(url, savepath):
    local_filename = savepath  # url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return local_filename


def download_origin(url):
    _file = os.path.dirname(__file__)
    savepath = os.path.join(_file, 'weights/mobilenetv2.pth')

    if os.path.exists(os.path.join(_file, 'weights')):
        os.mkdir(os.path.join(_file, 'weights'))
        os.chmod(0o666)

    if not os.path.exists(savepath):
        print("Original weight is downloaded at ",
              download_file(url, savepath))
    else:
        print('Already download Original weight')

    return savepath


def load_pretrained_weight(model, path_pretrained, watching=True):
    import torch
    # 0. load pretrained state dict
    pretrained_dict = torch.load(path_pretrained)
    # 0.1 get state dict
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    tmp_pretrained_dict = {}
    #

    for (k, v), (mk, mv) in zip(list(pretrained_dict.items()), list(model_dict.items())):
        #if k in model_dict:
        if watching:
            print(f'load==>{k}')
        tmp_pretrained_dict.update({mk: v})
        #else:
        #    if watching:
        #        print(f'unmatch==>{k}')
    # 2. overwrite entries in the existing state dict
    model_dict.update(tmp_pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
