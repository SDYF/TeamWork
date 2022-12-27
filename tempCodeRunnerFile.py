set(data_list_path=filename,
                            aug_mode=aug_mode,
                            fea_mode=fea_mode,
                            mode='eval',
                            sample_rate=44100)

    data, __ = dataset.__getitem__(idx=0)
    print(data.shape)