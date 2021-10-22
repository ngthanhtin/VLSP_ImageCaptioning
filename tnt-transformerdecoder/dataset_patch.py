from common import *
from bms import *
from configure import *
from patch import *
#from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

#(2424186, 6)
#Index(['image_id', 'InChI', 'formula', 'text', 'sequence', 'length'], dtype='object')
def make_fold(mode='train-1'):
    if 'train' in mode:
        df = read_pickle_from_file(data_dir+'/df_train.more.csv.pickle')
        #df_fold = pd.read_csv(data_dir+'/df_fold.csv')
        df_fold = pd.read_csv(data_dir+'/df_fold.fine.csv')
        df_meta = pd.read_csv(data_dir+'/df_train_image_meta.csv')
        df = df.merge(df_fold, on='image_id')
        df = df.merge(df_meta, on='image_id')
        df.loc[:,'path']='train_patch16_s0.800'

        df['fold'] = df['fold'].astype(int)
        #print(df.groupby(['fold']).size()) #404_031
        #print(df.columns)

        fold = int(mode[-1])*10
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_train, df_valid

    # Index(['image_id', 'InChI'], dtype='object')
    if 'test' in mode:
        #df = pd.read_csv(data_dir+'/sample_submission.csv')
        df = pd.read_csv(data_dir+'/submit_lb3.80.csv')
        df_meta = pd.read_csv(data_dir+'/df_test_image_meta.csv')
        df = df.merge(df_meta, on='image_id')

        df.loc[:, 'path'] = 'test'
        #df.loc[:, 'InChI'] = '0'
        df.loc[:, 'formula'] = '0'
        df.loc[:, 'text'] =  '0'
        df.loc[:, 'sequence'] = pd.Series([[0]] * len(df))
        df.loc[:, 'length'] = df.InChI.str.len()

        df_test = df
        return df_test

#-----------------------------------------------------------------------
# tokenization, padding, ...
def pad_sequence_to_max_length(sequence, max_length, padding_value):
    batch_size =len(sequence)
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L, ...] = s
    return pad_sequence

def load_tokenizer():
    tokenizer = YNakamaTokenizer(is_load=True)
    print('len(tokenizer) : vocab_size', len(tokenizer))
    for k,v in STOI.items():
        assert  tokenizer.stoi[k]==v
    return tokenizer




############################################################################################################

def null_augment(r):
    return r


class BmsDataset(Dataset):
    def __init__(self, df, tokenizer, augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        g = self.df['length'].values.astype(np.int32)//20
        g = np.bincount(g,minlength=14)
        string += '\tlength distribution\n'
        for n in range(14):
            string += '\t\t %3d = %8d (%0.4f)\n'%((n+1)*20,g[n], g[n]/g.sum() )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        token = d.sequence

        patch_file = data_dir +'/%s/%s/%s/%s/%s.pickle'%(d.path, d.image_id[0], d.image_id[1], d.image_id[2], d.image_id)
        k = read_pickle_from_file(patch_file)

        patch = uncompress_array(k['patch'])
        patch = np.concatenate([
            np.zeros((1, patch_size+2*pixel_pad, patch_size+2*pixel_pad), np.uint8),
            patch],0) #cls token

        coord  = k['coord']
        w = k['width' ]
        h = k['height']

        h = h // patch_size -1
        w = w // patch_size -1
        coord = np.insert(coord, 0, [h, w], 0) #cls token

        #debug
        # image = patch_to_image(patch, coord, k['width' ], k['height'])
        # image_show('image', image, resize=1)
        # cv2.waitKey(0)
        #image = cv2.imread(image_file,cv2.IMREAD_GRAYSCALE) #

        r = {
            'index'    : index,
            'image_id' : d.image_id,
            'InChI'    : d.InChI,
            'd' : d,
            'token' : token,
            #'image' : image,
            'patch' : patch,
            'coord' : coord,
        }
        if self.augment is not None: r = self.augment(r)
        return r


def null_collate(batch, is_sort_decreasing_length=True):
    collate = defaultdict(list)

    if is_sort_decreasing_length: #sort by decreasing length
        sort  = np.argsort([-len(r['token']) for r in batch])
        batch = [batch[s] for s in sort]

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)
    #----

    batch_size = len(batch)
    collate['length'] = [len(l) for l in collate['token']]

    token  = [np.array(t,np.int32) for t in collate['token']]
    token  = pad_sequence_to_max_length(token, max_length=max_length, padding_value=STOI['<pad>'])
    collate['token'] = torch.from_numpy(token).long()

    max_of_length = max(collate['length'])
    token_pad_mask  = np.zeros((batch_size, max_of_length, max_of_length))
    for b in range(batch_size):
        L = collate['length'][b]
        token_pad_mask [b, :L, :L] = 1 #+1 for cls_token

    collate['token_pad_mask'] = torch.from_numpy(token_pad_mask).byte()
    #-----
    # image = np.stack(collate['image'])
    # image = image.astype(np.float32) / 255
    # collate['image'] = torch.from_numpy(image).unsqueeze(1).repeat(1,3,1,1)

    #-----

    collate['num_patch'] = [len(l) for l in collate['patch']]

    max_of_num_patch = max(collate['num_patch'])
    patch_pad_mask  = np.zeros((batch_size, max_of_num_patch, max_of_num_patch))
    patch = np.full((batch_size, max_of_num_patch, patch_size+2*pixel_pad, patch_size+2*pixel_pad),255) #pad as 255
    coord = np.zeros((batch_size, max_of_num_patch, 2))
    for b in range(batch_size):
        N = collate['num_patch'][b]
        patch[b, :N] = collate['patch'][b]
        coord[b, :N] = collate['coord'][b]
        patch_pad_mask [b, :N, :N] = 1 #+1 for cls_token

    collate['patch'] = torch.from_numpy(patch).half() / 255
    collate['coord'] = torch.from_numpy(coord).long()
    collate['patch_pad_mask' ] = torch.from_numpy(patch_pad_mask).byte()
    return collate



##############################################################################################################




def run_check_dataset():
    tokenizer = load_tokenizer()
    df_train, df_valid = make_fold('train-1')

    # df_train = make_fold('test') #1616107
    # dataset = BmsDataset(df_train, tokenizer, remote_augment)

    dataset = BmsDataset(df_valid, tokenizer)
    print(dataset)


    # for i in range(len(dataset)):
    for i in range(50):
        #i = np.random.choice(len(dataset))
        r = dataset[i]

        print(r['index'])
        print(r['image_id'])
        #print(r['formula'])
        print(r['InChI'])
        print(r['token'])

        print('image : ')
        #print('\t', r['image'].shape)
        print('')

        #---
        image = patch_to_image(r['patch'], r['coord'], width=1024, height=1024)
        image_show('image', image, resize=1)
        cv2.waitKey(0)



    #exit(0)
    loader = DataLoader(
        dataset,
        sampler = RandomSampler(dataset),
        batch_size  = 8,
        drop_last   = True,
        num_workers = 0,
        pin_memory  = True,
        collate_fn  = null_collate,
    )
    for t,batch in enumerate(loader):
        if t>30: break

        print(t, '-----------')
        print('index : ', batch['index'])
        print('image : ')
        #print('\t', batch['image'].shape, batch['image'].is_contiguous())
        print('\t', batch['patch'].shape, batch['patch'].is_contiguous())
        print('\t', batch['coord'].shape, batch['coord'].is_contiguous())
        print('\t', batch['mask'].shape, batch['mask'].is_contiguous())
        print('length  : ')
        print('\t',len( batch['length']))
        print('\t', batch['length'])
        print('token  : ')
        print('\t', batch['token'].shape, batch['token'].is_contiguous())
        print('\t', batch['token'])

        print('')




# main #################################################################
if __name__ == '__main__':
    run_check_dataset()
    #run_check_augment()
