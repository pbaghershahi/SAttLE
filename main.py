import argparse, time, torch, logging, os
import numpy as np
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from utils import fix_seed, get_mappings, get_main, calc_mrr, setup_logger
from model import SAttLE


def main(args):

    def eval_permit(cur_epoch):
        do_eval = False
        if cur_epoch >= args.start_test_at:
            do_eval = True
        return do_eval

    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M')+'-'+args.dataset
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./cache', exist_ok=True)
    log_file_path = './log/'+exec_name+'.log'
    model_state_file = './cache/'+exec_name+'.pth'
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    logger.info(args)
    seed_value = 2411
    fix_seed(seed_value, random_lib=True, numpy_lib=True, torch_lib=True)
    # load graph data
    if args.dataset == 'fb15k-237':
        ds_dir_name = './data/FB15K-237/'
    elif args.dataset == 'wn18rr':
        ds_dir_name = './data/WN18RR/'

    names2ids, rels2ids = get_mappings(
        [ds_dir_name + 'train.txt',
         ds_dir_name + 'valid.txt',
         ds_dir_name + 'test.txt']
    )

    train_data = get_main(
        ds_dir_name,
        'train.txt',
        names2ids,
        rels2ids,
        add_inverse=True
    )['triples']

    valid_data = get_main(
        ds_dir_name,
        'valid.txt',
        names2ids,
        rels2ids,
        add_inverse=False
    )['triples']

    test_data = get_main(
        ds_dir_name,
        'test.txt',
        names2ids,
        rels2ids,
        add_inverse=False
    )['triples']

    n_samples = train_data.shape[0] // 2
    batch_size = args.batch_size // 2
    packed_train_data = [
        np.vstack(
            (train_data[i * batch_size:min((i + 1) * batch_size, n_samples)],
             train_data[i * batch_size + n_samples:min((i + 1) * batch_size, n_samples) + n_samples])
        )
        for i in range(int(np.ceil(n_samples / batch_size)))
    ]

    logger.info(f'train shape: {train_data.shape}, valid shape: {valid_data.shape}, test shape: {test_data.shape}')
    logger.info(f'num entities: {len(names2ids)}, num relations: {len(rels2ids)}')

    num_nodes = len(names2ids.keys())
    num_rels = len(rels2ids.keys())

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    model = SAttLE(
        num_nodes,
        num_rels,
        args.n_layers,
        args.d_embed,
        args.d_k,
        args.d_v,
        args.d_model,
        args.d_inner,
        args.num_heads,
        **{'dr_enc': args.dr_enc,
           'dr_pff': args.dr_pff,
           'dr_sdp': args.dr_sdp,
           'dr_mha': args.dr_mha,
           'decoder': args.decoder}
    )

    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)
    if use_cuda:
        valid_data = valid_data.cuda()
        test_data = test_data.cuda()
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step_decay, gamma=args.lr_decay)
    # scheduler = ExponentialLR(optimizer, args.lr_decay)

    # training loop
    logger.info('start training...')
    best_mrr = 0
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        for edges in packed_train_data:
            model.train()

            edges = torch.as_tensor(edges)

            if use_cuda:
                edges = edges.cuda()

            scores = model(edges)
            loss = model.cal_loss(scores, edges)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients

            optimizer.step()
            optimizer.zero_grad()

        if epoch <= args.decay_until:
            scheduler.step()

        if epoch % 5 == 0:
            logger.info(f'Epoch {epoch:04d} | Loss {loss.item():.7f} | Best MRR {best_mrr:.4f} | Best epoch {best_epoch:04d}')

        # validation
        if eval_permit(epoch):
            with torch.no_grad():
                model.eval()
                logger.info('start eval')
                mrr = calc_mrr(model, torch.LongTensor(train_data[:train_data.shape[0] // 2]).cuda(), valid_data, test_data,
                           hits=[1, 3, 10], logger=logger)
                # save best model
                if best_mrr < mrr:
                    best_mrr = mrr
                    best_epoch = epoch
                    logger.info(f'Epoch {epoch:04d} | Loss {loss.item():.7f} | Best MRR {best_mrr:.4f} | Best epoch {best_epoch:04d}')
                    if epoch >= args.save_epochs:
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

                logger.info('eval done!')

    logger.info('training done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAttLE')
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--lr-decay", type=float, default=1,
                        help="learning rate decay rate")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of encoder layers")
    parser.add_argument("--n-epochs", type=int, default=1500,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use: fb15k-237 or wn18rr")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="number of triples to sample at each iteration")
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")
    parser.add_argument("--start-test-at", type=int, default=7000,
                        help="firs epoch to evaluate on test data for each epoch")
    parser.add_argument("--lr-step-decay", type=int, default=2,
                        help="decay lr every x steps")
    parser.add_argument("--save-epochs", type=int, default=1000,
                        help="save per epoch")
    parser.add_argument("--num-heads", type=int, default=64,
                        help="number of attention heads")
    parser.add_argument('--d-k', default=32, type=int,
                        help='Dimension of key')
    parser.add_argument('--d-v', default=50, type=int,
                        help='Dimension of value')
    parser.add_argument('--d-model', default=75, type=int,
                        help='Dimension of model')
    parser.add_argument('--d-embed', default=75, type=int,
                        help='Dimension of embedding')
    parser.add_argument('--d-inner', default=512, type=int,
                        help='Dimension of inner (FFN)')
    parser.add_argument('--label-smoothing', default=0.1, type=float,
                        help='label smoothing')
    parser.add_argument('--dr-enc', default=0.2, type=float,
                        help='encoder dropout')
    parser.add_argument('--dr-pff', default=0.3, type=float,
                        help='position feedforward dropout')
    parser.add_argument('--dr-sdp', default=0.2, type=float,
                        help='scaled dot product dropout')
    parser.add_argument('--dr-mha', default=0.3, type=float,
                        help='multi-head attention dropout')
    parser.add_argument('--decay-until', default=1050, type=int,
                        help='decay learning rate until')
    parser.add_argument('--decoder', default='twomult', type=str,
                        help='decoder')

    args = parser.parse_args()
    main(args)
