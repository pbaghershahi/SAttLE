import torch, random, os, logging
import numpy as np
from collections import OrderedDict
from numpy.random import RandomState


def setup_logger(
    name,
    level=logging.DEBUG,
    stream_handler=True,
    file_handler=True,
    log_file='default.log'
    ):
    open(log_file, 'w').close()
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
        )

    if stream_handler:
        sth = logging.StreamHandler()
        sth.setLevel(level)
        sth.setFormatter(formatter)
        logger.addHandler(sth)

    if file_handler:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def fix_seed(seed_value, random_lib=False, numpy_lib=False, torch_lib=False):
    if random_lib:
        random.seed(seed_value)
    if numpy_lib:
        np.random.seed(seed_value)
    if torch_lib:
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)


def gen_mappings(src_file_paths):
    nodes_name, relations = [], []

    for file_path in src_file_paths:
        with open(file_path, 'r') as file:
            file_lines = file.readlines()

        for line in file_lines:
            line = line.strip()
            src, rel, dst = line.split('\t')
            nodes_name += [src, dst]
            relations.append(rel)

    unique_names, unique_rels = sorted(set(nodes_name)), sorted(set(relations))
    total_names, total_rels = len(unique_names), len(unique_rels)
    names2ids = OrderedDict(zip(unique_names, range(total_names)))
    rels2ids = OrderedDict(zip(unique_rels, range(total_rels)))

    return names2ids, rels2ids


def save_mapping(mapping, mapping_save_path):
    with open(mapping_save_path, 'w') as mapping_file:
        mapping_file.write(str(len(mapping)) + '\n')
        for _name, _id in mapping.items():
            mapping_file.write(str(_name) + '\t' + str(_id) + '\n')


def get_mappings(src_file_paths):
    names2ids, rels2ids = gen_mappings(src_file_paths)
    return names2ids, rels2ids


def get_main_data(src_path, names2ids, rels2ids, dst_path=None, add_inverse=False):
    with open(src_path, 'r') as file:
        file_lines = file.readlines()

    num_entities = len(names2ids.keys())
    num_relations = len(rels2ids.keys())
    src_nodes, rel_types, dst_nodes = [], [], []
    triples = []
    nodes = []
    for line in file_lines:
        line = line.strip()
        src, rel, dst = line.split('\t')
        triples.append([names2ids[src], rels2ids[rel], names2ids[dst]])
    triples = np.array(triples)
    print(add_inverse)
    if add_inverse:
        triples = np.vstack((triples, triples[:, [2, 1, 0]]))
        triples[triples.shape[0] // 2:, 1] += num_relations

    return {
        'total_unique_nodes': len(names2ids.keys()),
        'total_unique_rels': len(rels2ids.keys()),
        'names2ids': names2ids,
        'rels2ids': rels2ids,
        'triples': triples
    }


def get_main(data_dir, file_path, names2ids, rels2ids, add_inverse):
    data_path = os.path.join(data_dir, file_path)
    main_data = get_main_data(data_path, names2ids, rels2ids, add_inverse=add_inverse)
    return main_data


def cal_accuracy(scores, true_labels):
    _, pred_labels = scores.max(dim=1)
    true_labels = true_labels.view(-1)
    return torch.sum(pred_labels == true_labels).item() / true_labels.size(0)



def sample_edge_uniform(n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)


def sample_random_triples(triplets, num_entities, sample_size):
    edges = sample_edge_uniform(len(triplets), sample_size)
    edges = triplets[edges]
    return edges


def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    for o in range(num_entities):
        if ((target_s, target_r, o) not in triplets_to_filter) or ((target_s, target_r, o) == (target_s, target_r, target_o)):
            filtered_o.append(o)
    return torch.LongTensor(filtered_o)


def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    for s in range(num_entities):
        if ((s, target_r, target_o) not in triplets_to_filter) or ((s, target_r, target_o) == (target_s, target_r, target_o)):
            filtered_s.append(s)
    return torch.LongTensor(filtered_s)


def perturb_o_and_get_filtered_rank(model, s, r, o, test_size, triplets_to_filter, logger):
    """ Perturb object in the triplets
    """
    num_entities = model.tr_ent_embedding.shape[0]
    ranks = []
    acc = 0.0
    for idx in range(test_size):
        if idx % 10000 == 0:
            logger.info("test triplet {} / {}, acc {}".format(idx, test_size, acc))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities)
        filtered_o = filtered_o.cuda()
        target_o_idx = int((filtered_o == target_o).nonzero())
        target_s = target_s.repeat(filtered_o.size())[:, None]
        target_r = target_r.repeat(filtered_o.size())[:, None]
        filtered_o = filtered_o[:, None]
        edges = torch.cat((target_s, target_r, filtered_o), dim=1)
        scores, acc = model.predict(edges)
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_o_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def perturb_s_and_get_filtered_rank(model, s, r, o, test_size, triplets_to_filter, logger):
    """ Perturb subject in the triplets
    """
    num_entities = model.tr_ent_embedding.shape[0]
    num_relations = model.rel_embedding.shape[0]//2
    ranks = []
    acc = 0.0
    for idx in range(test_size):
        if idx % 10000 == 0:
            logger.info("test triplet {} / {}, acc {}".format(idx, test_size, acc))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities)
        inv_target_r = target_r + num_relations
        filtered_s = filtered_s.cuda()
        target_s_idx = int((filtered_s == target_s).nonzero())
        target_o = target_o.repeat(filtered_s.size())[:, None]
        inv_target_r = inv_target_r.repeat(filtered_s.size())[:, None]
        filtered_s = filtered_s[:, None]
        edges = torch.cat((target_o, inv_target_r, filtered_s), dim=1)
        scores, acc = model.predict(edges)
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_s_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)


def calc_filtered_mrr(model, train_triplets, valid_triplets, test_triplets, hits=[], logger=None):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        logger.info('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(model, s, r, o, test_size, triplets_to_filter, logger)

        ranks_s += 1 # change to 1-indexed

        mrr_s = torch.mean(1.0 / ranks_s.float())
        logger.info("MRR-S (filtered): {:.6f}".format(mrr_s.item()))

        for hit in hits:
            avg_count = torch.mean((ranks_s <= hit).float())
            logger.info("Hits-S (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

        logger.info('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(model, s, r, o, test_size, triplets_to_filter, logger)

        ranks_o += 1 # change to 1-indexed

        mrr_o = torch.mean(1.0 / ranks_o.float())
        logger.info("MRR-O (filtered): {:.6f}".format(mrr_o.item()))

        for hit in hits:
            avg_count = torch.mean((ranks_o <= hit).float())
            logger.info("Hits-O (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

        ranks = torch.cat([ranks_s, ranks_o])
        # ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        logger.info("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            logger.info("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))

    return mrr.item()


def calc_mrr(model, train_triplets, valid_triplets, test_triplets, hits=[], logger=None):
    mrr = calc_filtered_mrr(model, train_triplets, valid_triplets, test_triplets, hits, logger)
    return mrr

